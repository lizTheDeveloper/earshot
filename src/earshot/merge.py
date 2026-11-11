"""Pool several nodes' event logs; where ≥3 positioned nodes heard the same event, estimate a rough area.

The area comes from onset-time differences: sound travels ~343 m/s, so a node 343 m farther away
hears the drone about a second later. Onsets are only known to the detector's hop (0.25 s ≈ 86 m of
sound travel) plus each node's clock error, so this is rough by design. Clocks must be synced (NTP).
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

C = 343.0
SAME_EVENT_S = 5.0
SAME_PITCH = 0.15
GRID_M = 10.0
MARGIN_M = 500.0
HOP_S = 0.25


def _t(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def _same_sound(a: float, b: float) -> bool:
    r = max(a, b) / min(a, b)
    return abs(r - 1) < SAME_PITCH or any(abs(r - m) / m < 0.05 for m in (2, 3))


def cluster(records: list[dict]) -> list[list[dict]]:
    """Group events from different nodes that start within SAME_EVENT_S and are the same sound
    (pitch within 15 %: Doppler differs per node; or an octave apart: nodes can lock onto different
    harmonics). Each node appears at most once per group."""
    recs = sorted(records, key=lambda r: _t(r["start"]))
    groups: list[list[dict]] = []
    for r in recs:
        t = _t(r["start"])
        for g in groups:
            if (t - _t(g[0]["start"]) <= SAME_EVENT_S and r["node"] not in {x["node"] for x in g}
                    and _same_sound(r["f0_hz"], g[0]["f0_hz"])):
                g.append(r)
                break
        else:
            groups.append([r])
    return groups


def _enu(pos: dict[str, tuple[float, float]]):
    lat0 = float(np.mean([p[0] for p in pos.values()]))
    lon0 = float(np.mean([p[1] for p in pos.values()]))
    k = 111_320.0
    to = lambda la, lo: ((lo - lon0) * k * math.cos(math.radians(lat0)), (la - lat0) * k)
    back = lambda x, y: (lat0 + y / k, lon0 + x / (k * math.cos(math.radians(lat0))))
    return to, back


def locate(group: list[dict], positions: dict[str, tuple[float, float]] | None) -> dict:
    """Loudest node always; an area estimate only with ≥3 nodes of known position."""
    loudest = max(group, key=lambda r: r["level_db"])["node"]
    out = {"loudest_node": loudest, "first_node": min(group, key=lambda r: _t(r["start"]))["node"],
           "lat_lon": None, "uncertainty_m": None, "residual_s": None}
    placed = [r for r in group if positions and r["node"] in positions]
    if len(placed) < 3:
        return out
    to, back = _enu(positions)
    xy = np.array([to(*positions[r["node"]]) for r in placed])
    t = np.array([_t(r["start"]) for r in placed])
    t = t - t.min()
    xs = np.arange(xy[:, 0].min() - MARGIN_M, xy[:, 0].max() + MARGIN_M + GRID_M, GRID_M)
    ys = np.arange(xy[:, 1].min() - MARGIN_M, xy[:, 1].max() + MARGIN_M + GRID_M, GRID_M)
    X, Y = np.meshgrid(xs, ys)
    d = np.hypot(X[..., None] - xy[:, 0], Y[..., None] - xy[:, 1])  # (gy, gx, nodes)
    emit = t - d / C  # each node's implied emission time at this point; they agree at the source
    rms = np.sqrt(np.mean((emit - emit.mean(axis=-1, keepdims=True)) ** 2, axis=-1))
    j = np.unravel_index(int(np.argmin(rms)), rms.shape)
    r = float(rms[j])
    out.update(lat_lon=back(float(X[j]), float(Y[j])), residual_s=round(r, 3),
               uncertainty_m=round(C * (r + HOP_S)))
    return out


def merge_logs(paths: list[Path], positions_path: Path | None = None) -> list[dict]:
    records = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    positions = None
    if positions_path:
        positions = {k: (float(v[0]), float(v[1])) for k, v in json.loads(Path(positions_path).read_text()).items()}
    out = []
    for g in cluster(records):
        loc = locate(g, positions)
        out.append({"start": min(r["start"] for r in g), "nodes": sorted(r["node"] for r in g),
                    "f0_hz": round(float(np.median([r["f0_hz"] for r in g])), 1),
                    "confidence": max(r["confidence"] for r in g), **loc})
    return out
