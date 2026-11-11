import json
import math

import numpy as np

from earshot.merge import cluster, locate, merge_logs

C = 343.0
LAT0, LON0 = 45.0, -122.0


def _to_latlon(x, y):
    return LAT0 + y / 111_320.0, LON0 + x / (111_320.0 * math.cos(math.radians(LAT0)))


def _iso(t):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="milliseconds")


def _scene(rng, source=(120.0, -80.0), jitter=0.03):
    nodes = {"a": (0, 0), "b": (400, 0), "c": (0, 400), "d": (400, 400)}
    t0 = 1_800_000_000.0
    recs, pos = [], {}
    for n, (x, y) in nodes.items():
        d = math.hypot(x - source[0], y - source[1])
        t = t0 + d / C + rng.normal(0, jitter)
        recs.append({"start": _iso(t), "end": _iso(t + 8), "node": n, "confidence": 0.8, "f0_hz": 180 * (1 + rng.normal(0, 0.01)),
                     "harmonics": 12, "level_db": -20 - 20 * math.log10(max(d, 1) / 10), "wobble": 0.01})
        pos[n] = _to_latlon(x, y)
    return recs, pos, _to_latlon(*source)


def _dist_m(a, b):
    dy = (a[0] - b[0]) * 111_320.0
    dx = (a[1] - b[1]) * 111_320.0 * math.cos(math.radians(LAT0))
    return math.hypot(dx, dy)


def test_four_nodes_locate_the_source_roughly():
    recs, pos, truth = _scene(np.random.default_rng(1))
    [cl] = cluster(recs)
    est = locate(cl, pos)
    assert est is not None and _dist_m(est["lat_lon"], truth) < 60
    assert est["loudest_node"] == "a"


def test_unrelated_events_are_not_pooled():
    recs, pos, _ = _scene(np.random.default_rng(2))
    late = dict(recs[0], node="b", start=_iso(1_800_000_600.0), end=_iso(1_800_000_610.0))
    assert len(cluster(recs + [late])) == 2


def test_two_nodes_give_no_area_only_the_loudest():
    recs, pos, _ = _scene(np.random.default_rng(3))
    [cl] = cluster(recs[:2])
    est = locate(cl, pos)
    assert est["lat_lon"] is None and est["loudest_node"] in ("a", "b")


def test_merge_logs_end_to_end(tmp_path):
    recs, pos, truth = _scene(np.random.default_rng(4))
    paths = []
    for r in recs:
        p = tmp_path / f"{r['node']}.jsonl"
        p.write_text(json.dumps(r) + "\n")
        paths.append(p)
    pp = tmp_path / "positions.json"
    pp.write_text(json.dumps({k: list(v) for k, v in pos.items()}))
    [out] = merge_logs(paths, pp)
    assert out["nodes"] == ["a", "b", "c", "d"] and _dist_m(out["lat_lon"], truth) < 60
