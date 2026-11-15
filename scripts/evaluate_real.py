"""Evaluation on real recordings: drones found, and false alarms on real music, crowds and street sound.

The data is not in this repo (licences differ, and it's ~2 GB). Get it into a folder with:

    salford/    DroneNoise Database, University of Salford (CC BY 4.0), doi:10.17866/rd.salford.22133411
                outdoor flights of 4 drones at 10 m height, past/over a line of 9 ground microphones
    svanstrom/  Drone detection dataset, Svanström et al. (zenodo 5500576): Audio/*.wav, 10 s clips
                of drones, helicopters, airplanes and background
    fma/        Free Music Archive small, one parquet shard (hf: benjamin-paine/free-music-archive-small),
                30 s clips of every genre (tracks are under various Creative Commons licences)
    esc50/      ESC-50 (CC BY-NC 3.0) parquet shards (hf: ashraq/esc50): 5 s everyday sounds

    uv run python scripts/evaluate_real.py ~/src/earshot-data [--quick]

Nothing is written; results are printed.
"""
from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt

from earshot.detect import detect_all

SR = 16000


def mono16k(x: np.ndarray, sr: int) -> np.ndarray:
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        g = np.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    return x.astype(np.float32)


def decode(blob: bytes) -> np.ndarray:
    """Any audio bytes (mp3, wav, flac) to mono 16 kHz, through ffmpeg."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
                         input=blob, capture_output=True).stdout
    return np.frombuffer(out, np.float32).copy()


# -- loaders ---------------------------------------------------------------------------------------
def salford(root: Path):
    """(name, audio) for each flight recording. Names: Ed_<drone>_<height>_<F15 fly-over | H00 hover>_..._M<mic>."""
    for p in sorted((root / "salford").glob("Ed_*.wav")):
        x, sr = sf.read(p, dtype="float32")
        yield p.stem, mono16k(x, sr)


def svanstrom(root: Path):
    """(class, name, audio) from the zip's Audio folder."""
    z = root / "svanstrom" / "dd.zip"
    if not z.exists() or not zipfile.is_zipfile(z):
        return
    with zipfile.ZipFile(z) as zf:
        for n in sorted(zf.namelist()):
            if n.lower().endswith(".wav") and "/audio/" in n.lower():
                cls = re.split(r"[_\d]", Path(n).stem)[0].upper()
                x, sr = sf.read(io.BytesIO(zf.read(n)), dtype="float32")
                yield cls, Path(n).stem, mono16k(x, sr)


def parquet_audio(path: Path, limit: int | None = None):
    import pyarrow.parquet as pq

    t = pq.read_table(path)
    cols = t.column_names
    n = t.num_rows if limit is None else min(limit, t.num_rows)
    for i in range(n):
        row = {c: t.column(c)[i].as_py() for c in cols}
        a = row["audio"]
        yield row, decode(a["bytes"])


def fma(root: Path, limit=None):
    for p in sorted((root / "fma").glob("*.parquet")):
        for row, x in parquet_audio(p, limit):
            yield (row.get("title") or "?"), x


def esc50(root: Path, limit=None):
    for p in sorted((root / "esc50").glob("*.parquet")):
        for row, x in parquet_audio(p, limit):
            yield row.get("category", "?"), x


# -- distance and mixing ---------------------------------------------------------------------------
def far(x: np.ndarray, extra_m: float, base_m: float = 10.0) -> np.ndarray:
    """The same sound from further away: spherical spreading plus air absorption (a gentle low-pass
    that tightens with distance: ~1 dB per 100 m at 1 kHz, ~4 dB at 4 kHz, ISO 9613-1 ballpark)."""
    r = base_m + extra_m
    gain = base_m / r
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    alpha_db_per_m = 0.01 * (f / 1000.0) ** 1.4 / 1.0  # dB per metre
    X *= 10 ** (-alpha_db_per_m * (r - base_m) / 20)
    return (np.fft.irfft(X, len(x)) * gain).astype(np.float32)


BAND = butter(4, [80, 4000], "bandpass", fs=SR, output="sos")


def band_power(x):
    return float(np.mean(sosfilt(BAND, x) ** 2)) + 1e-20


def mix(fg: np.ndarray, bg: np.ndarray, snr_db: float) -> np.ndarray:
    n = len(fg)
    if len(bg) < n:
        bg = np.tile(bg, n // len(bg) + 1)
    bg = bg[:n]
    fg = fg * np.sqrt(band_power(bg) / band_power(fg) * 10 ** (snr_db / 10))
    return fg + bg


def floor_power(x: np.ndarray) -> float:
    """Background level of a recording: the quietest 1 s in the 80–4000 Hz band (the drone far away)."""
    y = sosfilt(BAND, x)
    n = len(y) // SR
    return float(min(np.mean(y[i * SR:(i + 1) * SR] ** 2) for i in range(n))) + 1e-20


def far_in_noise(x: np.ndarray, extra_m: float, ambience: np.ndarray) -> np.ndarray:
    """Move the drone away while the place stays as loud as it was: the recording is attenuated, and
    real outdoor ambience is added back at the recording's own background level."""
    y = far(x, extra_m)
    if extra_m == 0:
        return y
    a = np.tile(ambience, len(y) // len(ambience) + 1)[:len(y)]
    return y + a * np.sqrt(floor_power(x) / band_power(a))


def found(x) -> bool:
    return len(detect_all(x, SR)) > 0


# -- reports ---------------------------------------------------------------------------------------
def report_drones(root: Path, quick: bool):
    rows = defaultdict(lambda: [0, 0])
    clips = []
    for name, x in salford(root):
        m = re.match(r"Ed_(\w+?)_(\d+)_([FH]\d+)_.*_M(\d+)$", name)
        drone, mode, mic = m.group(1), ("hover" if m.group(3).startswith("H") else "fly-over"), int(m.group(4))
        r = rows[(drone, mode)]
        hit = found(x)
        r[0] += hit
        r[1] += 1
        clips.append((drone, mode, mic, x))
    print("Salford DroneNoise (outdoor, 10 m up, ground mics): recordings with an event")
    for (d, mode), (h, n) in sorted(rows.items()):
        print(f"   {d:3s} {mode:8s}: {h}/{n}")
    tot = sum(h for h, _ in rows.values()), sum(n for _, n in rows.values())
    print(f"   all: {tot[0]}/{tot[1]}")

    sv = [(c, n, x) for c, n, x in svanstrom(root)]
    if sv:
        per = defaultdict(lambda: [0, 0])
        for c, _, x in sv:
            per[c][0] += found(x)
            per[c][1] += 1
        print("Svanström (10 s clips, mic on the ground, drones out to ~200 m): clips with an event")
        for c, (h, n) in sorted(per.items()):
            print(f"   {c:12s}: {h}/{n}")
    return clips, [x for c, _, x in sv if c == "DRONE"]


def report_far_and_mixed(root: Path, clips, sv_drones, quick: bool):
    rng = np.random.default_rng(0)
    # strongest mic of each fly-over/hover event: one recording per event
    best = {}
    for d, mode, mic, x in clips:
        best.setdefault((d, mode), []).append(x)
    srcs = [x for xs in best.values() for x in xs[:: (3 if quick else 1)]]
    dists = [0, 40, 90, 190, 290]
    amb = np.concatenate([x for c, x in esc50(root) if c in ("wind", "crickets", "chirping_birds")][:12])
    print("simulated distance (Salford recordings, extra metres added; real outdoor ambience held at the recording's level):")
    for dm in dists:
        h = sum(found(far_in_noise(x, dm, amb)) for x in srcs)
        print(f"   +{dm:3d} m (~{10 + dm} m away): {h}/{len(srcs)}")

    music = [x for _, x in fma(root, limit=40 if quick else 120)]
    crowd = [x for c, x in esc50(root) if c in ("crowd", "laughing", "clapping", "footsteps", "engine", "car_horn", "siren", "church_bells", "train", "wind", "rain")]
    print("drone mixed into real music / street sound, by SNR (80–4000 Hz band):")
    pool = srcs + sv_drones
    for name, bgs in (("music", music), ("street/crowd", crowd)):
        if not bgs:
            continue
        for snr in (6, 0, -6, -12):
            h = 0
            n = 0
            for i, x in enumerate(pool[:: (3 if quick else 1)]):
                bg = bgs[int(rng.integers(len(bgs)))]
                h += found(mix(x, bg, snr))
                n += 1
            print(f"   {name:12s} SNR {snr:+3d} dB: {h}/{n}")


def report_false_alarms(root: Path, quick: bool):
    print("false alarms on real non-drone sound:")
    secs = 0.0
    ev = 0
    worst = []
    for title, x in fma(root, limit=60 if quick else None):
        e = detect_all(x, SR)
        secs += len(x) / SR
        ev += len(e)
        if e:
            worst.append((title, [round(v.f0_hz) for v in e]))
    if secs:
        print(f"   music (FMA, all genres): {ev} events in {secs / 3600:.2f} h  = {ev / (secs / 3600):.1f}/h")
        for t, f in worst[:8]:
            print(f"      {t[:40]:40s} f0 {f}")
    per = defaultdict(lambda: [0, 0.0])
    for c, x in esc50(root):
        e = detect_all(x, SR)
        per[c][0] += len(e)
        per[c][1] += len(x) / SR
    if per:
        tot = sum(v[0] for k, v in per.items() if k not in ("airplane", "helicopter"))
        hrs = sum(v[1] for k, v in per.items() if k not in ("airplane", "helicopter")) / 3600
        print(f"   everyday sounds (ESC-50, excluding airplane/helicopter): {tot} events in {hrs:.2f} h")
        for c, (n, s) in sorted(per.items(), key=lambda kv: -kv[1][0])[:8]:
            if n:
                print(f"      {c:16s}: {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["drones", "far", "fa"], action="append")
    a = ap.parse_args()
    only = set(a.only or ["drones", "far", "fa"])
    clips, svd = report_drones(a.root, a.quick) if only & {"drones", "far"} else ([], [])
    if "far" in only:
        report_far_and_mixed(a.root, clips, svd, a.quick)
    if "fa" in only:
        report_false_alarms(a.root, a.quick)


if __name__ == "__main__":
    sys.exit(main())
