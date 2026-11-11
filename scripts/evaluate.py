"""Synthetic evaluation: detection rate vs SNR, and false alarms per hour on non-drone sounds.

    uv run python scripts/evaluate.py [--trials 20] [--hours 1]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).parents[1] / "tests"))
from conftest import SR, drone, engine, fan, hum, noise, speech_like  # noqa: E402
from earshot.detect import detect_all  # noqa: E402

BAND = butter(4, [80, 4000], "bandpass", fs=SR, output="sos")


def band_power(x):
    return float(np.mean(sosfilt(BAND, x) ** 2))


def detection(trials, snrs):
    print("detection rate (12 s clip, drone present 2–12 s; f0 80–600 Hz, 2–6 rotors, random wobble, half fly past)")
    for snr in snrs:
        hit = 0
        for i in range(trials):
            rng = np.random.default_rng([i, int(snr + 100)])
            bg = noise(12, rng, level=0.02)
            d = drone(10, rng, f0=rng.uniform(80, 600), rotors=int(rng.integers(2, 7)),
                      wobble=rng.uniform(0.004, 0.02), doppler_speed=rng.choice([0, rng.uniform(5, 20)]),
                      closest_m=rng.uniform(15, 60))
            d *= np.sqrt(band_power(bg) / band_power(d) * 10 ** (snr / 10))
            x = bg.copy()
            x[2 * SR:2 * SR + len(d)] += d
            hit += any(e.start_s < 12 and e.end_s > 2 for e in detect_all(x, SR))
        print(f"  SNR {snr:+5.0f} dB: {hit}/{trials}")


def false_alarms(hours):
    rng = np.random.default_rng(42)
    makers = [
        ("noise", lambda s: noise(s, rng)),
        ("speech", lambda s: noise(s, rng) + speech_like(s, rng)),
        ("hum 50", lambda s: noise(s, rng) + hum(s, 50.0)),
        ("hum 60", lambda s: noise(s, rng) + hum(s, 60.0)),
        ("fan", lambda s: noise(s, rng) + fan(s, f0=rng.uniform(80, 400))),
        ("engine", lambda s: noise(s, rng) + engine(s, rng, f0=rng.uniform(25, 160))),
    ]
    seg = 60.0
    n = int(hours * 3600 / seg)
    print(f"false alarms over {hours:g} h of non-drone sound ({seg:.0f} s segments, each kind equally often)")
    per = {k: 0 for k, _ in makers}
    for i in range(n):
        k, mk = makers[i % len(makers)]
        per[k] += len(detect_all(mk(seg), SR))
    for k, v in per.items():
        print(f"  {k:8s}: {v} in {hours / len(makers) * 60:.0f} min")
    print(f"  total: {sum(per.values())} in {hours:g} h")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--hours", type=float, default=1.0)
    a = ap.parse_args()
    detection(a.trials, [-20, -15, -10, -5, 0, 5])
    false_alarms(a.hours)
