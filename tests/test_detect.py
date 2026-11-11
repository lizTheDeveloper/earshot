import numpy as np
import pytest

from conftest import SR, drone, engine, fan, hum, noise, speech_like
from earshot.detect import Detector, Event, detect_all


def run(x):
    return detect_all(x, SR)


def test_hovering_drone_in_noise_is_one_event():
    rng = np.random.default_rng(1)
    x = noise(14, rng) + drone(14, rng, f0=180, level=0.01)
    ev = run(x)
    assert len(ev) == 1
    e = ev[0]
    assert abs(e.f0_hz - 180) / 180 < 0.06 and e.harmonics >= 4 and e.confidence >= 0.5


@pytest.mark.parametrize("f0", [95.0, 320.0, 520.0])
def test_drones_across_the_rotor_range(f0):
    rng = np.random.default_rng(int(f0))
    ev = run(noise(12, rng) + drone(12, rng, f0=f0, level=0.01))
    assert len(ev) == 1 and abs(ev[0].f0_hz - f0) / f0 < 0.06


def test_a_drone_flying_past_is_one_event():
    rng = np.random.default_rng(2)
    ev = run(noise(20, rng) + drone(20, rng, f0=200, level=0.02, doppler_speed=15, closest_m=25))
    assert len(ev) == 1


def test_noise_alone_is_nothing():
    rng = np.random.default_rng(3)
    assert run(noise(60, rng)) == []


def test_talking_is_nothing():
    rng = np.random.default_rng(4)
    assert run(noise(60, rng) + speech_like(60, rng)) == []


@pytest.mark.parametrize("mains", [50.0, 60.0])
def test_mains_hum_is_nothing(mains):
    rng = np.random.default_rng(5)
    assert run(noise(40, rng) + hum(40, mains)) == []


def test_a_piston_engine_is_nothing():
    rng = np.random.default_rng(6)
    assert run(noise(30, rng) + engine(30, rng, f0=110)) == []


def test_a_steady_fan_is_nothing():
    rng = np.random.default_rng(7)
    assert run(noise(40, rng) + fan(40)) == []


def test_drone_heard_over_a_running_fan():
    rng = np.random.default_rng(8)
    x = noise(60, rng) + fan(60)
    x[int(40 * SR):int(54 * SR)] += drone(14, rng, f0=230, level=0.012)
    ev = run(x)
    assert len(ev) == 1 and 38 <= ev[0].start_s <= 44


def test_streaming_in_odd_blocks_matches_one_shot():
    rng = np.random.default_rng(9)
    x = noise(20, rng) + drone(20, rng, f0=150, level=0.01)
    whole = run(x)
    d = Detector(SR)
    got = []
    i = 0
    sizes = rng.integers(1, 9000, 1000)
    for s in sizes:
        got += d.feed(x[i:i + s])
        i += s
        if i >= len(x):
            break
    got += d.flush()
    assert [(round(e.start_s, 2), round(e.f0_hz)) for e in got] == [(round(e.start_s, 2), round(e.f0_hz)) for e in whole]


def test_detector_keeps_only_one_frame_of_audio():
    rng = np.random.default_rng(10)
    d = Detector(SR)
    d.feed(noise(30, rng))
    assert d.buffered_samples() <= SR * 1.25  # never more than a frame plus a hop of audio in memory


def test_event_fields_are_summaries_only():
    rng = np.random.default_rng(11)
    e = run(noise(12, rng) + drone(12, rng, level=0.01))[0]
    assert set(e.to_record("n1", 0.0)) == {"start", "end", "node", "confidence", "f0_hz", "harmonics", "level_db", "wobble"}
