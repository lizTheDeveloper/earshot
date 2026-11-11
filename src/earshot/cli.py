"""earshot listen [--file F] [--node NAME] [--log FILE] | earshot merge LOG... [--positions F] | earshot devices"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path

import numpy as np

from .detect import CONFIDENCE, Detector


def listen(blocks: Iterable[np.ndarray], sr: int, *, node: str, t0: float,
           on_event: Callable[[dict], None], threshold: float = CONFIDENCE) -> int:
    """Feed audio blocks (mono or multichannel float) through the detector. Audio is dropped as soon as
    it has been analysed; only event summaries reach on_event. Returns the number of events."""
    d = Detector(sr, threshold)
    n = 0
    try:
        for b in blocks:
            b = np.asarray(b, np.float32)
            if b.ndim == 2:
                b = b.mean(axis=1)
            for e in d.feed(b):
                on_event(e.to_record(node, t0))
                n += 1
    finally:
        for e in d.flush():
            on_event(e.to_record(node, t0))
            n += 1
    return n


def _file_blocks(path: Path):
    import soundfile as sf

    info = sf.info(str(path))
    return info.samplerate, sf.blocks(str(path), blocksize=4096, dtype="float32", always_2d=True)


def _mic_blocks(device, sr: int | None):
    """Blocks from the microphone. They live in a short queue in memory and nowhere else."""
    import queue

    import sounddevice as sd

    rate = int(sr or sd.query_devices(device, "input")["default_samplerate"])
    q: queue.Queue = queue.Queue(maxsize=64)

    def cb(indata, frames, t, status):  # runs on the audio thread
        try:
            q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass  # we're behind: drop audio rather than pile it up

    def gen():
        with sd.InputStream(device=device, channels=1, samplerate=rate, blocksize=2048, dtype="float32", callback=cb):
            while True:
                yield q.get()

    return rate, gen()


def _printer(log: Path | None):
    fh = open(log, "a", buffering=1) if log else None

    def on_event(r: dict) -> None:
        when = datetime.fromisoformat(r["start"]).astimezone().strftime("%H:%M:%S")
        print(f"{when}  possible drone  {r['f0_hz']:.0f} Hz, {r['harmonics']} harmonics, "
              f"confidence {r['confidence']:.2f}, level {r['level_db']:.0f} dBFS", flush=True)
        if fh:
            fh.write(json.dumps(r) + "\n")

    return on_event, fh


def _t0(arg: str | None) -> float:
    if not arg:
        return time.time()
    return datetime.fromisoformat(arg.replace("Z", "+00:00")).timestamp()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="earshot", description="Notices when a small drone is nearby. Keeps no audio.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    li = sub.add_parser("listen", help="listen to the microphone (or a file) and report possible drones")
    li.add_argument("--file", type=Path, help="analyse an audio file instead of the microphone")
    li.add_argument("--node", default="earshot", help="a name for this listener, for pooling logs (default: earshot)")
    li.add_argument("--log", type=Path, help="append events as JSON lines to this file")
    li.add_argument("--threshold", type=float, default=CONFIDENCE, help=f"confidence needed to report (default {CONFIDENCE})")
    li.add_argument("--device", help="input device number or name (see: earshot devices)")
    li.add_argument("--rate", type=int, help="microphone sample rate (default: the device's)")
    li.add_argument("--start", help="for --file: the UTC time the recording started (ISO 8601); default now")
    me = sub.add_parser("merge", help="pool several listeners' logs and estimate where each drone was")
    me.add_argument("logs", nargs="+", type=Path)
    me.add_argument("--positions", type=Path, help='JSON: {"node": [lat, lon], ...}')
    me.add_argument("--json", action="store_true", help="print JSON instead of text")
    sub.add_parser("devices", help="list microphones")
    a = ap.parse_args(argv)

    if a.cmd == "devices":
        import sounddevice as sd
        print(sd.query_devices())
        return 0
    if a.cmd == "merge":
        from .merge import merge_logs
        out = merge_logs(a.logs, a.positions)
        if a.json:
            print(json.dumps(out, indent=2))
        else:
            for g in out:
                where = (f"around {g['lat_lon'][0]:.5f}, {g['lat_lon'][1]:.5f} (± ~{g['uncertainty_m']} m, rough)"
                         if g["lat_lon"] else "not enough placed listeners for an area")
                print(f"{g['start']}  {g['f0_hz']:.0f} Hz  heard by {', '.join(g['nodes'])}; "
                      f"loudest at {g['loudest_node']}; {where}")
        return 0

    on_event, fh = _printer(a.log)
    try:
        if a.file:
            sr, blocks = _file_blocks(a.file)
            listen(blocks, sr, node=a.node, t0=_t0(a.start), on_event=on_event, threshold=a.threshold)
        else:
            dev = int(a.device) if a.device and a.device.isdigit() else a.device
            sr, blocks = _mic_blocks(dev, a.rate)
            print(f"listening ({sr} Hz). Nothing is recorded; Ctrl-C to stop.", file=sys.stderr)
            # mic blocks start now; the detector's clock counts from here
            listen(blocks, sr, node=a.node, t0=time.time(), on_event=on_event, threshold=a.threshold)
    except KeyboardInterrupt:
        pass
    finally:
        if fh:
            fh.close()
    return 0
