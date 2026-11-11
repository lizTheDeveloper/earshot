import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from conftest import SR, drone, noise
from earshot.cli import main


def _clip(tmp_path, seconds=14, with_drone=True):
    rng = np.random.default_rng(1)
    x = noise(seconds, rng) + (drone(seconds, rng, level=0.01) if with_drone else 0)
    p = tmp_path / "in" / "clip.wav"
    p.parent.mkdir()
    sf.write(p, x, SR)
    return p


def test_listen_to_a_file_prints_and_logs_one_event(tmp_path, capsys):
    p = _clip(tmp_path)
    log = tmp_path / "events.jsonl"
    assert main(["listen", "--file", str(p), "--node", "roof", "--log", str(log), "--start", "2026-11-10T20:00:00Z"]) == 0
    out = capsys.readouterr().out
    assert "drone" in out.lower()
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(recs) == 1 and recs[0]["node"] == "roof" and recs[0]["start"].startswith("2026-11-10T20:00:0")


def test_listen_writes_nothing_but_the_log(tmp_path):
    p = _clip(tmp_path)
    before = {q for q in tmp_path.rglob("*")}
    log = tmp_path / "events.jsonl"
    main(["listen", "--file", str(p), "--log", str(log)])
    after = {q for q in tmp_path.rglob("*")}
    assert after - before == {log}


def test_quiet_file_logs_nothing(tmp_path, capsys):
    p = _clip(tmp_path, with_drone=False)
    log = tmp_path / "events.jsonl"
    assert main(["listen", "--file", str(p), "--log", str(log)]) == 0
    assert not log.exists() or log.read_text() == ""


def test_listen_streams_from_any_block_source():
    from earshot.cli import listen
    rng = np.random.default_rng(2)
    x = noise(14, rng) + drone(14, rng, level=0.01)
    blocks = (x[i:i + 1024] for i in range(0, len(x), 1024))
    got = []
    listen(blocks, SR, node="t", t0=0.0, on_event=got.append)
    assert len(got) == 1 and got[0]["node"] == "t"


def test_no_module_can_touch_the_network_or_save_audio():
    src = Path(__file__).parents[1] / "src" / "earshot"
    text = "\n".join(p.read_text() for p in src.glob("*.py"))
    assert not re.search(r"^\s*(import|from)\s+(socket|http|urllib|requests|asyncio|ssl|ftplib|smtplib)\b", text, re.M)
    assert not re.search(r"(sf|soundfile|wavfile|wave)\.(write|open\([^)]*['\"]w)", text)
    assert ".tofile(" not in text and "np.save" not in text
