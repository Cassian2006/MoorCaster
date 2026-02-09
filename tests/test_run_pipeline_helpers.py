from __future__ import annotations

from pathlib import Path

from scripts.run_pipeline import _reset_json_outputs


def test_reset_json_outputs_removes_only_json(tmp_path: Path) -> None:
    out = tmp_path / "outputs" / "yolo"
    out.mkdir(parents=True, exist_ok=True)
    (out / "a.json").write_text("{}", encoding="utf-8")
    (out / "b.txt").write_text("keep", encoding="utf-8")
    sub = out / "sub"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "c.json").write_text("{}", encoding="utf-8")

    removed = _reset_json_outputs(out)
    assert removed == 2
    assert not (out / "a.json").exists()
    assert not (sub / "c.json").exists()
    assert (out / "b.txt").exists()
