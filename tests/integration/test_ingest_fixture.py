"""Integration check of the ingest stage against the real passport fixture.

The fixture is a genuine create-char-passport export kept out of the repo
(``/home/serg/char-passport-fixtures/kael-thornwood``). It is absent in CI, so
this test skips there; locally it proves the adapter handles a real export, not
just synthetic noise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from make_char_dataset.ingest import import_passport
from make_char_dataset.workspace import Workspace

FIXTURE = Path("/home/serg/char-passport-fixtures/kael-thornwood")


@pytest.mark.skipif(
    not (FIXTURE / "state.json").is_file(),
    reason="real passport fixture not present (local-only)",
)
def test_import_real_kael_fixture(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "ws")

    result = import_passport(FIXTURE, ws, target_side=1024)

    assert result.character_id == "kael-thornwood"
    assert len(result.anchors) == 5
    assert sorted(a.role for a in result.anchors) == ["body", "body", "body", "face", "face"]
    for anchor in result.anchors:
        assert (ws.passport_import / anchor.dest).is_file()
        assert anchor.bucket == "1024x1024"
    assert (ws.passport_import / "manifest.json").is_file()
    assert (ws.passport_import / ".stage_complete").is_file()
    # the manifest carries the real character's textual identity for the generate stage
    assert result.identity["character_table"]["name"] == "Kael Thornwood"
    assert result.identity["base_outfit"]["prompt"]
