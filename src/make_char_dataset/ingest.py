"""Import a create-char-passport export into the pipeline's ``00_passport_import/``.

The passport set is the character's golden identity canon. Under the project's
*conditioning-only* doctrine these anchors are **not** training images — they
feed the generate stage (HLE-766) as img2img / ControlNet references. This stage
reads the export's ``state.json``, **walks its pointers** (never globs the
folders), role-tags each anchor, validates the five mandatory passport frames,
normalizes + aspect-buckets the images, and lands them under
``00_passport_import/`` with a manifest.

Pure-core: depends only on Pillow + the stdlib; it never imports
torch/diffusers, and it never imports ``create_char_passport`` (the input
contract is mirrored here, not imported), so the CPU/CI path stays light.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from make_char_dataset.config import get_settings
from make_char_dataset.workspace import Workspace

STATE_FILENAME = "state.json"
MANIFEST_FILENAME = "manifest.json"
STAGE_MARKER = ".stage_complete"

# The five mandatory passport frames, in canonical order (mirrors ccp
# ``PASSPORT_STEPS``); all five must be present and approved for an export to be
# importable.
PASSPORT_STEPS: tuple[str, ...] = (
    "passport_face",
    "passport_body",
    "passport_profile",
    "passport_back",
    "passport_3q",
)

# Conditioning role each passport frame plays downstream (the make-char-dataset
# mapping, per the epic): the face is anchored by the front + profile portraits;
# the body by the full-length body / back / three-quarter frames.
_PASSPORT_ROLE: dict[str, str] = {
    "passport_face": "face",
    "passport_profile": "face",
    "passport_body": "body",
    "passport_back": "body",
    "passport_3q": "body",
}


class IngestError(Exception):
    """Raised when a passport export is missing, incomplete, or not ready."""


@dataclass(frozen=True)
class Anchor:
    """A golden reference pointed to by ``state.json``, before normalization."""

    role: str  # face | body | style | outfit | emotion | prop
    key: str  # logical id, e.g. "passport_face", "style", "outfit_rogue_front"
    source: str  # path relative to the export root, exactly as state.json wrote it
    path: Path  # resolved absolute source path


@dataclass(frozen=True)
class ImportedAnchor:
    """An anchor after normalization + landing in ``00_passport_import/``."""

    role: str
    key: str
    source: str
    dest: str  # filename inside 00_passport_import/
    bucket: str  # aspect-bucket dimensions, e.g. "1024x1024"
    width: int
    height: int


@dataclass(frozen=True)
class IngestResult:
    """Outcome of :func:`import_passport`."""

    character_id: str
    import_dir: Path
    anchors: list[ImportedAnchor]
    # Textual identity copied from the export so 00_passport_import/ is
    # self-contained for the generate stage (character_table / prompt_layers /
    # base_outfit) — the generate stage never reaches back to the ccp export.
    identity: dict[str, Any]


def read_state(export_dir: Path) -> dict[str, Any]:
    """Load and parse the export's ``state.json`` into a dict."""
    state_path = export_dir / STATE_FILENAME
    try:
        raw = state_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise IngestError(f"cannot read {state_path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise IngestError(f"invalid JSON in {state_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise IngestError(f"{state_path} must contain a JSON object")
    return data


def is_ready(state: Mapping[str, Any]) -> bool:
    """Return ``True`` when the character is complete (``current_step`` is null)."""
    return state.get("current_step") is None


def _extract_identity(state: Mapping[str, Any]) -> dict[str, Any]:
    """Capture the textual identity the generate stage needs from ``state``.

    Copies ``character_table`` / ``prompt_layers`` / ``base_outfit`` into the
    manifest so ``00_passport_import/`` is self-contained: the generate stage
    builds its prompt from the manifest alone and never re-reads the ccp export.
    """

    def _section(key: str) -> dict[str, Any]:
        value = state.get(key)
        return dict(value) if isinstance(value, Mapping) else {}

    return {
        "character_table": _section("character_table"),
        "prompt_layers": _section("prompt_layers"),
        "base_outfit": _section("base_outfit"),
    }


def aspect_bucket(width: int, height: int, target_side: int) -> tuple[str, tuple[int, int]]:
    """Pick the nearest SDXL aspect bucket for a ``width`` x ``height`` image.

    Returns ``(label, (w, h))`` among square / 4:3 / 3:4 scaled to
    ``target_side``, never upscaling beyond the source's shorter side.
    """
    if width <= 0 or height <= 0:
        raise IngestError(f"invalid image size {width}x{height}")
    short = round(target_side * 3 / 4)
    candidates: dict[str, tuple[int, int]] = {
        "1:1": (target_side, target_side),
        "4:3": (target_side, short),
        "3:4": (short, target_side),
    }
    ratio = width / height
    label = min(candidates, key=lambda key: abs((candidates[key][0] / candidates[key][1]) - ratio))
    bw, bh = candidates[label]
    # No upscale: shrink the bucket to fit inside the source when the source is smaller.
    scale = min(1.0, width / bw, height / bh)
    return label, (max(1, round(bw * scale)), max(1, round(bh * scale)))


def _resolve(export_dir: Path, relative: str) -> Path:
    """Resolve a ``state.json`` pointer against the export root, rejecting escapes."""
    if not relative:
        raise IngestError("empty reference path in state.json")
    root = export_dir.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise IngestError(f"reference path escapes the export folder: {relative!r}")
    return candidate


def _as_list(value: Any) -> list[Any]:
    """Return ``value`` as a list, or an empty list when it is not a sequence."""
    return list(value) if isinstance(value, (list, tuple)) else []


def _optional_anchor(role: str, key: str, ref: str, export_dir: Path) -> Anchor:
    """Resolve and validate an optional (non-passport) reference into an Anchor."""
    path = _resolve(export_dir, ref)
    if not path.is_file():
        raise IngestError(f"{role} reference {ref!r} does not exist")
    return Anchor(role=role, key=key, source=ref, path=path)


def collect_anchors(state: Mapping[str, Any], export_dir: Path) -> list[Anchor]:
    """Walk the pointers in ``state`` (never globbing) and role-tag each anchor.

    Enforces that all five passport frames are present and resolve to real files;
    optional groups (style / outfits / props / emotions) are picked up when their
    pointers are non-null and skipped otherwise. A non-null pointer to a missing
    file is an error — a broken export, not an absent optional.

    # PLAYBOOK-START
    # id: pointer-walk-import
    # title: Import by walking an authoritative manifest, never by globbing
    # status: draft
    # category: pipeline
    # tags: [ingestion, manifest, contract]
    # When an upstream producer writes an index/state file that points at its
    # outputs, enumerate inputs by following those pointers — not by scanning the
    # directory. Globbing invents files the contract never promised and misses the
    # role/identity the pointer carries (which folder name cannot encode).
    # Substitution test passes: any importer fed by a producer's manifest.
    # PLAYBOOK-END
    """
    steps = state.get("steps")
    if not isinstance(steps, Mapping):
        raise IngestError("state.json has no 'steps' table")

    anchors: list[Anchor] = []
    missing: list[str] = []
    for step in PASSPORT_STEPS:
        record = steps.get(step)
        ref = record.get("approved_path") if isinstance(record, Mapping) else None
        if not ref:
            missing.append(step)
            continue
        path = _resolve(export_dir, ref)
        if not path.is_file():
            missing.append(step)
            continue
        anchors.append(Anchor(role=_PASSPORT_ROLE[step], key=step, source=ref, path=path))
    if missing:
        raise IngestError(
            "passport set incomplete; missing or unresolved frames: " + ", ".join(sorted(missing))
        )

    # Optional groups feed extra conditioning. They are imported by *presence of an
    # approved ref*, deliberately NOT gated on ccp's ``*_enabled`` flags: a ref that
    # physically exists is a real anchor under conditioning-only, and a disabled
    # group simply carries null refs. ``base_outfit.ref`` is intentionally skipped —
    # it aliases the already-imported ``passport_body`` frame (the same pointer), so
    # importing it would only duplicate the body anchor under an outfit tag.
    used_keys = {anchor.key for anchor in anchors}

    style_ref = state.get("style_ref")
    if style_ref:
        key = _unique("style", used_keys)
        anchors.append(_optional_anchor("style", key, str(style_ref), export_dir))

    for outfit in _as_list(state.get("outfits")):
        if not isinstance(outfit, Mapping):
            continue
        outfit_id = str(outfit.get("id") or "outfit")
        refs = outfit.get("refs")
        if isinstance(refs, Mapping):
            for slot, ref in refs.items():
                if ref:
                    key = _unique(f"outfit_{outfit_id}_{slot}", used_keys)
                    anchors.append(_optional_anchor("outfit", key, str(ref), export_dir))
        # Per-detail close-ups (OutfitEntry.details[].ref) are unique approved
        # anchors, structurally like prop shots — walk them too, not just refs.
        for index, detail in enumerate(_as_list(outfit.get("details"))):
            ref = detail.get("ref") if isinstance(detail, Mapping) else None
            if ref:
                key = _unique(f"outfit_{outfit_id}_detail_{index}", used_keys)
                anchors.append(_optional_anchor("outfit", key, str(ref), export_dir))

    for prop in _as_list(state.get("props")):
        if not isinstance(prop, Mapping):
            continue
        prop_id = str(prop.get("id") or "prop")
        for index, shot in enumerate(_as_list(prop.get("shots"))):
            ref = shot.get("ref") if isinstance(shot, Mapping) else None
            if ref:
                key = _unique(f"prop_{prop_id}_shot_{index + 1}", used_keys)
                anchors.append(_optional_anchor("prop", key, str(ref), export_dir))

    emotions = state.get("emotions")
    if isinstance(emotions, Mapping):
        base = emotions.get("base_emotion")
        if isinstance(base, Mapping) and base.get("ref"):
            key = _unique("emotion_base", used_keys)
            anchors.append(_optional_anchor("emotion", key, str(base["ref"]), export_dir))
        for index, item in enumerate(_as_list(emotions.get("items"))):
            ref = item.get("ref") if isinstance(item, Mapping) else None
            if ref:
                key = _unique(f"emotion_{index}", used_keys)
                anchors.append(_optional_anchor("emotion", key, str(ref), export_dir))

    return anchors


def _unique(key: str, used: set[str]) -> str:
    """Return a collision-free variant of ``key`` and record it in ``used``."""
    candidate = key
    suffix = 1
    while candidate in used:
        candidate = f"{key}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _dest_name(anchor: Anchor, used: set[str]) -> str:
    """Build a deterministic, collision-free ``<role>_<stem>.png`` output name."""
    base = f"{anchor.role}_{Path(anchor.source).stem}"
    name = f"{base}.png"
    suffix = 1
    while name in used:
        name = f"{base}_{suffix}.png"
        suffix += 1
    return name


def _write_normalized(src: Path, dest: Path, target_side: int) -> tuple[str, tuple[int, int]]:
    """Center-crop ``src`` to its aspect bucket, resize (downscale-only), save PNG.

    A referenced file that exists but does not decode as an image (truncated,
    empty, non-image, or a decompression bomb) is a broken export, surfaced as
    :class:`IngestError` rather than a raw Pillow error.
    """
    try:
        with Image.open(src) as image:
            # Honor EXIF orientation before bucketing so rotated sources are not
            # mis-cropped; ccp PNGs carry none, but the importer accepts any image.
            oriented = ImageOps.exif_transpose(image) or image
            rgb = oriented.convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        # Reference the deterministic dest name + exception type, never the raw
        # source path, so a broken-export error does not carry a filesystem path
        # into Sentry (send_default_pii scrubs request/user data, not exc strings).
        raise IngestError(
            f"reference for {dest.name!r} is not a readable image ({type(exc).__name__})"
        ) from exc
    label, (bucket_w, bucket_h) = aspect_bucket(rgb.width, rgb.height, target_side)
    fitted = ImageOps.fit(rgb, (bucket_w, bucket_h), method=Image.Resampling.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fitted.save(dest, format="PNG")
    return label, (bucket_w, bucket_h)


def _clear_owned(out: Path) -> None:
    """Remove the flat files this stage owns in ``out`` before a rebuild.

    Deliberately conservative: deletes files only, never recursing into or
    removing subdirectories, so an unrelated nested folder is left untouched. The
    stage itself only ever writes flat files, so this fully rebuilds its output.
    """
    for child in out.iterdir():
        if child.is_file():
            child.unlink()


def _write_manifest(
    out: Path, character_id: str, anchors: list[ImportedAnchor], identity: dict[str, Any]
) -> None:
    """Write the deterministic ``manifest.json`` describing the imported anchors."""
    payload = {
        "character_id": character_id,
        "doctrine": "conditioning-only",
        "identity": identity,
        "anchors": [asdict(anchor) for anchor in anchors],
    }
    (out / MANIFEST_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_manifest(out: Path, character_id: str) -> IngestResult:
    """Reconstruct an :class:`IngestResult` from a previously written manifest."""
    try:
        data = json.loads((out / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        anchors = [ImportedAnchor(**entry) for entry in data.get("anchors", [])]
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise IngestError(
            f"completed import at {out} has an unreadable/incompatible manifest "
            f"({exc}); re-run with force=True"
        ) from exc
    identity = data.get("identity", {})
    return IngestResult(
        character_id=str(data.get("character_id", character_id)),
        import_dir=out,
        anchors=anchors,
        identity=identity if isinstance(identity, dict) else {},
    )


def _try_complete_result(out: Path, character_id: str) -> IngestResult | None:
    """Return a prior import's result iff its manifest loads and every file exists."""
    try:
        result = _load_manifest(out, character_id)
    except IngestError:
        return None
    if not all((out / anchor.dest).is_file() for anchor in result.anchors):
        return None
    return result


def import_passport(
    export_dir: Path | str,
    workspace: Workspace,
    *,
    target_side: int = 1024,
    force: bool = False,
) -> IngestResult:
    """Import a passport export into ``workspace.passport_import`` as conditioning.

    Reads ``state.json``, requires ``current_step`` to be null, walks the pointers
    to collect role-tagged anchors, validates the five mandatory passport frames,
    then normalizes + aspect-buckets each image into ``00_passport_import/`` with a
    ``manifest.json``. Idempotent: a *complete* prior import is returned as-is
    unless ``force`` re-runs it; a marker left over a missing/corrupt manifest or
    deleted anchor file self-heals by rebuilding. Writes only inside its own folder.
    """
    export_dir = Path(export_dir)
    state = read_state(export_dir)
    if not is_ready(state):
        raise IngestError(
            f"character is not ready: current_step={state.get('current_step')!r} "
            "(passport unfinished)"
        )
    character_id = str(state.get("character_id") or export_dir.name)
    identity = _extract_identity(state)

    out = workspace.passport_import
    marker = out / STAGE_MARKER
    if marker.exists() and not force:
        cached = _try_complete_result(out, character_id)
        if cached is not None:
            return cached
        # marker present but the import is incomplete/corrupt -> rebuild below

    anchors = collect_anchors(state, export_dir)

    out.mkdir(parents=True, exist_ok=True)
    if force or marker.exists():
        _clear_owned(out)

    imported: list[ImportedAnchor] = []
    used_names: set[str] = set()
    for anchor in anchors:
        dest_name = _dest_name(anchor, used_names)
        used_names.add(dest_name)
        _label, (width, height) = _write_normalized(anchor.path, out / dest_name, target_side)
        imported.append(
            ImportedAnchor(
                role=anchor.role,
                key=anchor.key,
                source=anchor.source,
                dest=dest_name,
                bucket=f"{width}x{height}",
                width=width,
                height=height,
            )
        )

    _write_manifest(out, character_id, imported, identity)
    marker.write_text("ok\n", encoding="utf-8")
    return IngestResult(
        character_id=character_id, import_dir=out, anchors=imported, identity=identity
    )


def run_import(export_dir: Path | str, *, force: bool = False) -> IngestResult:
    """Settings-driven entry point: import ``export_dir`` into the configured workspace."""
    settings = get_settings()
    workspace = Workspace(settings.workspace)
    return import_passport(
        Path(export_dir), workspace, target_side=settings.target_side, force=force
    )
