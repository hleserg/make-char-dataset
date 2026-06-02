"""Doctor: validate the heavy-tier training environment before a real Flux run.

The dataset stages (import → generate → clean → caption) run GPU-free on stub
backends, but the ``train`` stage shells out to ai-toolkit and needs a real
environment: the ai-toolkit clone + its venv, an HF token for the gated
**FLUX.1-dev** base, and a built kohya dataset. ``doctor`` reports each
prerequisite so a missing piece is caught *before* a long Flux run starts.

The checks are pure predicates over the resolved settings (path/string tests), so
they unit-test without a GPU; the optional ``--probe`` spawns ai-toolkit's venv to
report the installed torch/CUDA and is the only uncovered part.

A later subtask (HLE-804 restylization generation) extends this with the
generation side — a reachable ComfyUI server exposing the Flux img2img +
ControlNet nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from make_char_dataset.config import get_settings
from make_char_dataset.train import RUN_SCRIPT, dataset_images, resolve_python
from make_char_dataset.workspace import Workspace

if TYPE_CHECKING:
    from make_char_dataset.config import Settings


@dataclass(frozen=True)
class Check:
    """One environment check: a name, its pass/fail, a human detail, and severity."""

    name: str
    ok: bool
    detail: str
    critical: bool = True


def _base_is_local_path(base: str) -> bool:
    """True when ``train_base_model`` names an on-disk file/dir, not an HF repo id.

    HF ids look like ``org/name`` and never exist on disk; a local path either
    contains a path separator and exists, or ends in ``.safetensors``.
    """
    base = base.strip()
    if not base:
        return False
    return Path(base).exists() or base.endswith(".safetensors")


def training_checks(settings: Settings) -> list[Check]:
    """Return the ordered training-env checks for the resolved ``settings`` (pure)."""
    checks: list[Check] = []

    aitk = settings.aitoolkit_dir
    checks.append(Check("ai-toolkit clone", aitk.is_dir(), aitk.as_posix()))

    run_py = aitk / RUN_SCRIPT
    checks.append(Check(f"ai-toolkit {RUN_SCRIPT}", run_py.is_file(), run_py.as_posix()))

    python = Path(resolve_python(settings))
    checks.append(
        Check(
            "trainer interpreter",
            python.is_file(),
            python.as_posix() + ("" if settings.train_python else "  (<aitoolkit_dir>/venv)"),
        )
    )

    base = settings.train_base_model.strip()
    base_is_local = _base_is_local_path(base)
    if base_is_local:
        checks.append(Check("Flux base model", Path(base).exists(), f"local: {base}"))
    else:
        # Gated HF repo → ai-toolkit needs an HF token to download it.
        has_token = bool(settings.huggingface_token.strip())
        checks.append(
            Check(
                "HF token for gated base",
                has_token,
                f"{base} needs HF_TOKEN (gated)" if not has_token else f"present for {base}",
                critical=False,
            )
        )

    workspace = Workspace(settings.workspace)
    dataset_dir = workspace.training_dir(settings.dataset_repeats, settings.trigger_token)
    images = dataset_images(dataset_dir)
    checks.append(
        Check(
            "kohya dataset",
            bool(images),
            f"{len(images)} image(s) in {dataset_dir.as_posix()}"
            if images
            else f"empty/missing {dataset_dir.as_posix()} — run the pipeline first",
            critical=False,
        )
    )
    return checks


def format_report(checks: list[Check]) -> tuple[str, int]:
    """Render checks to a printable report and an exit code (1 if any critical failed)."""
    lines = ["make-char-dataset doctor — training environment", ""]
    failed_critical = 0
    for check in checks:
        if check.ok:
            mark = "✓ PASS"
        elif check.critical:
            mark = "✗ FAIL"
            failed_critical += 1
        else:
            mark = "! WARN"
        lines.append(f"  {mark}  {check.name}: {check.detail}")
    verdict = "GREEN" if failed_critical == 0 else f"RED ({failed_critical} critical failure(s))"
    lines += ["", f"VERDICT: {verdict}"]
    return "\n".join(lines), (1 if failed_critical else 0)


def _probe_torch(settings: Settings) -> Check:  # pragma: no cover - spawns the GPU venv
    """Spawn ai-toolkit's venv to report the installed torch + CUDA availability."""
    import subprocess  # nosec B404 - fixed argv, no shell

    code = (
        "import torch;"
        "print(torch.__version__, torch.cuda.is_available(),"
        "torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda')"
    )
    try:
        result = subprocess.run(  # nosec B603 - resolved interpreter, list-form, shell=False
            [resolve_python(settings), "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return Check("torch / CUDA probe", False, f"could not launch interpreter: {exc}")
    detail = result.stdout.strip() or result.stderr.strip()
    return Check("torch / CUDA probe", result.returncode == 0, detail)


def run_doctor(*, probe: bool = False) -> int:
    """Print the training-env report and return an exit code (CLI entry; thin)."""
    settings = get_settings()
    checks = training_checks(settings)
    if probe:
        checks.append(_probe_torch(settings))
    report, code = format_report(checks)
    print(report)
    return code
