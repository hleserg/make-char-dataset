"""Train stage: train a character LoRA from the kohya dataset via **ai-toolkit**.

Reads the kohya dataset folder ``03_dataset/<repeats>_<trigger>/`` and trains a
character LoRA into ``06_lora/<name>/<name>.safetensors`` by shelling out to a
local clone of `ostris/ai-toolkit <https://github.com/ostris/ai-toolkit>`_.

Why ai-toolkit and not kohya sd-scripts (which the sibling style pipeline uses)?
The char-LoRA must train on **FLUX.1-dev** so it stacks with the cmcstyle style
LoRA (``Flux + cmcstyle + <char>_char``, HLE-802). kohya loads the whole Flux DiT
to the GPU before block-swap offload and OOMs on ~16 GB; it has no sub-fp8
*training* quantization. ai-toolkit can quantize the transformer + text encoder to
**qint4** with ``low_vram``, which fits ~16 GB — the part kohya cannot do.

Pure-core-lazy-backend (the same split as ``generate.py``/``caption.py``): the
ai-toolkit config builder, the launch-command builder and the stdout progress
parser are pure and unit-tested with a ``FakeTrainer``; only the subprocess launch
(:class:`AiToolkitTrainer`) is uncovered. The heavy training runs in ai-toolkit's
*own* (root-owned, cu128/sm_120) venv that we only spawn — never import — so its
torch/diffusers pins stay isolated from this package's numpy/Pillow.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from make_char_dataset.config import get_settings
from make_char_dataset.workspace import Workspace

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from make_char_dataset.config import Settings

NAME = "train"
CONFIG_FILENAME = "aitoolkit_config.json"
TRAIN_LOG_FILENAME = "train.log"
RUN_SCRIPT = "run.py"

#: Training toolchains with a wired config/launch builder.
SUPPORTED_TOOLS = frozenset({"ai-toolkit"})

#: Images the dataset folder must contain for training to be worth launching.
_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp"})

# ai-toolkit's progress bar is tqdm with ``desc=<name>`` and a postfix string
# ``lr: 1.0e-04 loss: 9.870e-02`` (set via ``set_postfix_str``). Anchor the step
# match on the presence of ``lr:``/``loss:`` so the latent-caching bar — which
# also prints ``N/total`` but no lr/loss — never false-matches.
_STEP_RE = re.compile(r"(\d+)/(\d+)")
_LOSS_RE = re.compile(r"loss:\s*([0-9.eE+\-]+)")
_LR_RE = re.compile(r"lr:\s*([0-9.eE+\-]+)")


class TrainError(Exception):
    """Raised when the train stage cannot run (bad config, no dataset, …)."""


@dataclass(frozen=True)
class TrainProgress:
    """A parsed line of ai-toolkit training progress (the tqdm step bar)."""

    step: int | None = None
    total_steps: int | None = None
    loss: float | None = None
    lr: float | None = None

    @property
    def fraction(self) -> float | None:
        """Completed fraction in ``[0, 1]`` from the step counter, or ``None``."""
        if self.step is None or not self.total_steps:
            return None
        return self.step / self.total_steps


@dataclass(frozen=True)
class TrainPlan:
    """A fully-resolved training invocation: what to run, where, what it writes."""

    command: list[str]
    cwd: Path
    config_path: Path
    config_json: str
    output_path: Path


class Trainer(Protocol):
    """Runs a :class:`TrainPlan` to completion and returns the produced LoRA path."""

    def train(self, plan: TrainPlan) -> Path:
        """Run training; return the produced ``.safetensors``. Raise on failure."""
        ...


# PLAYBOOK-START
# id: pure-core-lazy-backend
# title: Pure policy core behind a lazily-spawned heavy backend
# status: draft
# category: testability
# tags: [testing, dependency-injection, coverage, subprocess]
# The config/launch/parse logic is pure and unit-tested; only the subprocess
# launch stays uncovered. The heavy trainer lives in a separate venv we shell out
# to — never imported — so its conflicting deps can't reach our environment.
# PLAYBOOK-END
class AiToolkitTrainer:
    """Launch ai-toolkit's ``run.py`` as a subprocess in its own (cu128/sm_120) venv."""

    def __init__(
        self,
        hf_token: str = "",  # nosec B107 - empty default; the real token is injected from settings
        on_progress: Callable[[TrainProgress], None] | None = None,
    ) -> None:
        self._hf_token = hf_token
        self._on_progress = on_progress

    def train(self, plan: TrainPlan) -> Path:  # pragma: no cover - spawns a GPU subprocess
        """Write the config, run ai-toolkit, stream progress, verify the LoRA was written."""
        import os
        import shlex
        import subprocess  # nosec B404 - trusted, list-form, shell=False command

        plan.config_path.parent.mkdir(parents=True, exist_ok=True)
        plan.config_path.write_text(plan.config_json, encoding="utf-8")

        env = build_subprocess_env(os.environ, self._hf_token)
        process = subprocess.Popen(  # nosec B603 - fixed argv from validated settings, no shell
            plan.command,
            cwd=plan.cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # tqdm writes to stderr; merge it in
            text=True,
            bufsize=1,
            env=env,
        )
        stdout = process.stdout
        if stdout is None:  # PIPE always yields a stream; satisfy the type checker
            raise RuntimeError("failed to capture ai-toolkit output")
        for line in stdout:
            stripped = line.rstrip()
            print(stripped)
            progress = parse_progress(stripped)
            if progress is not None and self._on_progress is not None:
                self._on_progress(progress)
        code = process.wait()
        if code != 0:
            raise RuntimeError(
                f"ai-toolkit training failed (exit {code}): {shlex.join(plan.command)}"
            )
        if not (plan.output_path.is_file() and plan.output_path.stat().st_size > 0):
            raise RuntimeError(
                f"training finished but {plan.output_path} is missing or empty "
                f"(see {plan.config_path.parent / TRAIN_LOG_FILENAME} and the log above)."
            )
        return plan.output_path


def make_trainer(
    settings: Settings, on_progress: Callable[[TrainProgress], None] | None = None
) -> Trainer:
    """Return the training backend for the configured tool (only ai-toolkit is wired)."""
    tool = settings.train_tool.strip().lower()
    if tool not in SUPPORTED_TOOLS:
        raise TrainError(
            f"train_tool={tool!r} is not supported; set APP_TRAIN_TOOL=ai-toolkit "
            f"(the only wired toolchain)."
        )
    return AiToolkitTrainer(hf_token=settings.huggingface_token, on_progress=on_progress)


# --- Pure: ai-toolkit config, launch command, progress parsing (no torch/subprocess) ---


def build_subprocess_env(base_env: Mapping[str, str], hf_token: str) -> dict[str, str]:
    """Build the ai-toolkit subprocess environment from ``base_env`` (pure).

    Layers the alloc/telemetry tuning on top of the inherited env and — crucially —
    injects ``HF_TOKEN`` when one is configured. pydantic loads the token from
    ``.env`` into settings but never exports it to ``os.environ``, so without this
    the gated FLUX.1-dev download would 401 even though the token was supplied the
    documented way. Honors the repo rule (token comes via ``get_settings()``, not a
    direct ``os.environ`` read).
    """
    env = {
        **base_env,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "DISABLE_TELEMETRY": "YES",
    }
    if hf_token.strip():
        env["HF_TOKEN"] = hf_token
    return env


def _to_float(match: re.Match[str] | None) -> float | None:
    """Float of a regex group, or ``None`` when absent or not parseable."""
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_progress(line: str) -> TrainProgress | None:
    """Parse one ai-toolkit stdout/stderr line into :class:`TrainProgress`, or ``None``.

    Recognises the tqdm training bar (``<name>: 10%|..| 200/2000 [.. lr: 1.0e-04
    loss: 9.870e-02]``). Requires an ``lr:`` or ``loss:`` token so the
    latent-caching bar — which also prints ``N/total`` but no lr/loss — never
    false-matches. ``stderr`` is merged into the stream, so a foreign log line
    carrying a malformed ``loss:``/``lr:`` token must yield ``None``, never raise.
    """
    loss = _to_float(_LOSS_RE.search(line))
    lr = _to_float(_LR_RE.search(line))
    if loss is None and lr is None:
        # No numeric lr/loss — either no token, or a malformed one on a foreign
        # (merged-stderr) line. Not a training-progress line.
        return None
    step_match = _STEP_RE.search(line)
    step = int(step_match.group(1)) if step_match else None
    total = int(step_match.group(2)) if step_match else None
    return TrainProgress(step=step, total_steps=total, loss=loss, lr=lr)


def resolve_output_name(settings: Settings) -> str:
    """char-LoRA filename stem / ai-toolkit run name: ``train_output_name`` or the trigger."""
    return settings.train_output_name.strip() or settings.trigger_token


def resolve_python(settings: Settings) -> str:
    """Interpreter to launch ai-toolkit with — its own cu128/sm_120 venv by default."""
    if settings.train_python:
        return settings.train_python
    return (settings.aitoolkit_dir / "venv" / "bin" / "python").as_posix()


def expected_output_path(training_folder: Path, output_name: str) -> Path:
    """Where ai-toolkit writes the final LoRA: ``<training_folder>/<name>/<name>.safetensors``.

    Mirrors ai-toolkit's ``BaseTrainProcess.save_root = training_folder / name`` and
    its final save filename ``f"{name}.safetensors"`` (intermittent saves get a
    ``_<step>`` suffix; the final, step-less save is the one we verify).
    """
    return training_folder / output_name / f"{output_name}.safetensors"


def build_aitoolkit_config(
    settings: Settings,
    *,
    dataset_dir: Path,
    training_folder: Path,
    output_name: str,
) -> dict[str, Any]:
    """Build the ai-toolkit job config (pure; no I/O).

    Mirrors the proven FLUX.1-dev LoRA recipe (``config/ohwxwoman_flux.yaml``): a
    single ``sd_trainer`` process, LoRA network, the kohya dataset folder as the
    only subset, content+identity captions kept as-is (``shuffle_tokens: false`` so
    the leading ``<char>_char`` trigger token stays pinned), and — the bit kohya
    cannot do — ``model.quantize`` with ``qtype``/``qtype_te`` + ``low_vram`` so
    Flux training fits ~16 GB. Style is held by the external cmcstyle LoRA at
    inference, so it is never written into captions here.
    """
    tool = settings.train_tool.strip().lower()
    if tool not in SUPPORTED_TOOLS:
        raise TrainError(f"train_tool={tool!r} is not supported; set APP_TRAIN_TOOL=ai-toolkit.")
    if not settings.train_base_model.strip():
        raise TrainError(
            "train_base_model is empty; set APP_TRAIN_BASE_MODEL to the Flux base "
            "(e.g. 'black-forest-labs/FLUX.1-dev') — it must match the style LoRA's base."
        )
    process: dict[str, Any] = {
        "type": "sd_trainer",
        "training_folder": training_folder.resolve().as_posix(),
        "device": "cuda:0",
        # ai-toolkit appends the trigger to a caption only if absent; ours already
        # lead with it (Character-Locker), so this is a belt-and-braces safeguard.
        "trigger_word": settings.trigger_token,
        "network": {
            "type": "lora",
            "linear": settings.train_network_dim,
            "linear_alpha": settings.train_network_alpha,
        },
        "save": {
            "dtype": "float16",
            "save_every": settings.train_save_every,
            "max_step_saves_to_keep": 4,
            "push_to_hub": False,
        },
        "datasets": [
            {
                "folder_path": dataset_dir.resolve().as_posix(),
                "caption_ext": "txt",
                "caption_dropout_rate": 0.05,
                "shuffle_tokens": False,
                "cache_latents_to_disk": True,
                "resolution": [settings.train_resolution],
            }
        ],
        "train": {
            "batch_size": settings.train_batch_size,
            "steps": settings.train_steps,
            "gradient_accumulation_steps": 1,
            "train_unet": True,
            "train_text_encoder": False,  # not supported for Flux
            "gradient_checkpointing": settings.train_gradient_checkpointing,
            "noise_scheduler": "flowmatch",
            "optimizer": settings.train_optimizer,
            "lr": settings.train_learning_rate,
            "disable_sampling": settings.train_disable_sampling,
            "ema_config": {"use_ema": settings.train_use_ema, "ema_decay": 0.99},
            "dtype": "bf16",
        },
        "model": {
            "name_or_path": settings.train_base_model,
            "is_flux": True,
            "quantize": settings.train_quantize,
            "qtype": settings.train_qtype,
            "qtype_te": settings.train_qtype_te,
            "low_vram": settings.train_low_vram,
        },
        "sample": {
            "sampler": "flowmatch",
            "sample_every": settings.train_steps + 1,  # off unless disable_sampling is cleared
            "width": settings.train_resolution,
            "height": settings.train_resolution,
            "prompts": [f"{settings.trigger_token}, full body, plain background"],
            "neg": "",
            "seed": 42,
            "walk_seed": True,
            "guidance_scale": 4,
            "sample_steps": 20,
        },
    }
    return {
        "job": "extension",
        "config": {"name": output_name, "process": [process]},
        "meta": {"name": output_name, "version": "1.0"},
    }


def render_config(config: dict[str, Any]) -> str:
    """Serialize the ai-toolkit config to JSON (ai-toolkit reads .json/.yaml alike)."""
    return json.dumps(config, indent=2) + "\n"


def build_launch_command(
    settings: Settings, *, config_path: Path, log_path: Path | None = None
) -> list[str]:
    """Build the argv: ``<ai-toolkit venv python> run.py <config> [-l <log>]``.

    ``run.py`` inserts its own cwd onto ``sys.path`` and resolves the config by
    absolute path, so the command is launched with ``cwd`` = the ai-toolkit dir.
    """
    command = [
        resolve_python(settings),
        RUN_SCRIPT,
        config_path.resolve().as_posix(),
    ]
    if log_path is not None:
        command += ["-l", log_path.resolve().as_posix()]
    return command


# --- Orchestration ---------------------------------------------------------


def dataset_images(dataset_dir: Path) -> list[Path]:
    """List trainable images in the kohya dataset folder (sorted, by extension)."""
    if not dataset_dir.is_dir():
        return []
    return sorted(
        path
        for path in dataset_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTS
    )


def build_plan(settings: Settings, *, dataset_dir: Path, output_dir: Path) -> TrainPlan:
    """Assemble the full :class:`TrainPlan` from settings and resolved paths (pure)."""
    output_name = resolve_output_name(settings)
    config = build_aitoolkit_config(
        settings, dataset_dir=dataset_dir, training_folder=output_dir, output_name=output_name
    )
    config_path = output_dir / CONFIG_FILENAME
    log_path = output_dir / TRAIN_LOG_FILENAME
    command = build_launch_command(settings, config_path=config_path, log_path=log_path)
    return TrainPlan(
        command=command,
        cwd=settings.aitoolkit_dir,
        config_path=config_path,
        config_json=render_config(config),
        output_path=expected_output_path(output_dir, output_name),
    )


def run_train(*, force: bool = False, trainer: Trainer | None = None) -> Path:
    """Settings-driven entry: train the char-LoRA from ``03_dataset`` into ``06_lora``.

    Reads the kohya dataset folder, builds the ai-toolkit plan, and runs it through
    ``trainer`` (the ai-toolkit subprocess backend by default). Idempotent: an
    already-trained LoRA is returned as-is unless ``force``. With ``force`` the
    prior run directory is removed first — ai-toolkit has no resume-disable flag and
    unconditionally resumes from any checkpoint in its ``save_root``, so a stale
    ``06_lora/<name>/`` would otherwise make "force" reload the old weights instead
    of training fresh.
    """
    import shutil

    settings = get_settings()
    workspace = Workspace(settings.workspace)

    dataset_dir = workspace.training_dir(settings.dataset_repeats, settings.trigger_token)
    if not dataset_images(dataset_dir):
        raise TrainError(
            f"no dataset images in {dataset_dir}; run the caption stage (or the full "
            f"pipeline) first to produce the kohya dataset."
        )

    out = workspace.lora
    out.mkdir(parents=True, exist_ok=True)
    plan = build_plan(settings, dataset_dir=dataset_dir, output_dir=out)

    if plan.output_path.is_file() and plan.output_path.stat().st_size > 0 and not force:
        return plan.output_path

    # The generated config/log live in ``out`` (06_lora/); ai-toolkit owns the
    # ``<name>/`` subfolder (= save_root). Clearing only that subfolder on force
    # gives ai-toolkit a clean save_root without deleting the generated config.
    run_dir = plan.output_path.parent
    if force and run_dir.is_dir():
        shutil.rmtree(run_dir)

    runner = trainer if trainer is not None else make_trainer(settings)
    return runner.train(plan)
