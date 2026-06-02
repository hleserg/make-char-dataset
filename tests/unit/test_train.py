"""Tests for make_char_dataset.train (ai-toolkit char-LoRA training).

Pure surface only — the config/launch/parse builders and the orchestration with a
``FakeTrainer``. The real :class:`~make_char_dataset.train.AiToolkitTrainer` spawns
ai-toolkit's GPU venv and is excluded from coverage.

The centerpiece is ``test_config_matches_proven_flux_recipe``: a green stub proves
the *plumbing*, not that the emitted config is a valid ai-toolkit job. So we pin
the generated config's structure to the recipe that actually trained a Flux LoRA
on this box (ai-toolkit ``config/ohwxwoman_flux.yaml``) — same key nesting, and the
low-VRAM quantization knobs (``qtype: qfloat8`` + ``low_vram``) that are the whole
reason we use ai-toolkit over kohya.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from make_char_dataset.config import Settings, get_settings
from make_char_dataset.train import (
    AiToolkitTrainer,
    SshAiToolkitTrainer,
    TrainError,
    TrainPlan,
    TrainProgress,
    build_aitoolkit_config,
    build_launch_command,
    build_plan,
    build_rsync_command,
    build_ssh_command,
    build_subprocess_env,
    dataset_images,
    expected_output_path,
    make_trainer,
    parse_progress,
    remote_train_shell,
    render_config,
    resolve_output_name,
    resolve_python,
    resolve_ssh_python,
    rewrite_config_for_remote,
    run_train,
)
from make_char_dataset.workspace import Workspace


def _settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    return get_settings()


class FakeTrainer:
    """Records the plan it was handed and writes a stand-in LoRA (no subprocess)."""

    def __init__(self) -> None:
        self.calls: list[TrainPlan] = []

    def train(self, plan: TrainPlan) -> Path:
        self.calls.append(plan)
        plan.output_path.parent.mkdir(parents=True, exist_ok=True)
        plan.output_path.write_bytes(b"LORA")
        return plan.output_path


# --- parse_progress --------------------------------------------------------


def test_parse_progress_full_bar() -> None:
    line = "kael_char:  10%|██   | 200/2000 [01:30<13:30,  1.2it/s, lr: 1.0e-04 loss: 9.870e-02]"
    progress = parse_progress(line)
    assert progress is not None
    assert progress.step == 200
    assert progress.total_steps == 2000
    assert progress.loss == pytest.approx(0.0987)
    assert progress.lr == pytest.approx(1e-4)
    assert progress.fraction == pytest.approx(0.1)


def test_parse_progress_loss_without_step() -> None:
    progress = parse_progress("loss: 1.2e-01")
    assert progress is not None
    assert progress.step is None
    assert progress.fraction is None


def test_parse_progress_ignores_caching_bar() -> None:
    # The latent-caching bar prints N/total but has no lr/loss — must not match.
    assert parse_progress("Caching latents: 100%|████| 8/8 [00:02<00:00]") is None


def test_parse_progress_ignores_plain_text() -> None:
    assert parse_progress("Loading Flux model from black-forest-labs/FLUX.1-dev") is None


@pytest.mark.parametrize("line", ["loss: -inf", "lr: --", "loss: .", "loss: 1e", "lr: -"])
def test_parse_progress_malformed_token_returns_none(line: str) -> None:
    # stderr is merged into the stream; a foreign 'loss:'/'lr:' line must not crash
    # the read loop. Such tokens parse to None, not a ValueError.
    assert parse_progress(line) is None


def test_progress_fraction_zero_total() -> None:
    assert TrainProgress(step=5, total_steps=0).fraction is None


# --- build_subprocess_env (HF token propagation) ---------------------------


def test_build_subprocess_env_injects_token() -> None:
    env = build_subprocess_env({"PATH": "/usr/bin"}, "hf_secret")
    assert env["HF_TOKEN"] == "hf_secret"
    assert env["PATH"] == "/usr/bin"  # base env preserved
    assert env["DISABLE_TELEMETRY"] == "YES"


def test_build_subprocess_env_no_expandable_segments() -> None:
    # expandable_segments uses CUDA VMM calls that fail on WSL2 inside ai-toolkit's
    # quanto cast ("CUDA driver error: out of memory"). The proven launch is bare.
    assert "PYTORCH_CUDA_ALLOC_CONF" not in build_subprocess_env({}, "tok")


def test_build_subprocess_env_omits_empty_token() -> None:
    assert "HF_TOKEN" not in build_subprocess_env({}, "")
    assert "HF_TOKEN" not in build_subprocess_env({}, "   ")


# --- resolvers -------------------------------------------------------------


def test_resolve_output_name_defaults_to_trigger(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, APP_TRIGGER_TOKEN="kael", APP_TRAIN_OUTPUT_NAME="")
    assert resolve_output_name(settings) == "kael"


def test_resolve_output_name_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, APP_TRAIN_OUTPUT_NAME="kael_char_v2")
    assert resolve_output_name(settings) == "kael_char_v2"


def test_resolve_python_uses_venv_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, APP_AITOOLKIT_DIR="/opt/ai-toolkit", APP_TRAIN_PYTHON="")
    assert resolve_python(settings) == "/opt/ai-toolkit/venv/bin/python"


def test_resolve_python_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, APP_TRAIN_PYTHON="/usr/bin/python3")
    assert resolve_python(settings) == "/usr/bin/python3"


def test_expected_output_path() -> None:
    assert expected_output_path(Path("/ws/06_lora"), "kael") == Path(
        "/ws/06_lora/kael/kael.safetensors"
    )


# --- config (the golden recipe) --------------------------------------------


def test_config_matches_proven_flux_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(
        monkeypatch,
        APP_TRIGGER_TOKEN="kael_char",
        APP_TRAIN_RESOLUTION="768",
        APP_TRAIN_NETWORK_DIM="16",
        APP_TRAIN_NETWORK_ALPHA="16",
    )
    config = build_aitoolkit_config(
        settings,
        dataset_dir=Path("/ws/03_dataset/10_kael_char"),
        training_folder=Path("/ws/06_lora"),
        output_name="kael_char",
    )

    # Top-level job shape (job/config/meta), matching ohwxwoman_flux.yaml.
    assert config["job"] == "extension"
    assert config["config"]["name"] == "kael_char"
    assert config["meta"]["name"] == "kael_char"
    process = config["config"]["process"]
    assert isinstance(process, list) and len(process) == 1
    proc = process[0]
    assert proc["type"] == "sd_trainer"
    assert proc["training_folder"] == "/ws/06_lora"
    assert proc["trigger_word"] == "kael_char"

    # network: a LoRA at the configured rank/alpha.
    assert proc["network"] == {"type": "lora", "linear": 16, "linear_alpha": 16}

    # dataset: our kohya folder, .txt captions, trigger pinned (no shuffle), bucketed.
    dataset = proc["datasets"][0]
    assert dataset["folder_path"] == "/ws/03_dataset/10_kael_char"
    assert dataset["caption_ext"] == "txt"
    assert dataset["shuffle_tokens"] is False
    assert dataset["resolution"] == [768]

    # train: Flux flow-matching, text-encoder NOT trained, sampling off, low-VRAM optimizer.
    tr = proc["train"]
    assert tr["train_unet"] is True
    assert tr["train_text_encoder"] is False
    assert tr["noise_scheduler"] == "flowmatch"
    assert tr["dtype"] == "bf16"
    assert tr["disable_sampling"] is True
    assert tr["optimizer"] == "adafactor"  # near-zero state -> fits ~16 GB

    # model: Flux base + qfloat8 low-VRAM quantization (NOT qint4 — see the guard test).
    model = proc["model"]
    assert model["name_or_path"] == "black-forest-labs/FLUX.1-dev"
    assert model["is_flux"] is True
    assert model["quantize"] is True
    assert model["qtype"] == "qfloat8"
    assert model["qtype_te"] == "qfloat8"
    assert model["low_vram"] is True


def test_config_honours_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(
        monkeypatch,
        APP_TRAIN_BASE_MODEL="/models/flux1-dev.safetensors",
        APP_TRAIN_QTYPE="qint8",
        APP_TRAIN_LOW_VRAM="false",
        APP_TRAIN_DISABLE_SAMPLING="false",
        APP_TRAIN_OPTIMIZER="adamw8bit",
    )
    proc = build_aitoolkit_config(
        settings, dataset_dir=Path("/d"), training_folder=Path("/o"), output_name="c"
    )["config"]["process"][0]
    assert proc["model"]["name_or_path"] == "/models/flux1-dev.safetensors"
    assert proc["model"]["qtype"] == "qint8"
    assert proc["model"]["low_vram"] is False
    assert proc["train"]["disable_sampling"] is False
    assert proc["train"]["optimizer"] == "adamw8bit"


def test_config_rejects_qint4_with_low_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    # qint4's int4pack kernel is CUDA-only; low_vram quantizes on CPU -> ai-toolkit
    # throws and writes no weights. The builder must refuse the combo up front.
    settings = _settings(monkeypatch, APP_TRAIN_QTYPE="qint4")  # low_vram defaults true
    with pytest.raises(TrainError, match=r"qint4 .*incompatible with low_vram"):
        build_aitoolkit_config(
            settings, dataset_dir=Path("/d"), training_folder=Path("/o"), output_name="c"
        )
    # qint4 is fine when low_vram is off (it then quantizes on the GPU).
    ok = _settings(monkeypatch, APP_TRAIN_QTYPE="qint4", APP_TRAIN_LOW_VRAM="false")
    proc = build_aitoolkit_config(
        ok, dataset_dir=Path("/d"), training_folder=Path("/o"), output_name="c"
    )["config"]["process"][0]
    assert proc["model"]["qtype"] == "qint4"


def test_config_ema_defaults_off_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    # Default mirrors the proven low-VRAM recipe (use_ema=false); overridable on.
    default_proc = build_aitoolkit_config(
        _settings(monkeypatch), dataset_dir=Path("/d"), training_folder=Path("/o"), output_name="c"
    )["config"]["process"][0]
    assert default_proc["train"]["ema_config"]["use_ema"] is False

    on_proc = build_aitoolkit_config(
        _settings(monkeypatch, APP_TRAIN_USE_EMA="true"),
        dataset_dir=Path("/d"),
        training_folder=Path("/o"),
        output_name="c",
    )["config"]["process"][0]
    assert on_proc["train"]["ema_config"]["use_ema"] is True


def test_config_rejects_unknown_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, APP_TRAIN_TOOL="kohya")
    with pytest.raises(TrainError, match="not supported"):
        build_aitoolkit_config(
            settings, dataset_dir=Path("/d"), training_folder=Path("/o"), output_name="c"
        )


def test_config_rejects_empty_base_model(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, APP_TRAIN_BASE_MODEL="")
    with pytest.raises(TrainError, match="train_base_model is empty"):
        build_aitoolkit_config(
            settings, dataset_dir=Path("/d"), training_folder=Path("/o"), output_name="c"
        )


def test_render_config_round_trips() -> None:
    text = render_config({"job": "extension", "config": {"name": "x"}})
    assert text.endswith("\n")
    assert json.loads(text) == {"job": "extension", "config": {"name": "x"}}


# --- launch command --------------------------------------------------------


def test_build_launch_command_with_log(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, APP_TRAIN_PYTHON="/venv/python")
    command = build_launch_command(
        settings, config_path=Path("/o/cfg.json"), log_path=Path("/o/train.log")
    )
    assert command == ["/venv/python", "run.py", "/o/cfg.json", "-l", "/o/train.log"]


def test_build_launch_command_without_log(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, APP_TRAIN_PYTHON="/venv/python")
    command = build_launch_command(settings, config_path=Path("/o/cfg.json"))
    assert command == ["/venv/python", "run.py", "/o/cfg.json"]


# --- dataset_images --------------------------------------------------------


def test_dataset_images_lists_and_filters(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.jpg").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("caption")
    (tmp_path / "sub").mkdir()
    images = dataset_images(tmp_path)
    assert [p.name for p in images] == ["a.png", "b.jpg"]  # sorted, non-images + subdirs dropped


def test_dataset_images_missing_dir(tmp_path: Path) -> None:
    assert dataset_images(tmp_path / "nope") == []


# --- make_trainer ----------------------------------------------------------


def test_make_trainer_returns_aitoolkit(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    assert isinstance(make_trainer(settings), AiToolkitTrainer)


def test_make_trainer_threads_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    # The token configured via .env/HF_TOKEN must reach the trainer so it can be
    # injected into the ai-toolkit subprocess env (it is never on os.environ).
    settings = _settings(monkeypatch, HF_TOKEN="hf_from_env")
    trainer = make_trainer(settings)
    assert isinstance(trainer, AiToolkitTrainer)
    assert trainer._hf_token == "hf_from_env"


def test_make_trainer_rejects_unknown_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, APP_TRAIN_TOOL="onetrainer")
    with pytest.raises(TrainError, match="not supported"):
        make_trainer(settings)


# --- remote (ssh / cloud) training -----------------------------------------


def test_make_trainer_ssh(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, APP_TRAIN_BACKEND="ssh", APP_TRAIN_SSH_HOST="gpu@cloud")
    assert isinstance(make_trainer(settings), SshAiToolkitTrainer)


def test_make_trainer_ssh_requires_host(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, APP_TRAIN_BACKEND="ssh")  # no host
    with pytest.raises(TrainError, match="APP_TRAIN_SSH_HOST"):
        make_trainer(settings)


def test_make_trainer_rejects_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, APP_TRAIN_BACKEND="kubernetes")
    with pytest.raises(TrainError, match="train_backend="):
        make_trainer(settings)


def test_resolve_ssh_python(monkeypatch: pytest.MonkeyPatch) -> None:
    default = _settings(monkeypatch, APP_TRAIN_SSH_AITOOLKIT_DIR="/opt/ai-toolkit")
    assert resolve_ssh_python(default) == "/opt/ai-toolkit/venv/bin/python"
    override = _settings(monkeypatch, APP_TRAIN_SSH_PYTHON="/usr/bin/python3")
    assert resolve_ssh_python(override) == "/usr/bin/python3"


def test_rewrite_config_for_remote() -> None:
    local = build_aitoolkit_config(
        get_settings(),  # default trigger; paths get rewritten below
        dataset_dir=Path("/ws/03_dataset/10_kael"),
        training_folder=Path("/ws/06_lora"),
        output_name="kael",
    )
    remote_json = rewrite_config_for_remote(
        render_config(local), remote_workdir="~/work", dataset_name="10_kael"
    )
    proc = json.loads(remote_json)["config"]["process"][0]
    assert proc["training_folder"] == "~/work/06_lora"
    assert proc["datasets"][0]["folder_path"] == "~/work/03_dataset/10_kael"
    assert proc["model"]["qtype"] == "qfloat8"  # the recipe is untouched


def test_build_ssh_command() -> None:
    assert build_ssh_command("gpu@cloud", "echo hi") == ["ssh", "gpu@cloud", "echo hi"]
    assert build_ssh_command("gpu@cloud", "echo hi", port=2222) == [
        "ssh",
        "-p",
        "2222",
        "gpu@cloud",
        "echo hi",
    ]


def test_build_rsync_command() -> None:
    assert build_rsync_command("a/", "h:b/", delete=True) == [
        "rsync",
        "-az",
        "--delete",
        "a/",
        "h:b/",
    ]
    assert build_rsync_command("h:b/", "a/", port=2222) == [
        "rsync",
        "-az",
        "-e",
        "ssh -p 2222",
        "h:b/",
        "a/",
    ]


def test_remote_train_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, APP_TRAIN_SSH_AITOOLKIT_DIR="/opt/ai-toolkit")
    shell = remote_train_shell(
        settings,
        remote_config="~/w/cfg.json",
        remote_log="~/w/train.log",
        remote_run_dir="~/w/06_lora/kael",
    )
    assert "rm -rf ~/w/06_lora/kael" in shell  # fresh save_root → no stale resume
    assert "cd /opt/ai-toolkit" in shell
    assert "/opt/ai-toolkit/venv/bin/python run.py ~/w/cfg.json -l ~/w/train.log" in shell


# --- build_plan + run_train ------------------------------------------------


def _ws_with_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, trigger: str = "kael", repeats: int = 10
) -> Workspace:
    ws_root = tmp_path / "ws"
    monkeypatch.setenv("APP_WORKSPACE", str(ws_root))
    monkeypatch.setenv("APP_TRIGGER_TOKEN", trigger)
    monkeypatch.setenv("APP_DATASET_REPEATS", str(repeats))
    get_settings.cache_clear()
    ws = Workspace(ws_root)
    dataset = ws.training_dir(repeats, trigger)
    dataset.mkdir(parents=True)
    (dataset / "img_000.png").write_bytes(b"PNGDATA")
    (dataset / "img_000.txt").write_text("kael, full body", encoding="utf-8")
    return ws


def test_build_plan_wires_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, APP_TRIGGER_TOKEN="kael", APP_TRAIN_PYTHON="/venv/py")
    plan = build_plan(
        settings, dataset_dir=Path("/ws/03_dataset/10_kael"), output_dir=Path("/ws/06_lora")
    )
    assert plan.output_path == Path("/ws/06_lora/kael/kael.safetensors")
    assert plan.config_path == Path("/ws/06_lora/aitoolkit_config.json")
    assert plan.cwd == settings.aitoolkit_dir
    assert plan.command[:3] == ["/venv/py", "run.py", "/ws/06_lora/aitoolkit_config.json"]
    assert plan.dataset_dir == Path("/ws/03_dataset/10_kael")  # source the ssh backend uploads
    assert plan.output_name == "kael"
    # config_json round-trips and points the dataset at our kohya folder.
    config = json.loads(plan.config_json)
    assert config["config"]["process"][0]["datasets"][0]["folder_path"] == "/ws/03_dataset/10_kael"


def test_run_train_requires_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APP_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("APP_TRIGGER_TOKEN", "kael")
    get_settings.cache_clear()
    with pytest.raises(TrainError, match="no dataset images"):
        run_train(trainer=FakeTrainer())


def test_run_train_invokes_trainer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ws = _ws_with_dataset(monkeypatch, tmp_path)
    fake = FakeTrainer()
    result = run_train(trainer=fake)
    assert result == ws.lora / "kael" / "kael.safetensors"
    assert result.read_bytes() == b"LORA"
    assert len(fake.calls) == 1
    assert fake.calls[0].output_path == result


def test_run_train_idempotent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ws = _ws_with_dataset(monkeypatch, tmp_path)
    output = ws.lora / "kael" / "kael.safetensors"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"EXISTING")
    fake = FakeTrainer()
    result = run_train(trainer=fake)
    assert result == output
    assert result.read_bytes() == b"EXISTING"  # untouched
    assert fake.calls == []  # trainer never invoked


def test_run_train_force_retrains(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ws = _ws_with_dataset(monkeypatch, tmp_path)
    output = ws.lora / "kael" / "kael.safetensors"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"EXISTING")
    fake = FakeTrainer()
    run_train(force=True, trainer=fake)
    assert len(fake.calls) == 1
    assert output.read_bytes() == b"LORA"  # rewritten by the (fake) trainer


def test_run_train_force_clears_prior_run_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # force must wipe 06_lora/<name>/ before launching: ai-toolkit has no
    # resume-disable flag and would otherwise resume from the stale checkpoint.
    ws = _ws_with_dataset(monkeypatch, tmp_path)
    run_dir = ws.lora / "kael"
    run_dir.mkdir(parents=True)
    (run_dir / "kael.safetensors").write_bytes(b"OLD")
    (run_dir / "kael_000000250.safetensors").write_bytes(b"OLD-STEP")

    class DirProbeTrainer:
        def __init__(self) -> None:
            self.dir_existed_at_call: bool | None = None

        def train(self, plan: TrainPlan) -> Path:
            self.dir_existed_at_call = plan.output_path.parent.exists()
            plan.output_path.parent.mkdir(parents=True, exist_ok=True)
            plan.output_path.write_bytes(b"FRESH")
            return plan.output_path

    probe = DirProbeTrainer()
    run_train(force=True, trainer=probe)
    assert probe.dir_existed_at_call is False  # run dir cleared before launch
    assert not (run_dir / "kael_000000250.safetensors").exists()  # stale step save gone
    assert (run_dir / "kael.safetensors").read_bytes() == b"FRESH"
