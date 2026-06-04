"""ComfyUI generation backends — img2img anchor multiplication + external style LoRA.

Two backends share one HTTP transport (:class:`_ComfyHttpBackend`: submit to
``/prompt`` → poll ``/history`` → fetch ``/view`` over the **stdlib** ``urllib``,
no third-party client) and differ only in the pure prompt **graph** they build:

* :class:`ComfyBackend` — SDXL img2img + optional ControlNet (OpenPose/depth),
  the original local recipe.
* :class:`FluxImg2ImgBackend` — the **restylization** backend (HLE-804): Flux.1-dev
  img2img through the external **cmcstyle** style LoRA, redrawing each identity
  anchor into the comic style while a low denoise holds the character. This is the
  same proven Flux graph the stack-eval grid uses (``UNETLoader`` +
  ``DualCLIPLoader`` + ``VAELoader`` + ``LoraLoaderModelOnly`` + ``FluxGuidance``,
  ``cfg=1.0``), but img2img from the anchor instead of an empty latent.

Heavy and exercised only in the local heavy tier (a running ComfyUI server + the
downloaded base/style models). The transport is therefore lazy-imported behind
:class:`make_char_dataset.generate.GenerationBackend` and omitted from coverage;
the unit-tested parts are the pure ``_build_workflow`` graph builders and the
small ``_extract_error`` / ``_combine_prompt`` helpers.

License-safe by construction: img2img + ControlNet (OpenPose/depth) only — never
PuLID / IP-Adapter / InstantID (InsightFace is non-commercial, HLE-668).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from make_char_dataset.generate import VariantSpec

_NEGATIVE = (
    "lowres, bad anatomy, extra fingers, extra limbs, deformed, watermark, "
    "text, signature, jpeg artifacts, blurry"
)

# Structural-hint preprocessor per ControlNet mode (comfyui-controlnet-aux nodes).
_CONTROLNET_PREPROCESSOR: dict[str, str] = {
    "openpose": "OpenposePreprocessor",
    "depth": "Zoe-DepthMapPreprocessor",
}


def _combine_prompt(planned: str, style_prompt: str) -> str:
    """Join the style-free planned prompt with the external style prompt/trigger."""
    return ", ".join(part for part in (planned, style_prompt) if part)


def _extract_error(entry: dict[str, Any]) -> str | None:
    """Return a human-readable error from a finished ComfyUI ``/history`` entry, or None.

    ComfyUI records a failed prompt as a completed history entry whose ``status`` has
    ``status_str == "error"`` and an ``execution_error`` message — but with no output
    images. Surfacing that turns a silent "no image" into an actionable failure (and
    stops the poll loop from waiting out the whole timeout on an already-dead prompt).
    """
    status = entry.get("status")
    if not isinstance(status, dict):
        return None
    for message in status.get("messages", []):
        if (
            isinstance(message, list | tuple)
            and len(message) >= 2
            and message[0] == "execution_error"
        ):
            detail = message[1] if isinstance(message[1], dict) else {}
            node = detail.get("node_type") or detail.get("node_id")
            exc = (
                detail.get("exception_message") or detail.get("exception_type") or "execution error"
            )
            return f"{exc} (node {node})" if node else str(exc)
    if status.get("status_str") == "error":
        return "execution error"
    return None


@dataclass
class _ComfyHttpBackend:
    """Shared ComfyUI HTTP transport: submit a graph, wait, fetch the output image.

    Subclasses supply the pure prompt graph via :meth:`_build_workflow`; this base
    is graph-agnostic (no SDXL/Flux specifics) and never imported into the pure
    ``generate`` module — only instantiated lazily inside its backend selector.
    """

    base_url: str = "http://127.0.0.1:8188"
    poll_interval_s: float = 1.0
    timeout_s: float = 600.0

    def _build_workflow(self, spec: VariantSpec, anchor_name: str) -> dict[str, Any]:
        """Return the ComfyUI API-format prompt graph for ``spec`` (pure; overridden)."""
        raise NotImplementedError

    def render(  # pragma: no cover
        self, spec: VariantSpec, anchor_path: Path, out_path: Path
    ) -> None:
        """Submit the workflow to ComfyUI, wait for completion, and save the image."""
        import urllib.request

        uploaded = self._upload_image(anchor_path)
        workflow = self._build_workflow(spec, uploaded)
        body = json.dumps({"prompt": workflow}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/prompt",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:  # nosec B310
            prompt_id = json.loads(response.read().decode("utf-8"))["prompt_id"]
        image = self._await_image(prompt_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(image)

    def _upload_image(self, anchor_path: Path) -> str:  # pragma: no cover
        """Upload the anchor to ComfyUI's input store; return its server-side name."""
        import urllib.request

        boundary = "----make-char-dataset"
        payload = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{anchor_path.name}"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode()
        payload += anchor_path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        request = urllib.request.Request(
            f"{self.base_url}/upload/image",
            data=payload,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:  # nosec B310
            return str(json.loads(response.read().decode("utf-8"))["name"])

    def _await_image(self, prompt_id: str) -> bytes:  # pragma: no cover
        """Poll /history until the prompt finishes, then fetch the first output image.

        A prompt that lands in /history is done: return its first image, or — if it
        finished with no image (e.g. an OOM/validation error) — raise immediately with
        the recorded error instead of blocking until ``timeout_s`` (the bug the live
        denoise calibration hit: an errored prompt looked identical to a slow one).
        """
        import urllib.parse
        import urllib.request

        start = time.monotonic()
        while time.monotonic() - start < self.timeout_s:
            with urllib.request.urlopen(  # nosec B310
                f"{self.base_url}/history/{prompt_id}", timeout=self.timeout_s
            ) as response:
                history = json.loads(response.read().decode("utf-8"))
            entry = history.get(prompt_id)
            if entry:
                for node in entry.get("outputs", {}).values():
                    for image in node.get("images", []):
                        query = urllib.parse.urlencode(image)
                        with urllib.request.urlopen(  # nosec B310
                            f"{self.base_url}/view?{query}", timeout=self.timeout_s
                        ) as image_response:
                            return bytes(image_response.read())
                error = _extract_error(entry)
                raise RuntimeError(
                    f"ComfyUI prompt {prompt_id} finished with no image: "
                    f"{error or 'no output produced'}"
                )
            time.sleep(self.poll_interval_s)
        raise TimeoutError(f"ComfyUI prompt {prompt_id} did not finish in {self.timeout_s}s")


@dataclass
class ComfyBackend(_ComfyHttpBackend):
    """Drive a local ComfyUI server to render one variant via SDXL img2img + ControlNet."""

    base_model: str = "sdxl"
    style_lora_path: str = ""
    style_lora_weight: float = 0.85
    style_prompt: str = ""
    controlnet_model: str = ""

    def _checkpoint(self) -> str:
        """Map the configured base-model family to a ComfyUI checkpoint filename."""
        if self.base_model == "sdxl":
            return "sd_xl_base_1.0.safetensors"
        return f"{self.base_model}.safetensors"

    def _positive(self, spec: VariantSpec) -> str:
        """Combine the style-free planned prompt with the external style prompt."""
        return _combine_prompt(spec.prompt, self.style_prompt)

    def _build_workflow(self, spec: VariantSpec, anchor_name: str) -> dict[str, Any]:
        """Build a ComfyUI prompt graph for this variant (pure; no I/O).

        img2img from the anchor + optional style LoRA + optional ControlNet, fed to
        a KSampler at ``spec.denoise``. ControlNet/LoRA nodes are wired only when a
        model is configured, so the graph degrades cleanly when inputs are absent.
        """
        width, height = (int(value) for value in spec.bucket.split("x", 1))
        graph: dict[str, Any] = {
            "checkpoint": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": self._checkpoint()},
            },
            "anchor": {"class_type": "LoadImage", "inputs": {"image": anchor_name}},
            # Resize the anchor to the per-class aspect bucket BEFORE encoding, so
            # the img2img latent (and thus the output) actually takes the bucket —
            # KSampler has no width/height in an img2img graph.
            "scale": {
                "class_type": "ImageScale",
                "inputs": {
                    "image": ["anchor", 0],
                    "width": width,
                    "height": height,
                    "upscale_method": "lanczos",
                    "crop": "center",
                },
            },
            "vae_encode": {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["scale", 0], "vae": ["checkpoint", 2]},
            },
            "positive": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": self._positive(spec), "clip": ["checkpoint", 1]},
            },
            "negative": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": _NEGATIVE, "clip": ["checkpoint", 1]},
            },
            "sampler": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["checkpoint", 0],
                    "positive": ["positive", 0],
                    "negative": ["negative", 0],
                    "latent_image": ["vae_encode", 0],
                    "denoise": spec.denoise,
                    "seed": spec.seed,
                    "steps": 30,
                    "cfg": 6.5,
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "karras",
                },
            },
            "vae_decode": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["sampler", 0], "vae": ["checkpoint", 2]},
            },
            "save": {
                "class_type": "SaveImage",
                "inputs": {"images": ["vae_decode", 0], "filename_prefix": f"var_{spec.index:03d}"},
            },
        }
        if self.style_lora_path:
            graph["style_lora"] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": Path(self.style_lora_path).name,
                    "strength_model": self.style_lora_weight,
                    "strength_clip": self.style_lora_weight,
                    "model": ["checkpoint", 0],
                    "clip": ["checkpoint", 1],
                },
            }
            graph["sampler"]["inputs"]["model"] = ["style_lora", 0]
            graph["positive"]["inputs"]["clip"] = ["style_lora", 1]
            graph["negative"]["inputs"]["clip"] = ["style_lora", 1]
        if self.controlnet_model and spec.controlnet != "none":
            # Derive the structural hint from the scaled anchor via the matching
            # preprocessor (OpenPose / depth), so the configured mode is meaningful.
            preprocessor = _CONTROLNET_PREPROCESSOR.get(spec.controlnet)
            if preprocessor:
                graph["controlnet_pre"] = {
                    "class_type": preprocessor,
                    "inputs": {"image": ["scale", 0]},
                }
                control_image = ["controlnet_pre", 0]
            else:
                control_image = ["scale", 0]
            graph["controlnet"] = {
                "class_type": "ControlNetLoader",
                "inputs": {"control_net_name": self.controlnet_model},
            }
            graph["controlnet_apply"] = {
                "class_type": "ControlNetApply",
                "inputs": {
                    "conditioning": ["positive", 0],
                    "control_net": ["controlnet", 0],
                    "image": control_image,
                    "strength": 0.7,
                },
            }
            graph["sampler"]["inputs"]["positive"] = ["controlnet_apply", 0]
        return graph


@dataclass
class FluxImg2ImgBackend(_ComfyHttpBackend):
    """Restylize one anchor through Flux.1-dev + the external style LoRA (img2img).

    The HLE-804 restylization step: redraw each identity anchor in the comic style so
    the character and the cmcstyle LoRA stack cleanly. The Flux model is loaded as the
    split ``UNETLoader`` + ``DualCLIPLoader`` + ``VAELoader`` trio this box uses (not a
    single all-in-one checkpoint), the style LoRA is chained model-only, and a low
    ``spec.denoise`` keeps identity while the style is applied. ``cfg`` is 1.0 — Flux
    steers with ``FluxGuidance``, not classifier-free guidance, so the negative prompt
    is empty (a required-but-inert KSampler input).
    """

    unet: str = "flux1-dev-fp8.safetensors"
    clip_l: str = "clip_l.safetensors"
    t5xxl: str = "t5xxl_fp8_e4m3fn.safetensors"
    vae: str = "ae.safetensors"
    style_lora_path: str = ""
    style_lora_weight: float = 0.85
    style_prompt: str = ""
    guidance: float = 3.5
    steps: int = 24

    def _positive(self, spec: VariantSpec) -> str:
        """Combine the style-free planned prompt with the external style trigger."""
        return _combine_prompt(spec.prompt, self.style_prompt)

    def _build_workflow(self, spec: VariantSpec, anchor_name: str) -> dict[str, Any]:
        """Build the Flux img2img restylization graph for this variant (pure; no I/O).

        Mirrors the stack-eval Flux graph (split loaders + ``FluxGuidance`` + cfg 1.0)
        but seeds the latent from the anchor: ``LoadImage`` → ``ImageScale`` (to the
        per-class aspect bucket) → ``VAEEncode``, with KSampler at ``spec.denoise``.
        The style LoRA chains via ``LoraLoaderModelOnly`` only when configured, so the
        graph degrades to plain Flux img2img when no style LoRA is supplied.
        """
        width, height = (int(value) for value in spec.bucket.split("x", 1))
        graph: dict[str, Any] = {
            "10": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": self.unet, "weight_dtype": "default"},
            },
            "11": {
                "class_type": "DualCLIPLoader",
                "inputs": {"clip_name1": self.clip_l, "clip_name2": self.t5xxl, "type": "flux"},
            },
            "12": {"class_type": "VAELoader", "inputs": {"vae_name": self.vae}},
            "15": {"class_type": "LoadImage", "inputs": {"image": anchor_name}},
            "16": {
                "class_type": "ImageScale",
                "inputs": {
                    "image": ["15", 0],
                    "width": width,
                    "height": height,
                    "upscale_method": "lanczos",
                    "crop": "center",
                },
            },
            "17": {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["16", 0], "vae": ["12", 0]},
            },
            "20": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["11", 0], "text": self._positive(spec)},
            },
            "21": {
                "class_type": "FluxGuidance",
                "inputs": {"conditioning": ["20", 0], "guidance": self.guidance},
            },
            "22": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["11", 0], "text": ""}},
        }
        model_ref: list[Any] = ["10", 0]
        if self.style_lora_path:
            graph["13"] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": model_ref,
                    "lora_name": Path(self.style_lora_path).name,
                    "strength_model": self.style_lora_weight,
                },
            }
            model_ref = ["13", 0]
        graph["40"] = {
            "class_type": "KSampler",
            "inputs": {
                "model": model_ref,
                "positive": ["21", 0],
                "negative": ["22", 0],
                "latent_image": ["17", 0],
                "seed": spec.seed,
                "steps": self.steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": spec.denoise,
            },
        }
        graph["50"] = {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["40", 0], "vae": ["12", 0]},
        }
        graph["60"] = {
            "class_type": "SaveImage",
            "inputs": {"images": ["50", 0], "filename_prefix": f"var_{spec.index:03d}"},
        }
        return graph
