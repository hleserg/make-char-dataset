"""ComfyUI generation backend — img2img + ControlNet + external style LoRA.

Heavy, GPU/network-bound, and exercised only in the local heavy tier (a running
ComfyUI server + downloaded SDXL/ControlNet/style-LoRA models, all human-provided
inputs tracked in HLE-759). It is therefore lazy-imported behind
:class:`make_char_dataset.generate.GenerationBackend` and omitted from coverage;
its only unit-tested part is the pure :meth:`ComfyBackend._build_workflow` graph.

Transport: the ComfyUI HTTP API via the **stdlib** ``urllib`` (no third-party
client) — submit to ``/prompt``, poll ``/history/<id>`` for completion, fetch via
``/view``. NOTE: the issue says "HTTP+websocket"; websocket progress-streaming is
swapped for ``/history`` polling because the websocket client dependency cannot be
added in this offline/sandboxed environment. Progress streaming is deferred to [6];
the functional result (submit → wait → fetch) is unchanged.

License-safe by construction: SDXL img2img + ControlNet (OpenPose/depth) only —
never PuLID / IP-Adapter / InstantID (InsightFace is non-commercial, HLE-668).
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


@dataclass
class ComfyBackend:
    """Drive a local ComfyUI server to render one variant via img2img + ControlNet."""

    base_url: str = "http://127.0.0.1:8188"
    base_model: str = "sdxl"
    style_lora_path: str = ""
    style_lora_weight: float = 0.85
    style_prompt: str = ""
    controlnet_model: str = ""
    poll_interval_s: float = 1.0
    timeout_s: float = 600.0

    def _checkpoint(self) -> str:
        """Map the configured base-model family to a ComfyUI checkpoint filename."""
        if self.base_model == "sdxl":
            return "sd_xl_base_1.0.safetensors"
        return f"{self.base_model}.safetensors"

    def _positive(self, spec: VariantSpec) -> str:
        """Combine the style-free planned prompt with the external style prompt."""
        return ", ".join(part for part in (spec.prompt, self.style_prompt) if part)

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
        """Poll /history until the prompt finishes, then fetch the first output image."""
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
            time.sleep(self.poll_interval_s)
        raise TimeoutError(f"ComfyUI prompt {prompt_id} did not finish in {self.timeout_s}s")
