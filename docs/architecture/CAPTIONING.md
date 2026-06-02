# Captioning (VLM prose) — Character-Locker

The `caption` stage writes one `.txt` sidecar per image in
`03_dataset/<repeats>_<trigger>/`. What the caption **says** decides what the
character LoRA learns, so this is a load-bearing stage, not a formality.

## The contrastive rule (and the STYLE↔CHARACTER inversion)

A LoRA + caption train **contrastively**: whatever the caption *describes* binds to
those (varying) words; whatever is *left undescribed but constant* binds to the
**trigger**. You choose what the trigger absorbs by choosing what you **don't** say.

For a **character** LoRA (`<char>_char`) the trigger must absorb the **identity**
(face / hair / eyes / skin / **body build** / age). So the caption:

- **begins with the trigger** `<char>_char`;
- **describes only what varies**: pose, expression, **outfit**, props, background,
  camera shot, angle, lighting;
- **omits the invariant identity** — those bind to the trigger;
- **omits the art style** — it is owned by the external `cmcstyle` style LoRA and is
  already baked into the (re-stylised) training images (HLE-802 bridge).

This is the **inverse** of the style captioner in `make-style-dataset`, which
describes *all* content (including the characters' physical detail) so that only the
*style* is left to bind to its trigger.

**Outfit is described on purpose** — binding it to words keeps the character
**re-dressable** (prompt `<char>_char` + a new outfit at inference). The constant
**body build** is omitted, so the figure stays fixed to the trigger.

## Mechanism — VLM prose via the proxy

Booru tags (WD14) are weak for this: they are tag-soup (Flux's T5 wants prose) and
emit generic identity/style tags that quietly bind the wrong things. The default
captioner is therefore **Gemini prose**.

Gemini is geo-blocked from this box, so **every** call is relayed through the private
HF Space `hleserg/proxy_gemini_api` (`proxy.py` → `vlm_caption.py`). A read-scope HF
token is enough to call it (`HF_TOKEN`, or the `huggingface-cli` login cache). Images
are **downscaled to `vlm_max_image_side` (768 px) JPEG before upload** — full-res
1024²+ bodies stall the proxy's HTTPS upload. Calls run concurrently
(`vlm_concurrency`, ≤16) with retry/backoff on transient (5xx/429/network) errors;
the reply is normalised to a single trigger-first line.

## The captioner seam

`caption_dataset` captions each cleaned image through a pluggable `Captioner`
(`APP_CAPTIONER`):

| `APP_CAPTIONER` | Captioner | Use |
|---|---|---|
| `vlm` *(default)* | `VlmCaptioner` (Gemini prose via the proxy) | production |
| `wd14` | `Wd14Captioner` (booru tags → Character-Locker) | legacy fallback |
| `stub` | `StubCaptioner` (deterministic, offline) | tests / CI / the verifier |

Like the generation backend, the heavy/network captioners are lazy-imported and the
CI path uses the stub, so `make check` and the verifier stay GPU/network-free.

## Knobs (`APP_VLM_*`, see `.env.example`)

`APP_CAPTIONER`, `APP_VLM_MODEL` (`gemini-2.5-flash` for agent runs, `…-pro` for the
UI button), `APP_VLM_CONCURRENCY`, `APP_VLM_MAX_IMAGE_SIDE`, `APP_VLM_PROXY_URL`, and
`HF_TOKEN`.

## Ordering

In the real pipeline, caption the **re-stylised** dataset (after the HLE-802
restylization bridge), so the prose describes the in-style frames. The captioner
itself is independent of that and runs over any laid-out `03_dataset`.
