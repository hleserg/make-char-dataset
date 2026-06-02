# make_char_dataset

> ComfyUI/diffusers image generation -> kohya-ready character LoRA dataset pipeline

[English version](README.md)

`make-char-dataset` превращает **паспортный набор** персонажа — золотой набор
референсов из [create-char-passport](https://github.com/hleserg/create-char-passport)
(`state.json` + размеченные ролями `refs/`) — в **kohya-ready датасет персонажной
LoRA**. Это шаг *локального размножения* в пайплайне комикс-персонажей: несколько
якорей идентичности размножаются в ~30–40 разнообразных кадров в едином стиле с
помощью **обученной стилевой LoRA** (Style Locker) и коммерчески-безопасного
бэкенда img2img + ControlNet, затем дедуп, капшенинг (Character-Locker) и раскладка
под kohya.

---

## Как это вписано в пайплайн

Доктрина: **консистентность — в персонаже, разнообразие — во всём остальном.**
Платный API (Nano Banana) фиксирует идентичность (5 паспортных кадров + опц.
эмоции/наряды/предметы), а этот репозиторий размножает канон локально. Стиль
учится **отдельной LoRA** (Style Locker) и грузится как внешняя LoRA при генерации,
поэтому персонажная LoRA несёт только геометрию лица/тела, без стиля. InsightFace-
инструменты (PuLID / IP-Adapter / InstantID) для коммерческого датасета не
используются; поза/структура — через license-safe ControlNet.

Стадии (см. [docs/architecture/WORKSPACE.md](docs/architecture/WORKSPACE.md)):
`00_passport_import` → `01_generated` → `02_clean` (дедуп) →
`03_dataset/<repeats>_<trigger>` (картинки + `.txt` капшены). Пайплайн
перезапускаемый (`.stage_complete` + `--force`), тяжёлый бэкенд генерации внедрён
за Protocol — тесты и CI идут без GPU.

## Быстрый старт

```bash
uv sync --all-extras
cp .env.example .env
uv run make-char-dataset --version
make check
uv run python .claude/skills/verifier-dataset/smoke.py   # бесплатная проверка без GPU
```

## Инструменты

uv (окружение/зависимости), ruff (линт+формат), pyright (типы),
pytest (тесты, >=90%), bandit/pip-audit (безопасность), pre-commit,
commitizen (conventional commits -> версия + changelog), Sentry.

## Лицензия

MIT — см. [LICENSE](LICENSE). Если форкаешь copyleft-код (например AGPL) — поменяй.
