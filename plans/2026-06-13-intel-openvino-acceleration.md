# Intel HW Acceleration for TransClip via OpenVINO (Windows v1) — AS-BUILT

> Status: **Shipped.** Committed on branch `intel-accel` and submitted as
> [paulbrav/TransClip#43](https://github.com/paulbrav/TransClip/pull/43)
> (fork `sumgup0:intel-accel` → `paulbrav:master`). 21 files changed, +948/−10.
> Full suite: **377 passed, 2 skipped**; `compileall` clean.
>
> This document was updated to reflect what was actually written. A short
> "Deltas vs original plan" section at the end records where the as-built code
> diverged from the pre-implementation design.

## Context

TransClip's heavy models run only on NVIDIA CUDA (Windows/Linux), AMD ROCm
(Linux), or Apple MLX (macOS). On a typical Windows laptop with an Intel CPU +
integrated GPU (Iris Xe / Arc) and, on Core Ultra 1/2, an NPU, there was **no
acceleration path** — those machines fell back to `windows_cpu` (slow CPU-torch)
in [profiles.py](transclip/platform/profiles.py).

The current default ASR model, IBM **Granite Speech 4.1**, is a custom
speech-encoder + LLM architecture that **OpenVINO / `optimum-intel` do not
support**. So the Intel story is **"swap Granite→Whisper for ASR + route Qwen
through OpenVINO for cleanup"**, not "port Granite." Whisper is first-class on
OpenVINO GenAI (`WhisperPipeline`) and TransClip already ships a Whisper backend
on Apple Silicon; Qwen is a supported OpenVINO LLM (`LLMPipeline`). Pre-converted
OpenVINO IR is downloaded from HF, so there is no on-device conversion.

**Scope shipped:** Windows only; accelerate **both** ASR and text cleanup;
default device **AUTO** (OpenVINO picks NPU/iGPU/CPU); multiple Whisper builds
(int4 default, int8 accuracy, base lightweight, community turbo-NPU) and two OV
Qwen builds. Linux OpenVINO was deliberately deferred (its profile branch assumes
CUDA and would need a guard).

**Outcome:** on an Intel Windows laptop with no NVIDIA GPU, a fresh install
auto-selects an OpenVINO Whisper backend + OpenVINO Qwen cleanup, accelerated on
the iGPU/NPU. CUDA/ROCm/MLX/CPU paths are untouched.

## Design principle: keep OpenVINO device strings out of the torch path

OpenVINO uses its own device namespace (`CPU`, `GPU`, `GPU.0`, `NPU`, `AUTO`) via
`openvino.Core().available_devices`, separate from
[`resolve_torch_device`](transclip/device.py:11) (cpu/cuda/mps/rocm). The OV
backends route `asr_device` through `resolve_openvino_device` only; the torch
resolver is never called for them. `asr_device` is overloaded with namespaced
values (`auto`, `openvino:AUTO|GPU|NPU|CPU`) — no new ASR-device setting.

## What was built (file-by-file, as shipped)

### 1. `transclip/openvino_device.py` (new)
Import-safe device layer (never raises, so profile detection can call it):
- `openvino_available_devices() -> tuple[str, ...]` — `@lru_cache`; `import
  openvino; tuple(openvino.Core().available_devices)`, `except Exception: ()`.
  No subprocess smoke test (enumeration is cheap and allocates no device memory).
- `has_intel_accelerator() -> bool` — any `NPU`/`GPU`/`GPU.*` device.
- `resolve_openvino_device(requested="auto") -> str` — `auto`→`AUTO`; strips an
  `openvino:` prefix; `CPU` always allowed; explicit `GPU`/`NPU` verified against
  available devices (raises `RuntimeError` if absent); unknown → `ValueError`.

### 2. ASR backend: `transclip/asr.py`
- `OpenVINOWhisperBackend` (mirrors `MlxAudioASRBackend`: lazy, `RLock`-guarded
  load from a local HF snapshot via `mlx_snapshot_path`). `_load()` imports
  `openvino_genai`, sets `STATIC_PIPELINE=True` when device == `NPU`, builds
  `WhisperPipeline(model_dir, device)`. `_read_audio` folds to mono + linear
  resamples into a 1-D float32 numpy array fed straight to `generate()`.
  `transcribe()` drops `keywords` (Whisper has no biasing) with a one-time DEBUG
  log. Timing keys: `model_load` / `audio_prepare` / `generate`.
- `build_asr_backend` dispatch branch for `openvino_whisper`, routing
  `settings.asr_device` to the OV resolver (not `resolve_torch_device`).
- Helpers `_whisper_language_token` (`en`→`<|en|>`) and `_openvino_result_text`.
- MLX backend timing key renamed `generate_write`→`generate` to unify the
  snapshot-backend timing schema (no code consumes the key; verified by grep).

### 3. Text-gen backend: `transclip/text_generation.py`
- `OpenVINOTextGenerationBackend` using `openvino_genai.LLMPipeline`; lazy,
  lock-guarded; resolves the model dir via `mlx_snapshot_path`; device via
  `resolve_openvino_device`.
- `_render_chat_prompt` calls the OV tokenizer's `apply_chat_template` with
  `enable_thinking=False` (suppresses Qwen `<think>` traces), guarded by
  `try/except TypeError` for tokenizers/versions that reject the kwarg.
- `build_text_generation_backend` dispatch for `text_model_runtime == "openvino"`.

### 4. Catalog: `transclip/models/catalog.py` + `types.py`
- `ModelRuntimeKind` widened with `"openvino"`.
- **ASR Whisper entries** (`backend="openvino_whisper"`, `runtime_kind="openvino"`,
  `prefetch_strategy="snapshot_download"`, `dependency_extra="openvino"`,
  `supported_platforms={"Windows"}`):
  - `OpenVINO/whisper-large-v3-int4-ov` — **default** (fast int4)
  - `OpenVINO/whisper-large-v3-int8-ov` — higher accuracy
  - `OpenVINO/whisper-base-int8-ov` — lightweight / smoke
  - `FluidInference/whisper-large-v3-turbo-int4-ov-npu` — turbo, NPU-pre-exported
- **Text Qwen entries** (`backend="text_generation"`, `runtime_kind="openvino"`,
  `snapshot_download`, `supported_platforms={"Linux","Windows"}`):
  - `OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov` — default (lightweight)
  - `OpenVINO/Qwen2.5-7B-Instruct-int4-ov` — higher quality
- `ASR_BACKEND_ALIASES`: `openvino` / `openvino_whisper` / `ov` / `ov_whisper`
  → `openvino_whisper`.
- `validate_asr_model_backend`: guard requiring `"whisper"` in the model id
  (defensive; matches the existing granite-name guards).

### 5. Prefetch / cache — unchanged
`prefetch_strategy="snapshot_download"` already routes to
`huggingface_hub.snapshot_download`; `mlx_snapshot_path` is repo-agnostic and is
reused directly. (The optional `openvino_snapshot_path` alias from the original
plan was **not** added — the direct reuse was cleaner.) Only cosmetic change:
generalized the "MLX" wording in the prefetch ImportError message.

### 6. Platform profile + settings: `platform/profiles.py`, `settings.py`
- `ProfileRuntimeKind` += `"openvino"`; `ProfileId` += `"windows_openvino"`.
- `RuntimeProfile` gained `default_text_model_runtime="transformers"` and
  `default_text_model="Qwen/Qwen3.5-4B"` (defaults match `Settings`, so all other
  profiles are behavior-preserving).
- Windows branch: between the CUDA check and the CPU fallback, `if
  has_intel_accelerator(): return windows_openvino` with
  `default_asr_backend="openvino_whisper"`,
  `default_asr_model="OpenVINO/whisper-large-v3-int4-ov"`,
  `default_asr_device="auto"`, `default_text_model_runtime="openvino"`,
  `default_text_model="OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov"`. Selection order:
  **CUDA → Intel-OpenVINO → CPU-torch**.
- `default_settings` wires `text_model_runtime` and `text_model` from the profile.

### 7. Cleanup gating: `transclip/cleanup.py` (review fix — not in original plan)
`CleanupPlan.from_settings` previously hard-coded `text_model_runtime ==
"transformers"`, which on the `windows_openvino` profile silently disabled model
cleanup **and** hid the OV text model from doctor's cache check (while the engine
still built/called it → runtime failure on the first shell command). Fixed to
treat `text_model_runtime in {"transformers", "openvino"}` as model-capable.

### 8. Doctor: `transclip/doctor/asr.py`
`build_backend_checks` branch for `runtime_kind == "openvino"` →
`check_openvino_runtime`: imports `openvino` + `openvino_genai`, lists
`available_devices`, validates the requested device — the user-facing surface for
the **OS-driver prerequisite**. Exported from `transclip/doctor/__init__.py`.

### 9. Packaging / docs
- `pyproject.toml`: `openvino = ["openvino-genai>=2025.0", "openvino>=2025.0",
  "huggingface_hub>=0.27", "soundfile>=0.12", "numpy>=1.26"]`. `uv.lock` updated
  with openvino / openvino-genai / openvino-tokenizers / openvino-telemetry.
- `README.md`: "Intel acceleration (integrated GPU / NPU) via OpenVINO" section
  (install, prefetch, `asr_device` values, model list, caveats).

## Risks / sharp edges (as shipped)
- **Granite→Whisper is a product change:** different accuracy/latency, and
  keyword biasing is lost on Intel (documented; keywords silently dropped).
- **GPU/NPU OS drivers are out-of-band** (pip ships runtime, not drivers); doctor
  detects and explains.
- **NPU** needs static shapes (`STATIC_PIPELINE=True`, handled); best on Core
  Ultra Series 2; cold compile latency (mitigated by lazy load; OpenVINO
  `CACHE_DIR` warm-up is a future lever, not implemented).
- **Not validated on real Intel hardware** — no Intel HW on the dev host; the
  OpenVINO **CPU plugin** path is the proxy that was exercised.

## Verification (as run)
On this Windows host `ruff`/`ty` cannot execute (Defender blocks the native-binary
wheels), so the gate was `compileall` + `pytest` rather than `make check`:
- `.venv/Scripts/python.exe -m compileall -q transclip tests` — clean.
- `.venv/Scripts/python.exe -m pytest tests/ -q` — **377 passed, 2 skipped**.
- New tests: `tests/test_openvino_device.py` (resolver, has-accelerator),
  `test_asr.py` (selection, keyword-drop, NPU `STATIC_PIPELINE` + CPU negative,
  missing-artifacts error), `test_models.py` (catalog/alias/validation,
  Windows-only listing), `test_platform_runtime.py` (windows_openvino +
  regression), `test_text_generation.py` (LLMPipeline path, `enable_thinking`
  fallback, runtime dispatch), `test_cleanup.py` (openvino runtime gating),
  `test_doctor.py` (openvino runtime check).
- End-to-end on Windows: `transclip models list` shows the OV Whisper + Qwen
  entries with correct backend routing.
- ruff lint pre-audited against the configured rule set (`B,C4,E,F,I,RUF,SIM,UP`,
  line-length 120); fixed SIM117/RUF012 in new tests; kept the `elif`+nested-`if`
  catalog guard (matches existing granite branches on green `master`).

## Critical files
- [transclip/openvino_device.py](transclip/openvino_device.py) — device resolver (new)
- [transclip/asr.py](transclip/asr.py) — `OpenVINOWhisperBackend` + dispatch
- [transclip/text_generation.py](transclip/text_generation.py) — OV `LLMPipeline` backend
- [transclip/models/catalog.py](transclip/models/catalog.py) + [types.py](transclip/models/types.py)
- [transclip/platform/profiles.py](transclip/platform/profiles.py) + [settings.py](transclip/settings.py)
- [transclip/cleanup.py](transclip/cleanup.py) — model-capable runtime gate
- [transclip/doctor/asr.py](transclip/doctor/asr.py) — OpenVINO runtime check
- [pyproject.toml](pyproject.toml) — `openvino` extra (+ `uv.lock`)

## Deltas vs original plan
1. **Default ASR model**: planned `whisper-large-v3-turbo int4`; shipped
   `OpenVINO/whisper-large-v3-int4-ov` (no *official* turbo exists — turbo is a
   community catalog option instead).
2. **OV Qwen entries**: planned one; shipped two (1.5B default + 7B).
3. **`cleanup.py`**: not in the plan; added as a critical review fix so cleanup +
   doctor recognize the OpenVINO text runtime.
4. **`openvino_snapshot_path` alias**: planned (optional); not added — reused
   `mlx_snapshot_path` directly.
5. **Settings schema**: plan said "no schema change"; in practice `RuntimeProfile`
   gained two text-model default fields and `default_settings` was wired to them
   (the `Settings` dataclass itself is unchanged).
6. **Timing keys**: unified MLX `generate_write`→`generate` (review follow-up;
   not in plan).
7. **`enable_thinking=False`** guard in OV text-gen (review follow-up; not in plan).
8. **Verification**: plan referenced `make check`; actual gate was
   `compileall` + `pytest` due to the host Defender quirk.
