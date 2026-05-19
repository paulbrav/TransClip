# V1 Synthetic Eval

This directory defines a repeatable local benchmark for the V1 dictation path.
It is intentionally synthetic: clips are generated from technical-note reference
sentences with the locally installed Piper voice. This does not replace the
required 20-30 real-usage clip evaluation, but it gives the repo a stable
latency, WER, keyword-preservation, and cleanup regression gate.

Generate 16 kHz mono audio:

```bash
python3 scripts/generate_synthetic_eval.py
```

Run with the measured Linux `gfx1151` runtime:

```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
.venv-gfx1151/bin/python -m transclip.cli eval \
  eval/v1-synthetic/manifest.json \
  --output eval/v1-synthetic/results.json
```

The manifest contains one warmup case and 25 measured cases. Current steady
result after ROCm/MIOpen setup: 25/25 clips under 700 ms, mean latency
284.6 ms, mean WER 0.1125, and mean keyword preservation 1.0.
