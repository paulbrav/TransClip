## Summary

<!-- What changed and why? For macOS integration work: master cleanup preserved; macOS port is capability-only; hotkey fix included. For Windows integration: Granite autoregressive ASR only (no NAR); CUDA GPU path; Task Scheduler service. -->

## Test plan

- [ ] `uv run ruff check .`
- [ ] `uv run -m compileall transclip scripts tests`
- [ ] `uv run -m unittest discover -s tests -v`
- [ ] Manual smoke (if platform-specific): document steps here
