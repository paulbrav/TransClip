from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .service import InferenceEngine


def _manifest_gate_thresholds(thresholds: dict[str, Any]) -> tuple[float, float, float, float, float]:
    if "mean_release_to_ready_max_ms" in thresholds:
        mean_max_ms = float(thresholds["mean_release_to_ready_max_ms"])
        worst_max_ms = float(thresholds.get("worst_release_to_ready_max_ms", mean_max_ms))
    elif "release_to_ready_p95_ms" in thresholds:
        mean_max_ms = float(thresholds["release_to_ready_p95_ms"])
        worst_max_ms = float(thresholds.get("worst_release_to_ready_max_ms", mean_max_ms))
    else:
        raise ValueError("manifest thresholds require mean_release_to_ready_max_ms")
    if "under_700_min_ratio" in thresholds:
        under_700_ratio = float(thresholds["under_700_min_ratio"])
    else:
        under_700_ratio = 0.0 if mean_max_ms > 700.0 else 0.8
    return (
        mean_max_ms,
        worst_max_ms,
        under_700_ratio,
        float(thresholds["keyword_preservation_min"]),
        float(thresholds["wer_max"]),
    )


@dataclass(slots=True)
class EvalCaseResult:
    audio_path: str
    text: str
    raw_asr: str
    reference: str | None
    wer: float | None
    raw_asr_wer: float | None
    cleanup_drift_wer_delta: float | None
    cleanup_semantic_drift: bool | None
    keyword_preservation: float | None
    paste_attempted: bool | None
    paste_success: bool | None
    timings_ms: dict[str, float]


@dataclass(frozen=True, slots=True)
class EvalGatePolicy:
    min_cases: int = 20
    max_cases: int = 30
    max_mean_latency_ms: float = 700.0
    max_latency_ms: float = 1500.0
    min_under_700_ratio: float = 0.8
    min_keyword_preservation: float = 0.9
    max_mean_wer: float = 0.25
    max_cleanup_drift_failures: int = 0
    max_paste_failures: int = 0

    @classmethod
    def from_manifest(cls, path: Path) -> EvalGatePolicy:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be a JSON object")
        thresholds = manifest.get("thresholds")
        if not isinstance(thresholds, dict):
            raise ValueError("manifest must include thresholds")
        cases = manifest.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError("manifest must include at least one measured case")
        case_count = len(cases)
        mean_max_ms, worst_max_ms, under_700_ratio, keyword_min, wer_max = _manifest_gate_thresholds(thresholds)
        return cls(
            min_cases=case_count,
            max_cases=case_count,
            max_mean_latency_ms=mean_max_ms,
            max_latency_ms=worst_max_ms,
            min_under_700_ratio=under_700_ratio,
            min_keyword_preservation=keyword_min,
            max_mean_wer=wer_max,
        )

    def check_results(self, payload: dict[str, Any]) -> dict[str, Any]:
        summary = payload.get("summary")
        results = payload.get("results")
        if not isinstance(summary, dict) or not isinstance(results, list):
            raise ValueError("results JSON must contain summary and results")

        cases = int(summary.get("cases", len(results)))
        if cases != len(results):
            raise ValueError(f"summary cases {cases} does not match results length {len(results)}")
        if cases < self.min_cases or cases > self.max_cases:
            raise ValueError(f"expected {self.min_cases}-{self.max_cases} measured cases, found {cases}")

        latencies = [float(result.get("timings_ms", {}).get("end_to_end", 0.0)) for result in results]
        if not latencies or any(value <= 0 for value in latencies):
            raise ValueError("each result must include a positive timings_ms.end_to_end")
        worst_latency = max(latencies)
        if worst_latency > self.max_latency_ms:
            raise ValueError(f"worst latency {worst_latency:.3f}ms exceeds {self.max_latency_ms:.3f}ms")

        mean_latency = float(summary.get("mean_release_to_ready_ms") or 0.0)
        if mean_latency <= 0:
            raise ValueError("summary.mean_release_to_ready_ms must be positive")
        if mean_latency > self.max_mean_latency_ms:
            raise ValueError(f"mean latency {mean_latency:.3f}ms exceeds {self.max_mean_latency_ms:.3f}ms")

        under_700 = int(summary.get("under_700ms", 0))
        under_700_ratio = under_700 / cases
        if under_700_ratio < self.min_under_700_ratio:
            raise ValueError(f"under-700ms ratio {under_700_ratio:.3f} is below {self.min_under_700_ratio:.3f}")

        keyword_score = summary.get("mean_keyword_preservation")
        if keyword_score is not None and float(keyword_score) < self.min_keyword_preservation:
            raise ValueError(
                f"mean keyword preservation {float(keyword_score):.3f} is below {self.min_keyword_preservation:.3f}"
            )

        mean_wer = summary.get("mean_wer")
        if mean_wer is not None and float(mean_wer) > self.max_mean_wer:
            raise ValueError(f"mean WER {float(mean_wer):.3f} exceeds {self.max_mean_wer:.3f}")

        cleanup_drift_failures = int(summary.get("cleanup_semantic_drift_failures", 0))
        if cleanup_drift_failures > self.max_cleanup_drift_failures:
            raise ValueError(
                f"cleanup semantic drift failures {cleanup_drift_failures} exceed {self.max_cleanup_drift_failures}"
            )

        paste_failures = int(summary.get("paste_failures", 0))
        if paste_failures > self.max_paste_failures:
            raise ValueError(f"paste failures {paste_failures} exceed {self.max_paste_failures}")

        return {
            "status": "pass",
            "cases": cases,
            "mean_release_to_ready_ms": mean_latency,
            "worst_release_to_ready_ms": worst_latency,
            "under_700_ratio": under_700_ratio,
            "mean_keyword_preservation": keyword_score,
            "mean_wer": mean_wer,
            "cleanup_semantic_drift_failures": cleanup_drift_failures,
            "paste_failures": paste_failures,
        }


def run_eval(manifest_path: Path, engine: InferenceEngine) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    warmup_cases = manifest.get("warmup_cases", []) if isinstance(manifest, dict) else []
    cases = manifest["cases"] if isinstance(manifest, dict) else manifest
    for case in warmup_cases:
        transcribe_case(manifest_path, engine, case)
    results: list[EvalCaseResult] = []
    for case in cases:
        audio_path, result = transcribe_case(manifest_path, engine, case)
        reference = case.get("reference")
        keywords = case.get("keywords", [])
        cleaned_wer = word_error_rate(reference, result["text"]) if reference else None
        raw_asr_wer = word_error_rate(reference, result["raw_asr"]) if reference else None
        drift_delta = cleanup_drift_delta(raw_asr_wer, cleaned_wer)
        results.append(
            EvalCaseResult(
                audio_path=str(audio_path),
                text=result["text"],
                raw_asr=result["raw_asr"],
                reference=reference,
                wer=cleaned_wer,
                raw_asr_wer=raw_asr_wer,
                cleanup_drift_wer_delta=drift_delta,
                cleanup_semantic_drift=is_cleanup_semantic_drift(drift_delta),
                keyword_preservation=keyword_preservation(result["text"], keywords),
                paste_attempted=optional_bool(case.get("paste_attempted")),
                paste_success=optional_bool(case.get("paste_success")),
                timings_ms=result["timings_ms"],
            )
        )
    summary = summarize(results)
    summary["summary"]["warmup_cases"] = len(warmup_cases)
    return summary


def transcribe_case(
    manifest_path: Path,
    engine: InferenceEngine,
    case: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    audio_path = resolve_audio_path(manifest_path, case)
    return audio_path, engine.transcribe(
        audio_path,
        cleanup=case.get("cleanup"),
        keywords=case.get("keywords"),
    )


def resolve_audio_path(manifest_path: Path, case: dict[str, Any]) -> Path:
    audio_path = Path(case["audio_path"]).expanduser()
    if not audio_path.is_absolute():
        audio_path = manifest_path.parent / audio_path
    return audio_path


def summarize(results: list[EvalCaseResult]) -> dict[str, Any]:
    wers = [result.wer for result in results if result.wer is not None]
    raw_asr_wers = [result.raw_asr_wer for result in results if result.raw_asr_wer is not None]
    drift_deltas = [result.cleanup_drift_wer_delta for result in results if result.cleanup_drift_wer_delta is not None]
    keyword_scores = [result.keyword_preservation for result in results if result.keyword_preservation is not None]
    release_latencies = [result.timings_ms.get("end_to_end", 0.0) for result in results]
    return {
        "summary": {
            "cases": len(results),
            "mean_wer": mean(wers) if wers else None,
            "mean_raw_asr_wer": mean(raw_asr_wers) if raw_asr_wers else None,
            "mean_cleanup_drift_wer_delta": mean(drift_deltas) if drift_deltas else None,
            "cleanup_semantic_drift_failures": sum(1 for result in results if result.cleanup_semantic_drift),
            "mean_keyword_preservation": mean(keyword_scores) if keyword_scores else None,
            "mean_release_to_ready_ms": mean(release_latencies) if release_latencies else None,
            "under_700ms": sum(1 for value in release_latencies if value < 700),
            "under_1500ms": sum(1 for value in release_latencies if value < 1500),
            "paste_attempts": sum(1 for result in results if result.paste_attempted),
            "paste_successes": sum(1 for result in results if result.paste_success),
            "paste_failures": sum(1 for result in results if result.paste_attempted and not result.paste_success),
        },
        "results": [asdict(result) for result in results],
    }


def optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = word_tokens(reference)
    hyp = word_tokens(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
        previous = current
    return previous[-1] / len(ref)


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def cleanup_drift_delta(raw_asr_wer: float | None, cleaned_wer: float | None) -> float | None:
    if raw_asr_wer is None or cleaned_wer is None:
        return None
    return max(0.0, cleaned_wer - raw_asr_wer)


def is_cleanup_semantic_drift(drift_delta: float | None, threshold: float = 0.05) -> bool | None:
    if drift_delta is None:
        return None
    return drift_delta > threshold


def keyword_preservation(text: str, keywords: list[str]) -> float | None:
    if not keywords:
        return None
    lowered = text.lower()
    preserved = sum(1 for keyword in keywords if keyword.lower() in lowered)
    return preserved / len(keywords)


def build_manifest(
    clip_dir: Path,
    output_path: Path = Path("eval/real-usage/manifest.json"),
    warmup_stem: str | None = None,
    global_keywords: list[str] | None = None,
    min_cases: int = 20,
    max_cases: int = 30,
    allow_small: bool = False,
) -> dict[str, list[dict[str, object]]]:
    clip_dir = clip_dir.expanduser().resolve()
    if not clip_dir.is_dir():
        raise ValueError(f"clip directory does not exist: {clip_dir}")

    global_keywords = global_keywords or []
    entries = []
    for wav_path in sorted(clip_dir.glob("*.wav")):
        reference_path = wav_path.with_suffix(".txt")
        if not reference_path.exists():
            raise ValueError(f"missing reference text for {wav_path.name}: {reference_path.name}")
        reference = reference_path.read_text(encoding="utf-8").strip()
        if not reference:
            raise ValueError(f"empty reference text: {reference_path}")
        entries.append(
            {
                "audio_path": manifest_path(wav_path, output_path),
                "reference": reference,
                "keywords": keywords_for_case(wav_path, reference, global_keywords),
                "cleanup": True,
            }
        )

    if not entries:
        raise ValueError(f"no .wav clips found in {clip_dir}")

    warmup_cases = []
    measured = entries
    if warmup_stem:
        warmup_name = f"{warmup_stem}.wav"
        warmup_cases = [entry for entry in entries if Path(str(entry["audio_path"])).name == warmup_name]
        if not warmup_cases:
            raise ValueError(f"warmup stem not found: {warmup_stem}")
        measured = [entry for entry in entries if Path(str(entry["audio_path"])).name != warmup_name]

    if len(measured) > max_cases:
        raise ValueError(f"expected at most {max_cases} measured clips, found {len(measured)}")
    if len(measured) < min_cases and not allow_small:
        raise ValueError(
            f"expected at least {min_cases} measured clips, found {len(measured)}; "
            "pass --allow-small only for smoke tests"
        )

    manifest: dict[str, list[dict[str, object]]] = {"cases": measured}
    if warmup_cases:
        manifest = {"warmup_cases": warmup_cases, "cases": measured}
    return manifest


def keywords_for_case(wav_path: Path, reference: str, global_keywords: list[str]) -> list[str]:
    keywords = []
    reference_lower = reference.lower()
    for keyword in global_keywords:
        if keyword.lower() in reference_lower:
            keywords.append(keyword)
    for suffix in (".keywords.txt", ".keywords"):
        keyword_path = wav_path.with_suffix(suffix)
        if keyword_path.exists():
            keywords.extend(load_keyword_file(keyword_path))
            break
    return list(dict.fromkeys(keywords))


def load_keyword_file(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.expanduser().read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def manifest_path(path: Path, output_path: Path) -> str:
    output_dir = output_path.expanduser().resolve().parent
    relative = os.path.relpath(path.resolve(), output_dir)
    return relative.replace("\\", "/")


def check_results(
    payload: dict[str, Any],
    min_cases: int = 20,
    max_cases: int = 30,
    max_mean_latency_ms: float = 700.0,
    max_latency_ms: float = 1500.0,
    min_under_700_ratio: float = 0.8,
    min_keyword_preservation: float = 0.9,
    max_mean_wer: float = 0.25,
    max_cleanup_drift_failures: int = 0,
    max_paste_failures: int = 0,
) -> dict[str, Any]:
    return EvalGatePolicy(
        min_cases=min_cases,
        max_cases=max_cases,
        max_mean_latency_ms=max_mean_latency_ms,
        max_latency_ms=max_latency_ms,
        min_under_700_ratio=min_under_700_ratio,
        min_keyword_preservation=min_keyword_preservation,
        max_mean_wer=max_mean_wer,
        max_cleanup_drift_failures=max_cleanup_drift_failures,
        max_paste_failures=max_paste_failures,
    ).check_results(payload)
