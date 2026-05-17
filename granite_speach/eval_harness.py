from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .service import InferenceEngine


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
        keywords = case.get("keywords", engine.keywords)
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


def run_keyword_ablation(manifest_path: Path, engine: InferenceEngine) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    warmup_cases = manifest.get("warmup_cases", []) if isinstance(manifest, dict) else []
    cases = manifest["cases"] if isinstance(manifest, dict) else manifest
    for case in warmup_cases:
        audio_path = resolve_audio_path(manifest_path, case)
        keywords = case.get("keywords", engine.keywords)
        engine.transcribe(audio_path, cleanup=case.get("cleanup"), keywords=keywords)
        engine.transcribe(audio_path, cleanup=case.get("cleanup"), keywords=[])

    results: list[dict[str, Any]] = []
    for case in cases:
        audio_path = resolve_audio_path(manifest_path, case)
        keywords = case.get("keywords", engine.keywords)
        with_keywords = engine.transcribe(
            audio_path,
            cleanup=case.get("cleanup"),
            keywords=keywords,
        )
        without_keywords = engine.transcribe(
            audio_path,
            cleanup=case.get("cleanup"),
            keywords=[],
        )
        reference = case.get("reference")
        with_score = keyword_preservation(with_keywords["text"], keywords)
        without_score = keyword_preservation(without_keywords["text"], keywords)
        delta = with_score - without_score if with_score is not None and without_score is not None else None
        results.append(
            {
                "audio_path": str(audio_path),
                "reference": reference,
                "keywords": keywords,
                "with_keywords_text": with_keywords["text"],
                "without_keywords_text": without_keywords["text"],
                "with_keywords_preservation": with_score,
                "without_keywords_preservation": without_score,
                "keyword_preservation_delta": delta,
                "with_keywords_wer": word_error_rate(reference, with_keywords["text"]) if reference else None,
                "without_keywords_wer": word_error_rate(reference, without_keywords["text"]) if reference else None,
                "with_keywords_timings_ms": with_keywords["timings_ms"],
                "without_keywords_timings_ms": without_keywords["timings_ms"],
            }
        )
    return summarize_keyword_ablation(results, warmup_cases=len(warmup_cases))


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


def summarize_keyword_ablation(
    results: list[dict[str, Any]],
    warmup_cases: int = 0,
) -> dict[str, Any]:
    deltas = [
        result["keyword_preservation_delta"] for result in results if result["keyword_preservation_delta"] is not None
    ]
    with_scores = [
        result["with_keywords_preservation"] for result in results if result["with_keywords_preservation"] is not None
    ]
    without_scores = [
        result["without_keywords_preservation"]
        for result in results
        if result["without_keywords_preservation"] is not None
    ]
    with_wers = [result["with_keywords_wer"] for result in results if result["with_keywords_wer"] is not None]
    without_wers = [result["without_keywords_wer"] for result in results if result["without_keywords_wer"] is not None]
    return {
        "summary": {
            "cases": len(results),
            "warmup_cases": warmup_cases,
            "keyword_cases": len(deltas),
            "mean_with_keywords_preservation": mean(with_scores) if with_scores else None,
            "mean_without_keywords_preservation": mean(without_scores) if without_scores else None,
            "mean_keyword_preservation_delta": mean(deltas) if deltas else None,
            "improved_cases": sum(1 for delta in deltas if delta > 0),
            "regressed_cases": sum(1 for delta in deltas if delta < 0),
            "mean_with_keywords_wer": mean(with_wers) if with_wers else None,
            "mean_without_keywords_wer": mean(without_wers) if without_wers else None,
        },
        "results": results,
    }


def optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref:
        return 0.0 if not hyp else 1.0
    distances = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        distances[i][0] = i
    for j in range(len(hyp) + 1):
        distances[0][j] = j
    for i, ref_word in enumerate(ref, start=1):
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            distances[i][j] = min(
                distances[i - 1][j] + 1,
                distances[i][j - 1] + 1,
                distances[i - 1][j - 1] + cost,
            )
    return distances[-1][-1] / len(ref)


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
