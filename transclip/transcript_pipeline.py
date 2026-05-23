from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .cleanup import (
    CleanupBackend,
    CleanupPlan,
    CleanupResult,
    ModelCleanupProcessor,
    apply_dictation_cleanup,
)
from .mode_routing import VoiceModeRoute
from .settings import Settings
from .shell_command import ShellCommandProcessor, ShellCommandResult


@dataclass(slots=True)
class TranscriptOutcome:
    text: str
    raw_asr: str
    voice_mode: str
    voice_trigger: str | None
    voice_literal: bool
    cleanup: CleanupResult | None
    shell: ShellCommandResult | None
    submit: bool | None
    timings_ms: dict[str, float]
    asr_backend: str
    asr_model: str
    cleanup_backend: str
    cleanup_enabled: bool

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "text": self.text,
            "raw_asr": self.raw_asr,
            "voice_mode": self.voice_mode,
            "voice_trigger": self.voice_trigger,
            "voice_literal": self.voice_literal,
            "cleanup": asdict(self.cleanup) if self.cleanup else None,
            "shell": shell_metadata(self.shell),
            "submit": self.submit,
            "timings_ms": dict(self.timings_ms),
            "asr_backend": self.asr_backend,
            "asr_model": self.asr_model,
            "cleanup_backend": self.cleanup_backend,
            "cleanup_enabled": self.cleanup_enabled,
        }
        return payload


class TranscriptProcessor:
    def __init__(
        self,
        settings: Settings,
        *,
        rule_cleanup: CleanupBackend,
        model_cleanup: ModelCleanupProcessor,
        shell_command: ShellCommandProcessor,
    ):
        self.settings = settings
        self.rule_cleanup = rule_cleanup
        self.model_cleanup = model_cleanup
        self.shell_command = shell_command

    def cleanup_dictation(self, text: str) -> CleanupResult:
        plan = CleanupPlan.from_settings(self.settings)
        return apply_dictation_cleanup(
            text,
            plan,
            rule_cleanup=self.rule_cleanup,
            model_cleanup=self.model_cleanup,
        )

    def process(
        self,
        raw_asr: str,
        route: VoiceModeRoute,
        *,
        cleanup: bool | None = None,
        asr_backend: str,
        asr_model: str,
        timings_ms: dict[str, float] | None = None,
    ) -> TranscriptOutcome:
        should_cleanup = self.settings.cleanup_enabled if cleanup is None else cleanup
        plan = CleanupPlan.from_settings(self.settings)
        timings = dict(timings_ms or {})
        cleanup_result: CleanupResult | None = None
        shell_result: ShellCommandResult | None = None
        cleaned = route.payload if route.literal else raw_asr
        cleanup_backend = self.rule_cleanup.name

        if route.mode == "cleanup":
            cleanup_result = self._run_model_cleanup(route.payload, timings)
            cleaned = cleanup_result.text
            cleanup_backend = cleanup_result.backend
        elif route.mode == "shell":
            shell_result = self.shell_command.generate(route.payload)
            cleaned = shell_result.text
            timings.update(shell_result.timings_ms)
        elif should_cleanup and not route.literal:
            cleanup_result = apply_dictation_cleanup(
                raw_asr,
                plan,
                rule_cleanup=self.rule_cleanup,
                model_cleanup=self.model_cleanup,
            )
            cleaned = cleanup_result.text
            timings.update(cleanup_result.timings_ms)
            cleanup_backend = cleanup_result.backend

        return TranscriptOutcome(
            text=cleaned,
            raw_asr=raw_asr,
            voice_mode=route.mode,
            voice_trigger=route.trigger,
            voice_literal=route.literal,
            cleanup=cleanup_result,
            shell=shell_result,
            submit=False if route.mode == "shell" else None,
            timings_ms=timings,
            asr_backend=asr_backend,
            asr_model=asr_model,
            cleanup_backend=cleanup_backend,
            cleanup_enabled=should_cleanup,
        )

    def _run_model_cleanup(self, text: str, timings: dict[str, float]) -> CleanupResult:
        result = self.model_cleanup.cleanup(text)
        timings.update(result.timings_ms)
        return result


def shell_metadata(shell_result: ShellCommandResult | None) -> dict[str, Any] | None:
    if shell_result is None:
        return None
    return {
        "command": shell_result.command,
        "valid": shell_result.valid,
        "diagnostics": shell_result.diagnostics,
        "backend": shell_result.backend,
        "model": shell_result.model,
        "validation": shell_result.validation,
    }
