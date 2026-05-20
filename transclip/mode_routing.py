from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

VoiceMode = Literal["dictation", "cleanup", "shell"]

CLEANUP_TRIGGERS = ("clean up", "trans cleanup")
SHELL_TRIGGERS = ("shell command", "bash command", "terminal command")
TRIGGERS = CLEANUP_TRIGGERS + SHELL_TRIGGERS
LITERAL_PREFIX = "literal"


@dataclass(frozen=True, slots=True)
class VoiceModeRoute:
    mode: VoiceMode
    payload: str
    trigger: str | None = None
    literal: bool = False


@dataclass(frozen=True, slots=True)
class _TriggerMatch:
    payload: str
    trigger: str


def route_voice_mode(
    transcript: str,
    *,
    routing_enabled: bool = True,
    shell_enabled: bool = True,
) -> VoiceModeRoute:
    if not routing_enabled:
        return VoiceModeRoute("dictation", transcript)

    literal = _match_trigger(transcript, (LITERAL_PREFIX,))
    if literal is not None:
        escaped = _match_trigger(literal.payload, TRIGGERS)
        if escaped is not None and escaped.payload:
            return VoiceModeRoute("dictation", literal.payload, escaped.trigger, literal=True)
        return VoiceModeRoute("dictation", transcript)

    cleanup = _match_trigger(transcript, CLEANUP_TRIGGERS)
    if cleanup is not None:
        if cleanup.payload:
            return VoiceModeRoute("cleanup", cleanup.payload, cleanup.trigger)
        return VoiceModeRoute("dictation", transcript)

    shell = _match_trigger(transcript, SHELL_TRIGGERS)
    if shell is not None:
        if shell.payload and shell_enabled:
            return VoiceModeRoute("shell", shell.payload, shell.trigger)
        return VoiceModeRoute("dictation", transcript)

    return VoiceModeRoute("dictation", transcript)


def _match_trigger(text: str, triggers: tuple[str, ...]) -> _TriggerMatch | None:
    stripped = text.lstrip()
    for trigger in triggers:
        pattern = re.compile(rf"^{re.escape(trigger)}(?:$|[\s,.:;!?-]+)", re.IGNORECASE)
        match = pattern.match(stripped)
        if not match:
            continue
        payload = stripped[match.end() :].lstrip(" \t\r\n,.:;!?-")
        return _TriggerMatch(payload, trigger)
    return None
