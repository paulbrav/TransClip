from __future__ import annotations

import json
import os
import pwd
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transclip.platform.runtime import PlatformRuntime, get_runtime

from .settings import Settings
from .text_generation import TextGenerationBackend

SHELL_SYNTAX_ERROR_CODES = ("SC107", "SC108")
SHELL_VALIDATION_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class ShellValidationResult:
    ok: bool
    diagnostics: list[str]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ShellCommandResult:
    text: str
    command: str
    valid: bool
    diagnostics: list[str]
    timings_ms: dict[str, float]
    backend: str
    model: str
    validation: dict[str, Any]


class ShellCommandProcessor:
    def __init__(
        self,
        settings: Settings,
        text_backend: TextGenerationBackend,
        runtime: PlatformRuntime | None = None,
    ):
        self.settings = settings
        self.text_backend = text_backend
        self.runtime = get_runtime(runtime)

    def generate(self, task: str) -> ShellCommandResult:
        try:
            generation = self.text_backend.generate(shell_command_messages(task), max_new_tokens=64)
        except Exception as exc:
            reason = shell_generation_failure_reason(exc, self.text_backend)
            diagnostic = commented_diagnostic("TransClip could not produce valid Bash: " + reason)
            return ShellCommandResult(
                diagnostic,
                "",
                False,
                [reason],
                {},
                getattr(self.text_backend, "name", "text-generation"),
                getattr(self.text_backend, "model_name", ""),
                {},
            )
        command = parse_shell_command(generation.text)
        if not command:
            diagnostic = commented_diagnostic("TransClip could not produce valid Bash: empty command")
            return ShellCommandResult(
                diagnostic,
                "",
                False,
                ["empty command"],
                dict(generation.timings_ms),
                generation.backend,
                generation.model,
                {},
            )

        validation = validate_shell_command(command, self.settings, runtime=self.runtime)
        if not validation.ok:
            diagnostic = commented_diagnostic(
                "TransClip could not produce valid Bash: " + "; ".join(validation.diagnostics)
            )
            return ShellCommandResult(
                diagnostic,
                command,
                False,
                validation.diagnostics,
                dict(generation.timings_ms),
                generation.backend,
                generation.model,
                validation.metadata,
            )

        return ShellCommandResult(
            command.rstrip("\r\n"),
            command.rstrip("\r\n"),
            True,
            validation.diagnostics,
            dict(generation.timings_ms),
            generation.backend,
            generation.model,
            validation.metadata,
        )


def shell_command_messages(task: str, default_shell_path: str | None = None) -> list[dict[str, str]]:
    default_shell_path = default_shell_path or detect_default_shell()
    default_shell_name = Path(default_shell_path).name or default_shell_path
    return [
        {
            "role": "system",
            "content": (
                'Return JSON only: {"command":"..."}. Generate one Bash-compatible command for the task. '
                f"Default shell: {default_shell_name} ({default_shell_path}). No prose or Markdown."
            ),
        },
        {
            "role": "user",
            "content": task,
        },
    ]


def detect_default_shell() -> str:
    shell = os.environ.get("SHELL", "").strip()
    if shell:
        return shell
    try:
        shell = pwd.getpwuid(os.getuid()).pw_shell.strip()
    except (KeyError, OSError):
        shell = ""
    return shell or "/bin/sh"


def parse_shell_command(response: str) -> str:
    text = response.strip()
    if not text:
        return ""
    parsed = _parse_json_command(text)
    if parsed is not None:
        return parsed.rstrip("\r\n")
    fenced = _first_fenced_block(text)
    if fenced is not None:
        text = fenced.strip()
        parsed = _parse_json_command(text)
        if parsed is not None:
            return parsed.rstrip("\r\n")
    text = re.sub(r"^\s*(?:command|bash)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip().rstrip("\r\n")


def validate_shell_command(
    command: str,
    settings: Settings,
    runtime: PlatformRuntime | None = None,
) -> ShellValidationResult:
    platform_runtime = get_runtime(runtime)
    metadata: dict[str, Any] = {}
    diagnostics: list[str] = []
    if not settings.shell_syntax_validation_enabled:
        return ShellValidationResult(True, diagnostics, metadata)

    bash = platform_runtime.which("bash")
    metadata["bash_available"] = bool(bash)
    if bash:
        try:
            completed = platform_runtime.run(
                [bash, "-n", "-c", command],
                text=True,
                capture_output=True,
                check=False,
                timeout=SHELL_VALIDATION_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            metadata["bash_timeout"] = True
            diagnostics.append(f"bash syntax check timed out after {SHELL_VALIDATION_TIMEOUT_SECONDS:.1f}s")
            return ShellValidationResult(False, diagnostics, metadata)
        metadata["bash_returncode"] = completed.returncode
        if completed.stderr.strip():
            metadata["bash_stderr"] = completed.stderr.strip()
        if completed.returncode != 0:
            diagnostics.append(_single_line(completed.stderr) or "bash syntax check failed")
            return ShellValidationResult(False, diagnostics, metadata)

    shellcheck = platform_runtime.which("shellcheck") if settings.shellcheck_enabled else None
    metadata["shellcheck_available"] = bool(shellcheck)
    if shellcheck:
        try:
            completed = platform_runtime.run(
                [shellcheck, "-s", "bash", "-"],
                input=command,
                text=True,
                capture_output=True,
                check=False,
                timeout=SHELL_VALIDATION_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            metadata["shellcheck_timeout"] = True
            return ShellValidationResult(True, diagnostics, metadata)
        output = "\n".join(part for part in (completed.stdout, completed.stderr) if part.strip())
        metadata["shellcheck_returncode"] = completed.returncode
        if output.strip():
            metadata["shellcheck_output"] = output.strip()
        if completed.returncode != 0 and _shellcheck_blocks(output):
            diagnostics.append(_single_line(output) or "ShellCheck syntax check failed")
            return ShellValidationResult(False, diagnostics, metadata)

    return ShellValidationResult(True, diagnostics, metadata)


def commented_diagnostic(message: str) -> str:
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    if not lines:
        lines = ["TransClip could not produce valid Bash"]
    return "\n".join(f"# {line}" for line in lines)


def shell_generation_failure_reason(exc: Exception, text_backend: TextGenerationBackend) -> str:
    model = getattr(text_backend, "model_name", "")
    detail = _single_line(str(exc)) or type(exc).__name__
    lowered = detail.lower()
    if "cached files" in lowered or "couldn't connect" in lowered or "offline mode" in lowered:
        if model:
            return (
                f"text model {model} is not available in the local Hugging Face cache; "
                f"run: uv run -m transclip.cli models prefetch --model {model}"
            )
        return "text model is not available in the local Hugging Face cache"
    if "install transclip[models]" in lowered or ("transformers" in lowered and "required" in lowered):
        return "text model dependencies are missing; install: uv pip install -e '.[models]'"
    return f"model generation failed: {detail}"


def _parse_json_command(text: str) -> str | None:
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            for key in ("command", "cmd", "text"):
                if key in payload:
                    command = payload[key]
                    return command.strip() if isinstance(command, str) else ""
            return ""
    return None


def _first_fenced_block(text: str) -> str | None:
    match = re.search(r"```(?:bash|sh|shell)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return match.group(1)


def _shellcheck_blocks(output: str) -> bool:
    lowered = output.lower()
    return "error:" in lowered or any(code in output for code in SHELL_SYNTAX_ERROR_CODES)


def _single_line(text: str) -> str:
    return " ".join(text.strip().split())
