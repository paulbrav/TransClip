import subprocess
import unittest
from unittest.mock import patch

from transclip.settings import Settings
from transclip.shell_command import (
    ShellCommandProcessor,
    commented_diagnostic,
    detect_default_shell,
    parse_shell_command,
    shell_command_messages,
    shell_generation_failure_reason,
    validate_shell_command,
)

from tests.service_helpers import FakeRuntime, FakeTextBackend


class FailingTextBackend:
    name = "failing"
    model_name = "missing-model"

    def generate(self, messages, *, max_new_tokens):
        raise RuntimeError("transformers dependencies missing")


class ShellCommandTests(unittest.TestCase):
    def test_structured_response_parsing(self):
        self.assertEqual(parse_shell_command('{"command": "ls -la"}'), "ls -la")

    def test_empty_structured_command_is_rejected(self):
        self.assertEqual(parse_shell_command('{"command": ""}'), "")

    def test_prose_and_fence_stripping(self):
        self.assertEqual(parse_shell_command("```bash\nls -la\n```"), "ls -la")
        self.assertEqual(parse_shell_command("Command: pwd"), "pwd")

    def test_bash_syntax_validation_passes(self):
        runtime = FakeRuntime(available={"bash": "/usr/bin/bash"}, run_func=lambda _command, **_kwargs: completed(0))

        result = validate_shell_command("ls -la", Settings(), runtime=runtime)

        self.assertTrue(result.ok)
        self.assertEqual(runtime.run_calls, [["/usr/bin/bash", "-n", "-c", "ls -la"]])

    def test_bash_syntax_validation_fails(self):
        runtime = FakeRuntime(
            available={"bash": "/usr/bin/bash"},
            run_func=lambda _command, **_kwargs: completed(2, stderr="bash: syntax error near unexpected token `fi'\n"),
        )

        result = validate_shell_command("if true; then echo hi; fi fi", Settings(), runtime=runtime)

        self.assertFalse(result.ok)
        self.assertIn("syntax error", result.diagnostics[0])

    def test_shellcheck_syntax_errors_block_but_warnings_do_not(self):
        results = iter(
            [
                completed(0),
                completed(1, stdout="In - line 1:\necho $x\n     ^-- SC2086 (info): Double quote\n"),
                completed(0),
                completed(1, stdout="In - line 1:\nif then\n^-- SC1073 (error): Couldn't parse\n"),
            ]
        )
        runtime = FakeRuntime(
            available={"bash": "/usr/bin/bash", "shellcheck": "/usr/bin/shellcheck"},
            run_func=lambda _command, **_kwargs: next(results),
        )

        warning = validate_shell_command("echo $x", Settings(), runtime=runtime)
        syntax_error = validate_shell_command("if then", Settings(), runtime=runtime)

        self.assertTrue(warning.ok)
        self.assertFalse(syntax_error.ok)
        self.assertIn("SC1073", syntax_error.diagnostics[0])

    def test_bash_syntax_validation_timeout_blocks_command(self):
        runtime = FakeRuntime(
            available={"bash": "/usr/bin/bash"},
            run_func=lambda _command, **_kwargs: raise_timeout(["bash"]),
        )

        result = validate_shell_command("ls -la", Settings(), runtime=runtime)

        self.assertFalse(result.ok)
        self.assertIn("timed out", result.diagnostics[0])
        self.assertTrue(result.metadata["bash_timeout"])

    def test_shellcheck_timeout_does_not_block_after_bash_passes(self):
        results = iter([completed(0), subprocess.TimeoutExpired(["shellcheck"], 2.0)])

        def run(_command, **_kwargs):
            result = next(results)
            if isinstance(result, Exception):
                raise result
            return result

        runtime = FakeRuntime(available={"bash": "/usr/bin/bash", "shellcheck": "/usr/bin/shellcheck"}, run_func=run)

        result = validate_shell_command("ls -la", Settings(), runtime=runtime)

        self.assertTrue(result.ok)
        self.assertTrue(result.metadata["shellcheck_timeout"])

    @patch("transclip.shell_command.validate_shell_command")
    def test_processor_renders_command_only_and_no_submit_newline(self, validate):
        validate.return_value = type("Validation", (), {"ok": True, "diagnostics": [], "metadata": {"bash": True}})()
        processor = ShellCommandProcessor(Settings(), FakeTextBackend('{"command": "ls -la\\n"}'))

        result = processor.generate("list files")

        self.assertEqual(result.text, "ls -la")
        self.assertFalse(result.text.endswith("\n"))
        self.assertTrue(result.valid)

    @patch("transclip.shell_command.validate_shell_command")
    def test_processor_uses_small_shell_generation_budget(self, validate):
        validate.return_value = type("Validation", (), {"ok": True, "diagnostics": [], "metadata": {}})()
        backend = FakeTextBackend('{"command": "pwd"}')
        processor = ShellCommandProcessor(Settings(), backend)

        processor.generate("print current directory")

        self.assertEqual(backend.max_new_tokens, [64])

    @patch("transclip.shell_command.validate_shell_command")
    def test_invalid_command_renders_commented_diagnostic(self, validate):
        validate.return_value = type(
            "Validation",
            (),
            {"ok": False, "diagnostics": ["syntax error"], "metadata": {"bash_returncode": 2}},
        )()
        processor = ShellCommandProcessor(Settings(), FakeTextBackend('{"command": "if then"}'))

        result = processor.generate("bad command")

        self.assertFalse(result.valid)
        self.assertEqual(result.command, "if then")
        self.assertTrue(result.text.startswith("# TransClip could not produce valid Bash: syntax error"))

    def test_processor_renders_diagnostic_when_model_generation_fails(self):
        processor = ShellCommandProcessor(Settings(), FailingTextBackend())

        result = processor.generate("list files")

        self.assertFalse(result.valid)
        self.assertEqual(result.command, "")
        self.assertEqual(result.backend, "failing")
        self.assertEqual(result.model, "missing-model")
        self.assertIn("model generation failed", result.diagnostics[0])
        self.assertTrue(result.text.startswith("# TransClip could not produce valid Bash: model generation failed"))

    def test_huggingface_cache_failure_is_short_and_actionable(self):
        reason = shell_generation_failure_reason(
            RuntimeError(
                "We couldn't connect to 'https://huggingface.co' to load the files, "
                "and couldn't find them in the cached files. Check your internet connection "
                "or see how to run the library in offline mode at "
                "'https://huggingface.co/docs/transformers/installation#offline-mode'."
            ),
            FakeTextBackend("{}"),
        )

        self.assertIn("fake-model is not available in the local Hugging Face cache", reason)
        self.assertIn("uv run -m transclip.cli models prefetch --model fake-model", reason)
        self.assertNotIn("https://huggingface.co", reason)

    def test_prompt_requests_json_command_only(self):
        messages = shell_command_messages("list files", default_shell_path="/usr/bin/zsh")

        self.assertIn("Return JSON only", messages[0]["content"])
        self.assertIn('"command"', messages[0]["content"])
        self.assertIn("zsh (/usr/bin/zsh)", messages[0]["content"])
        self.assertIn("Bash-compatible", messages[0]["content"])
        self.assertEqual(messages[1]["content"], "list files")

    @patch.dict("os.environ", {"SHELL": "/bin/fish"}, clear=True)
    def test_detect_default_shell_prefers_environment_shell(self):
        self.assertEqual(detect_default_shell(), "/bin/fish")

    @patch("transclip.shell_command.validate_shell_command")
    @patch.dict("os.environ", {"SHELL": "/usr/bin/zsh"}, clear=True)
    def test_processor_prompt_includes_default_shell(self, validate):
        validate.return_value = type("Validation", (), {"ok": True, "diagnostics": [], "metadata": {}})()
        backend = FakeTextBackend('{"command": "ls -la"}')
        processor = ShellCommandProcessor(Settings(), backend)

        processor.generate("list files")

        self.assertIn("zsh (/usr/bin/zsh)", backend.messages[0][0]["content"])

    def test_commented_diagnostic_comments_every_line(self):
        self.assertEqual(commented_diagnostic("first\nsecond"), "# first\n# second")


def completed(returncode: int, stdout: str = "", stderr: str = ""):
    return type("Completed", (), {"returncode": returncode, "stdout": stdout, "stderr": stderr})()


def raise_timeout(command: list[str]):
    raise subprocess.TimeoutExpired(command, 2.0)


if __name__ == "__main__":
    unittest.main()
