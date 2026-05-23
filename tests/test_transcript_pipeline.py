import unittest

from transclip.cleanup import FaithfulRuleCleanupBackend, ModelCleanupProcessor
from transclip.mode_routing import route_voice_mode
from transclip.settings import Settings
from transclip.shell_command import ShellCommandProcessor
from transclip.transcript_pipeline import TranscriptProcessor

from tests.service_helpers import FakeTextBackend


class TranscriptPipelineTests(unittest.TestCase):
    def setUp(self):
        self.rule_cleanup = FaithfulRuleCleanupBackend()
        self.settings = Settings(shellcheck_enabled=False)

    def _processor(
        self,
        settings: Settings | None = None,
        text_backend: FakeTextBackend | None = None,
    ) -> TranscriptProcessor:
        settings = settings or self.settings
        backend = text_backend or FakeTextBackend(["unused"])
        return TranscriptProcessor(
            settings,
            rule_cleanup=self.rule_cleanup,
            model_cleanup=ModelCleanupProcessor(backend),
            shell_command=ShellCommandProcessor(settings, backend),
        )

    def test_normal_dictation_applies_rule_cleanup(self):
        text_backend = FakeTextBackend(["unused"])
        outcome = self._processor(text_backend=text_backend).process(
            "hello ,world",
            route_voice_mode("hello ,world"),
            asr_backend="fake",
            asr_model="fake-model",
        )

        self.assertEqual(outcome.text, "Hello, world.")
        self.assertEqual(outcome.voice_mode, "dictation")
        self.assertEqual(outcome.cleanup_backend, "rule-based")
        self.assertEqual(text_backend.messages, [])

    def test_explicit_cleanup_trigger_uses_model_cleanup(self):
        text_backend = FakeTextBackend(["Model-cleaned text"])
        outcome = self._processor(text_backend=text_backend).process(
            "clean up hello ,world",
            route_voice_mode("clean up hello ,world"),
            asr_backend="fake",
            asr_model="fake-model",
        )

        self.assertEqual(outcome.text, "Model-cleaned text")
        self.assertEqual(outcome.voice_mode, "cleanup")
        self.assertEqual(outcome.voice_trigger, "clean up")
        self.assertIn("hello ,world", text_backend.messages[0][1]["content"])

    def test_literal_escape_skips_cleanup(self):
        text_backend = FakeTextBackend(["unused"])
        outcome = self._processor(text_backend=text_backend).process(
            "literal shell command list files",
            route_voice_mode("literal shell command list files"),
            asr_backend="fake",
            asr_model="fake-model",
        )

        self.assertEqual(outcome.text, "shell command list files")
        self.assertTrue(outcome.voice_literal)
        self.assertIsNone(outcome.cleanup)

    def test_always_on_model_cleanup_applies_to_normal_dictation(self):
        text_backend = FakeTextBackend(["Model cleaned normal dictation"])
        settings = Settings(voice_model_cleanup_always_on=True)
        outcome = self._processor(settings=settings, text_backend=text_backend).process(
            "hello ,world",
            route_voice_mode("hello ,world"),
            asr_backend="fake",
            asr_model="fake-model",
        )

        self.assertEqual(outcome.text, "Model cleaned normal dictation")
        self.assertEqual(outcome.voice_mode, "dictation")
        self.assertEqual(len(text_backend.messages), 1)

    def test_shell_trigger_sets_submit_false(self):
        text_backend = FakeTextBackend(['{"command": "ls -la"}'])
        outcome = self._processor(text_backend=text_backend).process(
            "shell command list files",
            route_voice_mode("shell command list files"),
            asr_backend="fake",
            asr_model="fake-model",
        )

        self.assertEqual(outcome.text, "ls -la")
        self.assertEqual(outcome.voice_mode, "shell")
        self.assertIs(outcome.submit, False)


if __name__ == "__main__":
    unittest.main()
