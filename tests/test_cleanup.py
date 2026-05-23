import unittest

from transclip.cleanup import (
    MODEL_CLEANUP_POLICY,
    CleanupPlan,
    FaithfulRuleCleanupBackend,
    ModelCleanupProcessor,
    apply_dictation_cleanup,
    conservative_cleanup,
)
from transclip.settings import Settings

from tests.service_helpers import FakeTextBackend


class CleanupTests(unittest.TestCase):
    def test_conservative_cleanup_punctuation_and_capitalization(self):
        self.assertEqual(
            conservative_cleanup("hello world this is PyTorch"),
            "Hello world this is PyTorch.",
        )

    def test_cleanup_backend_returns_timing(self):
        result = FaithfulRuleCleanupBackend().cleanup("hello ,world")
        self.assertEqual(result.text, "Hello, world.")
        self.assertIn("cleanup", result.timings_ms)
        self.assertEqual(result.backend, "rule-based")

    def test_cleanup_plan_defaults_to_rule_dictation(self):
        plan = CleanupPlan.from_settings(Settings())

        self.assertEqual(plan.dictation_mode, "rule")
        self.assertTrue(plan.requires_text_model)
        self.assertEqual(
            plan.backend_label(rule_name="rule-based", text_backend="transformers", text_model="Qwen/Qwen3.5-4B"),
            "rule-based",
        )

    def test_cleanup_plan_uses_model_dictation_when_always_on(self):
        plan = CleanupPlan.from_settings(Settings(voice_model_cleanup_always_on=True))

        self.assertEqual(plan.dictation_mode, "model")
        self.assertTrue(plan.requires_text_model)
        self.assertEqual(
            plan.backend_label(rule_name="rule-based", text_backend="transformers", text_model="Qwen/Qwen3.5-4B"),
            "transformers:Qwen/Qwen3.5-4B",
        )

    def test_cleanup_plan_skips_text_model_when_routing_disabled_and_always_on_off(self):
        plan = CleanupPlan.from_settings(
            Settings(voice_mode_routing_enabled=False, voice_model_cleanup_always_on=False)
        )

        self.assertEqual(plan.dictation_mode, "rule")
        self.assertFalse(plan.requires_text_model)

    def test_cleanup_plan_requires_transformers_text_runtime(self):
        plan = CleanupPlan.from_settings(Settings(voice_model_cleanup_always_on=True, text_model_runtime="disabled"))

        self.assertEqual(plan.dictation_mode, "rule")
        self.assertFalse(plan.requires_text_model)

    def test_model_cleanup_prompt_contract_and_output_parsing(self):
        messages = MODEL_CLEANUP_POLICY.messages("um hello --flag /tmp/file")

        self.assertIn("Preserve meaning", messages[0]["content"])
        self.assertIn("Remove filler only", messages[0]["content"])
        self.assertIn("flags, identifiers, paths", messages[0]["content"])
        self.assertIn("Do not add facts", messages[0]["content"])
        self.assertEqual(messages[1]["content"], "Transcript:\num hello --flag /tmp/file")

    def test_model_cleanup_processor_uses_text_backend(self):
        backend = FakeTextBackend("Cleaned text")
        result = ModelCleanupProcessor(backend).cleanup("raw text")

        self.assertEqual(result.text, "Cleaned text")
        self.assertEqual(result.backend, "fake-text:fake-model")
        self.assertEqual(backend.messages[0][1]["content"], "Transcript:\nraw text")

    def test_model_cleanup_policy_rejects_empty_output(self):
        with self.assertRaisesRegex(RuntimeError, "fake cleanup produced an empty response"):
            MODEL_CLEANUP_POLICY.validate_output(" ", "fake")

    def test_apply_dictation_cleanup_uses_model_path_when_configured(self):
        text_backend = FakeTextBackend("Model cleaned")
        rule = FaithfulRuleCleanupBackend()
        model = ModelCleanupProcessor(text_backend)
        settings = Settings(voice_model_cleanup_always_on=True)
        plan = CleanupPlan.from_settings(settings)

        result = apply_dictation_cleanup("hello ,world", plan, rule_cleanup=rule, model_cleanup=model)

        self.assertEqual(result.text, "Model cleaned")
        self.assertEqual(result.backend, "fake-text:fake-model")

    def test_apply_dictation_cleanup_uses_rule_path_by_default(self):
        text_backend = FakeTextBackend("should not run")
        rule = FaithfulRuleCleanupBackend()
        model = ModelCleanupProcessor(text_backend)

        result = apply_dictation_cleanup(
            "hello ,world",
            CleanupPlan.from_settings(Settings()),
            rule_cleanup=rule,
            model_cleanup=model,
        )

        self.assertEqual(result.text, "Hello, world.")
        self.assertEqual(result.backend, "rule-based")
        self.assertEqual(text_backend.messages, [])


if __name__ == "__main__":
    unittest.main()
