import unittest

from transclip.cleanup import (
    FaithfulCleanupPolicy,
    FaithfulRuleCleanupBackend,
    GemmaTransformersCleanupBackend,
    ModelCleanupPolicy,
    ModelCleanupProcessor,
    build_cleanup_backend,
    conservative_cleanup,
    faithful_cleanup_messages,
    model_cleanup_messages,
)
from transclip.settings import Settings
from transclip.text_generation import TextGenerationResult


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

    def test_cleanup_backend_selection_is_explicit(self):
        backend = build_cleanup_backend(Settings(model_cache_dir="/models"))
        self.assertIsInstance(backend, FaithfulRuleCleanupBackend)
        gemma = build_cleanup_backend(Settings(cleanup_runtime="transformers", model_cache_dir="/models"))
        self.assertIsInstance(gemma, GemmaTransformersCleanupBackend)
        self.assertTrue(gemma.local_files_only)
        self.assertEqual(gemma.cache_dir, "/models")
        self.assertIsInstance(
            build_cleanup_backend(Settings(cleanup_runtime="test_rule")),
            FaithfulRuleCleanupBackend,
        )
        with self.assertRaises(ValueError):
            build_cleanup_backend(Settings(cleanup_runtime="unknown"))
        with self.assertRaisesRegex(ValueError, "Unsupported cleanup runtime: llama_cpp"):
            build_cleanup_backend(Settings(cleanup_runtime="llama_cpp"))

    def test_faithful_cleanup_messages_are_conservative(self):
        messages = faithful_cleanup_messages("hello pytorch")
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Do not add facts", messages[0]["content"])
        self.assertEqual(messages[1]["content"], "Transcript:\nhello pytorch")

    def test_faithful_cleanup_policy_is_provider_independent(self):
        policy = FaithfulCleanupPolicy()

        self.assertEqual(policy.token_budget("one two"), 64)
        self.assertEqual(policy.token_budget("word " * 300), 512)
        self.assertEqual(policy.validate_output(" cleaned ", "provider"), "cleaned")
        with self.assertRaisesRegex(RuntimeError, "provider cleanup produced an empty response"):
            policy.validate_output("   ", "provider")

    def test_model_cleanup_prompt_contract_and_output_parsing(self):
        messages = model_cleanup_messages("um hello --flag /tmp/file")

        self.assertIn("Preserve meaning", messages[0]["content"])
        self.assertIn("Remove filler only", messages[0]["content"])
        self.assertIn("flags, identifiers, paths", messages[0]["content"])
        self.assertIn("Do not add facts", messages[0]["content"])
        self.assertEqual(messages[1]["content"], "Transcript:\num hello --flag /tmp/file")

    def test_model_cleanup_processor_uses_text_backend(self):
        backend = FakeTextBackend("Cleaned text")
        result = ModelCleanupProcessor(backend).cleanup("raw text")

        self.assertEqual(result.text, "Cleaned text")
        self.assertEqual(result.backend, "fake:qwen")
        self.assertEqual(backend.messages[0][1]["content"], "Transcript:\nraw text")

    def test_model_cleanup_policy_rejects_empty_output(self):
        with self.assertRaisesRegex(RuntimeError, "fake cleanup produced an empty response"):
            ModelCleanupPolicy().validate_output(" ", "fake")


class FakeTextBackend:
    name = "fake"
    model_name = "qwen"

    def __init__(self, response: str):
        self.response = response
        self.messages = []

    def generate(self, messages, *, max_new_tokens):
        self.messages.append(messages)
        return TextGenerationResult(self.response, {"text_generation": 1.0}, self.name, self.model_name)


if __name__ == "__main__":
    unittest.main()
