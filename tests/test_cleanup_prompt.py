import importlib
import sys
import types
import unittest
from unittest import mock


class CleanupPromptTests(unittest.TestCase):
    def test_custom_prompt_used(self) -> None:
        llama_mod = types.ModuleType("llama_cpp")

        class DummyLlama:
            def __init__(self, *args, **kwargs) -> None:
                self.last_prompt = None

            def __call__(self, prompt: str, max_tokens: int = 0, temperature: float = 0.0):
                self.last_prompt = prompt
                return {"choices": [{"text": "CLEAN"}]}

        llama_mod.Llama = DummyLlama
        trans_module = types.ModuleType("transclip.transcription")
        trans_module.WhisperModelType = type("WhisperModelType", (), {})
        patches = {"llama_cpp": llama_mod, "transclip.transcription": trans_module}
        with mock.patch.dict(sys.modules, patches):
            import transclip.cleanup as cleanup_mod
            importlib.reload(cleanup_mod)
            cfg = {
                "cleanup_enable": True,
                "cleanup_llm_model": "model.gguf",
                "cleanup_punct_model": "none",
                "cleanup_prompt": "PROMPT {text}",
            }
            cleaner = cleanup_mod.Cleaner(cfg)
            result = cleaner("hello")
            self.assertEqual(result, "CLEAN")
            assert isinstance(cleaner.llm, DummyLlama)
            self.assertEqual(cleaner.llm.last_prompt, "PROMPT hello")


if __name__ == "__main__":
    unittest.main()
