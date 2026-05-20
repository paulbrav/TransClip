import sys
import threading
import time
import unittest
from unittest.mock import patch

from transclip.text_generation import TransformersTextGenerationBackend


class TextGenerationTests(unittest.TestCase):
    def test_transformers_backend_serializes_lazy_load_and_generation(self):
        state = FakeTransformersState()
        backend = TransformersTextGenerationBackend("local/text-model")

        with (
            patch.dict(
                sys.modules,
                {
                    "transformers": state.transformers_module(),
                    "torch": FakeTorch(),
                },
            ),
            patch("transclip.text_generation.resolve_torch_device", return_value="cpu"),
        ):
            threads = [
                threading.Thread(
                    target=backend.generate,
                    args=([{"role": "user", "content": "task"}],),
                    kwargs={"max_new_tokens": 8},
                )
                for _ in range(4)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(state.tokenizer_loads, 1)
        self.assertEqual(state.model_loads, 1)
        self.assertEqual(state.model.max_active_generations, 1)

    def test_transformers_backend_disables_chat_template_thinking(self):
        state = FakeTransformersState()
        backend = TransformersTextGenerationBackend("local/text-model")

        with (
            patch.dict(
                sys.modules,
                {
                    "transformers": state.transformers_module(),
                    "torch": FakeTorch(),
                },
            ),
            patch("transclip.text_generation.resolve_torch_device", return_value="cpu"),
        ):
            backend.generate([{"role": "user", "content": "shell task"}], max_new_tokens=8)

        self.assertEqual(state.tokenizer.template_kwargs["enable_thinking"], False)


class FakeTransformersState:
    def __init__(self):
        self.tokenizer_loads = 0
        self.model_loads = 0
        self.tokenizer = FakeTokenizer()
        self.model = FakeModel()

    def transformers_module(self):
        state = self

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                state.tokenizer_loads += 1
                time.sleep(0.01)
                return state.tokenizer

        class AutoModelForCausalLM:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                state.model_loads += 1
                time.sleep(0.01)
                return state.model

        return type(
            "TransformersModule",
            (),
            {
                "AutoProcessor": AutoTokenizer,
                "AutoModelForImageTextToText": AutoModelForCausalLM,
            },
        )()


class FakeTokenizer:
    eos_token_id = 1

    def __init__(self):
        self.template_kwargs = {}

    def apply_chat_template(self, *_args, **_kwargs):
        self.template_kwargs = _kwargs
        return FakeInputs()

    def decode(self, *_args, **_kwargs):
        return "generated"


class FakeInputs(dict):
    def __init__(self):
        super().__init__({"input_ids": type("InputIds", (), {"shape": (1, 3)})()})

    def to(self, _device):
        return self


class FakeModel:
    device = "cpu"

    def __init__(self):
        self.generation_config = type("GenerationConfig", (), {})()
        self.active_generations = 0
        self.max_active_generations = 0
        self.lock = threading.Lock()

    def eval(self):
        return None

    def generate(self, **_kwargs):
        with self.lock:
            self.active_generations += 1
            self.max_active_generations = max(self.max_active_generations, self.active_generations)
        time.sleep(0.01)
        with self.lock:
            self.active_generations -= 1
        return [[0, 1, 2, 3]]


class FakeTorch:
    def inference_mode(self):
        return self

    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False


if __name__ == "__main__":
    unittest.main()
