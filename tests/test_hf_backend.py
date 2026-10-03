"""CPU-only equivalence checks using a tiny random Qwen2, with no downloads."""

import inspect
import unittest

from student_sim_cd.inference import HFBackend, InferenceConfig

try:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
except ImportError:
    torch = None
    Qwen2Config = Qwen2ForCausalLM = None


@unittest.skipIf(torch is None, "optional torch/transformers not installed")
class HFBackendEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(314159)
        config = Qwen2Config(
            vocab_size=41, hidden_size=24, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=256, attention_dropout=0.0,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
        )
        cls.model = Qwen2ForCausalLM(config).to("cpu").float().eval()

    def backend(self, model):
        # Bypass from_pretrained entirely: use the random in-memory CPU model.
        backend = HFBackend.__new__(HFBackend)
        backend.torch = torch
        backend.model = model
        backend.eos_id = 2
        backend.config = InferenceConfig(model_path="unused", device="cpu", dtype="float32",
                                         max_context_tokens=256, max_new_tokens=32)
        return backend

    def full_forward_reference(self, prompt, completion):
        """Independent full-forward definition: shift all logits, then slice targets."""
        ids = torch.tensor([prompt + completion], dtype=torch.long)
        with torch.inference_mode():
            logits = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
            logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
            selected = logp.gather(-1, ids[:, 1:, None]).squeeze(-1)
            return selected[0, len(prompt) - 1:].double().sum().item()

    def test_selected_positions_equal_full_forward_for_eos_and_multitoken(self):
        if "logits_to_keep" not in inspect.signature(self.model.forward).parameters:
            self.skipTest("installed Qwen2 does not expose logits_to_keep")
        calls = []
        underlying = self.model

        class RecordingModel:
            def forward(self, input_ids, attention_mask, use_cache, logits_to_keep=0):
                result = underlying(input_ids=input_ids, attention_mask=attention_mask,
                                    use_cache=use_cache, logits_to_keep=logits_to_keep)
                calls.append((logits_to_keep.tolist(), result.logits.shape))
                return result

            __call__ = forward

        backend = self.backend(RecordingModel())
        cases = [([1], [2]), ([1, 5, 8, 11], [2]),
                 ([1, 5, 8, 11], [6, 9, 13, 2]),
                 ([1, 5, 8, 11], [6] * 130 + [2])]
        for prompt, completion in cases:
            with self.subTest(prompt_length=len(prompt), completion_length=len(completion)):
                expected = self.full_forward_reference(prompt, completion)
                actual = backend.sequence_logp(prompt, completion)
                self.assertAlmostEqual(actual, expected, delta=1e-5)
                positions, shape = calls[-1]
                self.assertEqual(positions, list(range(len(prompt) - 1, len(prompt) + len(completion) - 1)))
                self.assertEqual(tuple(shape), (1, len(completion), 41))

    def test_model_without_explicit_parameter_uses_equivalent_full_fallback(self):
        calls = []
        underlying = self.model

        class LegacyModel:
            def forward(self, input_ids, attention_mask, use_cache):
                result = underlying(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache)
                calls.append(result.logits.shape)
                return result

            __call__ = forward

        backend = self.backend(LegacyModel())
        for completion in ([2], [6, 9, 13, 2], [6] * 130 + [2]):
            prompt = [1, 5, 8, 11]
            self.assertAlmostEqual(backend.sequence_logp(prompt, completion),
                                   self.full_forward_reference(prompt, completion), delta=1e-5)
            self.assertEqual(tuple(calls[-1]), (1, len(prompt) + len(completion), 41))

    def test_kwargs_alone_does_not_claim_logit_selection_support(self):
        underlying = self.model
        received = []

        class KwargsModel:
            def forward(self, **kwargs):
                received.append(kwargs)
                return underlying(**kwargs)

            __call__ = forward

        prompt, completion = [1, 7, 8], [9, 2]
        actual = self.backend(KwargsModel()).sequence_logp(prompt, completion)
        self.assertAlmostEqual(actual, self.full_forward_reference(prompt, completion), delta=1e-5)
        self.assertNotIn("logits_to_keep", received[0])

    def test_selected_logit_shape_mismatch_fails_loudly(self):
        if "logits_to_keep" not in inspect.signature(self.model.forward).parameters:
            self.skipTest("installed Qwen2 does not expose logits_to_keep")
        underlying = self.model

        class IgnoringModel:
            def forward(self, input_ids, attention_mask, use_cache, logits_to_keep=0):
                return underlying(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache)

            __call__ = forward

        with self.assertRaisesRegex(ValueError, "unexpected number of logit positions"):
            self.backend(IgnoringModel()).sequence_logp([1, 5, 8], [9, 2])


if __name__ == "__main__":
    unittest.main()
