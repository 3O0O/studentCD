from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from student_sim_cd import inference, prompt_baselines
from tests.test_inference import sample


class Backend:
    eos_id = 256

    def __init__(self, config, fail_after=None, truncated_greedy=False):
        self.config, self.fail_after = config, fail_after
        self.truncated_greedy = truncated_greedy
        self.generated = []

    def prompt_tokens(self, messages):
        return [100, len(messages[0]["content"])]

    def code_tokens(self, code):
        return list(code.encode()) + [self.eos_id]

    def generate(self, prompt, sample_id, attempt):
        if self.fail_after is not None and len(self.generated) >= self.fail_after:
            raise RuntimeError("synthetic interruption")
        self.generated.append((sample_id, attempt))
        # Explicit duplicates test that attempts are not topped up.
        code = "print(1)\n" if attempt % 2 else "```python\nprint(2)\n```\n"
        complete = not (self.truncated_greedy and attempt == 0)
        tokens = self.code_tokens(code)
        if not complete:
            tokens = [65] * self.config.max_new_tokens
            code = 'A' * self.config.max_new_tokens
        return {"attempt": attempt, "source": "greedy" if attempt == 0 else "sampled",
            "seed": int(inference.object_hash([self.config.seed, sample_id, attempt])[:8], 16),
            "raw_text": code, "generated_token_ids": tokens,
            "eos_reached": complete, "finish_reason": "eos" if complete else "length_limit"}

    def sequence_logp(self, prompt, completion):
        return -float(len(completion))


class PromptBaselineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name).resolve() / "baseline"
        self.config = inference.InferenceConfig(model_path="synthetic", seed=20260929)
        self.rows = [sample()]
        self.references = {}

    def run_baselines(self, backend, resume=False):
        with redirect_stdout(io.StringIO()):
            return prompt_baselines.run(self.rows, self.references, self.config, backend,
                self.output, protocol_sha256="a" * 64, resume=resume)

    def test_full_attempts_dedup_exact_extraction_and_predict_without_labels(self):
        backend = Backend(self.config)
        result = self.run_baselines(backend)
        self.assertEqual(len(backend.generated), 26)
        self.assertEqual(result["attempts"], 26)
        self.assertFalse(result["labels_read"])
        self.assertFalse(result["student_execution"])
        for method in prompt_baselines.PROMPTS:
            greedy = inference.read_jsonl(self.output / (method + "_greedy.jsonl"))
            self.assertEqual(greedy[0]["predicted_code"], "print(2)\n")
        self.assertEqual((self.output / "SUCCESS").read_text().strip(),
                         inference.file_hash(self.output / "manifest.json"))
        # Completion reuse performs no hidden extra generation.
        other = Backend(self.config)
        self.assertEqual(self.run_baselines(other, resume=True), result)
        self.assertEqual(other.generated, [])

    def test_uncertain_generation_is_not_automatically_repeated(self):
        with self.assertRaisesRegex(RuntimeError, "interruption"):
            self.run_baselines(Backend(self.config, fail_after=5))
        original = (self.output / "raw.jsonl").read_bytes()
        backend = Backend(self.config)
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            self.run_baselines(backend, resume=True)
        self.assertEqual(backend.generated, [])
        self.assertEqual((self.output / "raw.jsonl").read_bytes(), original)

    def test_scoring_failure_resumes_only_missing_generation(self):
        backend = Backend(self.config)
        backend.sequence_logp = lambda *_: (_ for _ in ()).throw(RuntimeError('score failure'))
        with self.assertRaisesRegex(RuntimeError, 'score failure'):
            self.run_baselines(backend)
        original = (self.output / 'raw.jsonl').read_bytes()
        other = Backend(self.config)
        self.run_baselines(other, resume=True)
        self.assertEqual(len(other.generated), 13)
        self.assertTrue((self.output / 'raw.jsonl').read_bytes().startswith(original))

    def test_greedy_truncation_fails_without_fake_copy_or_success(self):
        with self.assertRaisesRegex(ValueError, "no substitute"):
            self.run_baselines(Backend(self.config, truncated_greedy=True))
        self.assertFalse((self.output / "SUCCESS").exists())
        self.assertEqual(len(inference.read_jsonl(self.output / "raw.jsonl")), 1)

    def test_corrupt_checkpoint_refuses_before_new_generation(self):
        with self.assertRaises(RuntimeError):
            self.run_baselines(Backend(self.config, fail_after=2))
        path = self.output / "raw.jsonl"
        rows = inference.read_jsonl(path)
        rows[0]["raw_text"] = "changed"
        path.write_text("".join(inference.canonical_json(row) + "\n" for row in rows))
        backend = Backend(self.config)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.run_baselines(backend, resume=True)
        self.assertEqual(backend.generated, [])

    def test_prompt_data_keeps_actual_feedback_but_no_labels_or_ids(self):
        for method in prompt_baselines.PROMPTS:
            messages = prompt_baselines.messages(self.rows[0], self.references, method)
            payload = json.loads(messages[1]["content"])
            self.assertEqual(payload["current_feedback"], self.rows[0]["feedback"])
            self.assertNotIn("sample_id", payload)
            self.assertNotIn("target_code", payload)
            self.assertEqual(payload["history"], self.rows[0]["history"])

    def test_rehashed_bad_raw_eos_refuses_before_new_generation(self):
        backend = Backend(self.config)
        backend.sequence_logp = lambda *_: (_ for _ in ()).throw(RuntimeError('score failure'))
        with self.assertRaises(RuntimeError):
            self.run_baselines(backend)
        path = self.output / 'raw.jsonl'
        records = inference.read_jsonl(path)
        records[0]['generated_token_ids'][-1] = 65
        records[0]['record_sha256'] = inference.object_hash(
            {key: value for key, value in records[0].items() if key != 'record_sha256'})
        path.write_text(''.join(inference.canonical_json(item) + '\n' for item in records))
        other = Backend(self.config)
        with self.assertRaisesRegex(ValueError, 'raw generation provenance'):
            self.run_baselines(other, resume=True)
        self.assertEqual(other.generated, [])

    def test_future_label_rejected_before_output_or_generation(self):
        self.rows[0]['target_code'] = 'future'
        backend = Backend(self.config)
        with self.assertRaisesRegex(ValueError, 'target_code'):
            self.run_baselines(backend)
        self.assertEqual(backend.generated, [])
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
