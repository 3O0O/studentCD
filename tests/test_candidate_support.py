"""Synthetic full-cohort support regression; no model, GPU or student execution."""

from contextlib import redirect_stdout
import copy
from dataclasses import asdict, replace
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from student_sim_cd import cache_rescore, candidate_support as support, inference, predict
from tests.test_history_interventions import row
from tests.test_inference import FakeBackend


class Tokenizer:
    eos_token_id = 256
    all_special_ids = [256]
    chat_template = "synthetic chat template"

    def encode(self, text, *, add_special_tokens=False, truncation=False):
        if add_special_tokens or truncation:
            raise AssertionError("Canonical text protocol changed")
        return list(text.encode("utf-8"))

    def decode(self, tokens, *, skip_special_tokens, clean_up_tokenization_spaces):
        if skip_special_tokens or clean_up_tokenization_spaces:
            raise AssertionError("Original raw decode protocol changed")
        return bytes(tokens).decode("utf-8")

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, truncation):
        if not tokenize or not add_generation_prompt or truncation:
            raise AssertionError("Full untruncated prompt required")
        return list(messages[1]["content"].encode("utf-8"))


class Backend(FakeBackend):
    eos_id = 256

    def __init__(self, config, *, original=False, no_eos=False, bad_seed=False, nonfinite=False):
        super().__init__()
        self.config, self.original = config, original
        self.tokenizer = Tokenizer()
        self.no_eos, self.bad_seed, self.nonfinite = no_eos, bad_seed, nonfinite
        self.calls = []

    def generate(self, prompt, sample_id, attempt):
        payload = json.loads(bytes(prompt))
        condition = ("1" if payload["history"][0]["code"].startswith("query-") else "0") + ("1" if payload["current_feedback"] else "0")
        self.calls.append((sample_id, condition, attempt))
        if self.original:
            index = int(sample_id[1:])
            number = attempt + 1 if index < 38 or attempt < 3 else 3
            text = f"print({number})\n"
        else:
            text = f"```python\nprint({100 * int(condition, 2) + attempt})\n```\n"
        complete = not (self.no_eos and len(self.calls) == 1)
        if not complete:
            text = "x" * self.config.max_new_tokens
        seed = int(hashlib.sha256(inference.canonical_json([self.config.seed, sample_id, attempt]).encode()).hexdigest()[:8], 16)
        return {"attempt": attempt, "source": "greedy" if attempt == 0 else "sampled",
                "seed": seed + int(self.bad_seed), "raw_text": text,
                "generated_token_ids": self.code_tokens(text) if complete else self.code_tokens(text)[:-1],
                "eos_reached": complete, "finish_reason": "eos" if complete else "length_limit"}

    def sequence_logp(self, prompt, completion):
        return float("nan") if self.nonfinite else super().sequence_logp(prompt, completion)


class CandidateSupportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="candidate-support-tests-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text('{"max_position_embeddings":16384}\n')
        (self.model / "tokenizer.json").write_text('{"synthetic":true}\n')
        (self.model / "model.safetensors").write_bytes(b"unloadable synthetic model")
        self.source_config = inference.InferenceConfig(model_path=str(self.model), seed=20260929,
            num_generations=4, max_context_tokens=16384, max_new_tokens=2048)
        self.config = replace(self.source_config, fence_policy="unwrap-single")
        self.rows = [row(f"q{index}", f"dev{index % 17}", "print(0)\n", day=20, history_code=f"query-{index}\n") for index in range(70)]
        references = [{"sample_id": item["sample_id"], "reference_kind": "matched",
            "reference_history": [{"timestamp": "2020-01-01-10-00-00", "code": "reference\n", "results": [], "feedback": []}],
            "provenance": {"split": "train", "rule": "past-only", "labels_used": False}} for item in self.rows]
        inputs, refs = self.root / "inputs.jsonl", self.root / "refs.jsonl"
        inputs.write_bytes(b"".join(support.encoded(item) for item in self.rows))
        refs.write_bytes(b"".join(support.encoded(item) for item in references))
        self.source = self.root / "original"
        source_run = self.source / "inference"
        manifest = inference.make_manifest(self.source_config, inputs, refs, inference.model_fingerprint(self.model))
        digest = inference.prepare_output(source_run, manifest, False)
        with redirect_stdout(io.StringIO()):
            inference.run_inference(self.rows, inference.load_references(refs, {item["sample_id"] for item in self.rows}),
                self.source_config, source_run, digest, Backend(self.source_config, original=True))
        completion = {"schema_version": "student-sim-cd.experiment.v1", "status": "success", "exit_code": 0,
            "configuration": asdict(self.source_config),
            "implementation_sha256": {"inference.py": manifest["implementation_sha256"], "scoring.py": manifest["scoring_sha256"]},
            "source_sha256": {"inputs": manifest["inputs_sha256"], "references": manifest["references_sha256"]},
            "stages": [{"name": name, "status": "success", "exit_code": 0} for name in
                ("inference", "predict", "evaluate-base", "evaluate-history_d0", "evaluate-cd", "evaluate-b", "evaluate-copy")]}
        self.seal(self.source, completion)
        self.baseline = self.root / "baseline"
        with redirect_stdout(io.StringIO()):
            cache_rescore.prepare(source_run, inputs, refs, self.baseline / "prepared")
            cache_rescore.score(self.baseline / "prepared", self.config, Backend(self.config))
        release = "a" * 64
        self.seal(self.baseline, {"status": "success", "exit_code": 0, "source_root": "/synthetic/releases/" + release,
            "stages": [{"name": str(index), "status": "success", "exit_code": 0} for index in range(11)]})
        self.specification = {"schema_version": support.PROTOCOL_VERSION, "expected_samples": 70, "expected_students": 17,
            "baseline_release_sha256": release,
            "baseline_sha256": {name: inference.file_hash(self.baseline / name) for name in
                ("manifest.json", "SUCCESS", "prepared/inputs.jsonl", "prepared/references.jsonl",
                 "prepared/inference/manifest.json", "prepared/inference/candidates.jsonl", "prepared/inference/scores.jsonl")},
            "baseline_run": str(self.baseline), "seed": 20260929, "conditions": list(inference.CONDITIONS),
            "balanced_random_per_condition": 3, "control_random_count": 12, "shared_greedy_condition": "11",
            "canonical_protocol": "cached-python-fence-v1", "student_execution": False}
        self.prepared = self.root / "new-support"

    def seal(self, directory, manifest):
        (directory / "manifest.json").write_bytes(support.encoded(manifest))
        (directory / "SUCCESS").write_text(inference.file_hash(directory / "manifest.json") + "\n")

    def prepare(self, **options):
        return support.prepare(self.baseline, self.prepared, protocol=self.specification, config=self.config,
                               tokenizer=options.pop("tokenizer", Tokenizer()), **options)

    def generate(self, backend=None, **options):
        with redirect_stdout(io.StringIO()):
            return support.generate(self.prepared, self.config, backend or Backend(self.config), **options)

    def score(self, backend=None, **options):
        with redirect_stdout(io.StringIO()):
            return support.score(self.prepared, self.config, backend or Backend(self.config), **options)

    def test_prepare_checks_complete_cohort_without_model_or_labels(self):
        with patch.object(inference, "HFBackend", side_effect=AssertionError("Unexpected model allocation")):
            result = self.prepare()
        self.assertEqual((result["samples"], result["students"], result["new_draws"]), (70, 17, 1260))
        bundle = support.load_prepared(self.prepared, self.config, tokenizer=Tokenizer())
        self.assertEqual(len(bundle["rows"]), 70)
        self.assertEqual(len(bundle["references"]), 70)
        self.assertEqual(len(bundle["prompt_tokens"]), 70)
        self.assertFalse((self.prepared / "draw-checkpoints.jsonl").exists())

    def test_protocol_cannot_shrink_cohort_change_sources_or_sample_budget(self):
        for field, value in (("expected_samples", 69), ("expected_students", 16), ("seed", 99),
                             ("conditions", ["11"]), ("balanced_random_per_condition", 4),
                             ("control_random_count", 16), ("shared_greedy_condition", "01"),
                             ("canonical_protocol", "strip-and-repair"), ("student_execution", True)):
            altered = dict(self.specification, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                support.protocol(altered)

    def test_schedule_pairs_global_seeds_and_has_no_greedy_repetition(self):
        schedule = support.draw_schedule(["sample"])
        self.assertEqual(len(schedule), 18)
        self.assertEqual([draw for _, condition, draw, _ in schedule if condition == "11"], list(range(4, 13)))
        for condition, draws in (("01", [4, 5, 6]), ("10", [7, 8, 9]), ("00", [10, 11, 12])):
            self.assertEqual([draw for _, cond, draw, _ in schedule if cond == condition], draws)
        expected = int(hashlib.sha256(b'[20260929,"sample",4]').hexdigest()[:8], 16)
        self.assertEqual(support.draw_seed(20260929, "sample", 4), expected)

    def test_full_budget_raw_provenance_two_pools_and_single_unique_scoring(self):
        original_files = {name: (self.baseline / name).read_bytes() for name in self.specification["baseline_sha256"]}
        self.prepare()
        active = Backend(self.config)
        with redirect_stdout(io.StringIO()):
            result = support.run(self.prepared, self.config, backend=active)
        self.assertEqual(len(active.calls), 1260)
        self.assertEqual(result["baseline_scores_reused"], 318)
        self.assertEqual(result["new_score_records"], 1260)
        self.assertEqual(len(active.scored), 1260 * 4)
        cost = result["cost_accounting"]
        events = inference.read_jsonl(self.prepared / "draw-checkpoints.jsonl")[1::2]
        self.assertEqual(cost["generation_seconds"], math.fsum(event["elapsed_seconds"] for event in events))
        self.assertEqual(cost["new_generated_tokens_including_eos_when_reached"],
                         sum(len(event["raw_generation"]["generated_token_ids"]) for event in events))
        self.assertEqual(cost["new_generation_attempts"], 1260)
        self.assertEqual(cost["new_scoring_branch_count"], 5040)
        self.assertGreater(cost["new_generation_prompt_tokens"], 0)
        self.assertGreater(cost["new_scoring_branch_prompt_tokens"], 0)
        self.assertGreater(cost["new_scoring_branch_completion_tokens"], 0)
        self.assertTrue(math.isfinite(cost["scoring_phase_wall_seconds"]))
        self.assertGreaterEqual(cost["scoring_phase_wall_seconds"], 0)
        for pool in support.POOLS:
            root = self.prepared / "pools" / pool / "inference"
            sources = {"inputs": (self.prepared / "inputs.jsonl").read_bytes(),
                **{name: (root / (name + ".json" + ("l" if name != "manifest" else ""))).read_bytes()
                   for name in ("manifest", "candidates", "scores")}}
            _, _, pools, scores, _, _ = predict._validated_run(sources)
            self.assertEqual((len(pools), len(scores)), (70, 948))
            records = inference.read_jsonl(root / "candidates.jsonl")
            for record in records:
                self.assertEqual(len(record["attempts"]), 13)
                keys = {(raw["condition"], raw["draw_id"]) for raw in record["attempts"]}
                expected = {("11", index) for index in range(13)} if pool == "11_only12" else {
                    ("11", index) for index in range(4)} | {("01", index) for index in range(4, 7)} | {
                    ("10", index) for index in range(7, 10)} | {("00", index) for index in range(10, 13)}
                self.assertEqual(keys, expected)
        self.assertEqual(original_files, {name: (self.baseline / name).read_bytes() for name in original_files})
        self.assertTrue((self.prepared / "SCORE_SUCCESS").exists())
        from student_sim_cd import candidate_support_analysis as analysis
        cohort = {item["sample_id"]: item for item in self.rows}
        for pool in support.POOLS:
            root = self.prepared / "pools" / pool / "inference"
            artifact = analysis.SupportArtifact(**{name: (root / (name + ".json" + ("l" if name != "manifest" else ""))).read_bytes()
                for name in ("manifest", "candidates", "scores")})
            checked = analysis._cache(artifact, (self.prepared / "inputs.jsonl").read_bytes(), cohort, pool,
                                     inference.file_hash(self.prepared / "protocol.json"))
            self.assertIsNotNone(checked)
        resumed = Backend(self.config)
        self.assertEqual(self.score(resumed, resume=True), result)
        self.assertEqual(resumed.scored, [])

    def test_raw_is_durable_before_validation_and_no_automatic_bad_seed_retry(self):
        self.prepare()
        active = Backend(self.config, bad_seed=True)
        with self.assertRaisesRegex(ValueError, "seed/sampling"):
            self.generate(active)
        events = inference.read_jsonl(self.prepared / "draw-checkpoints.jsonl")
        self.assertEqual([event["event"] for event in events], ["reserved", "raw"])
        self.assertIn("```python", events[1]["raw_generation"]["raw_text"])
        before = len(active.calls)
        with self.assertRaises(ValueError):
            self.generate(active, resume=True)
        self.assertEqual(len(active.calls), before)
        self.assertFalse((self.prepared / "GENERATION_SUCCESS").exists())

    def test_reserved_without_raw_blocks_resume_and_preserves_failure(self):
        self.prepare()
        active = Backend(self.config)
        with patch.object(active, "generate", side_effect=RuntimeError("synthetic process death")):
            with self.assertRaises(RuntimeError):
                self.generate(active)
        checkpoint = (self.prepared / "draw-checkpoints.jsonl").read_bytes()
        with self.assertRaisesRegex(ValueError, "Interrupted reserved draw"):
            self.generate(active, resume=True)
        self.assertEqual((self.prepared / "draw-checkpoints.jsonl").read_bytes(), checkpoint)

    def test_clean_completed_draw_prefix_resumes_without_regeneration(self):
        self.prepare()
        first = Backend(self.config)
        original = support._event
        def fail_before_next_reservation(path, value):
            if value.get("event") == "reserved" and len(first.calls) == 2:
                raise RuntimeError("synthetic interruption between completed draws")
            original(path, value)
        with patch.object(support, "_event", side_effect=fail_before_next_reservation):
            with self.assertRaises(RuntimeError):
                self.generate(first)
        resumed = Backend(self.config)
        self.generate(resumed, resume=True)
        self.assertEqual((len(first.calls), len(resumed.calls)), (2, 1258))

    def test_no_eos_attempt_remains_in_raw_budget_and_source_map(self):
        self.prepare()
        self.generate(Backend(self.config, no_eos=True))
        mappings = inference.read_jsonl(self.prepared / "source-mappings.jsonl")
        excluded = [item for item in mappings if not item["eligible"]]
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]["raw_generation"]["raw_text"], "x" * 2048)
        self.assertIn("no EOS", excluded[0]["exclusion_reason"])
        self.assertEqual(len(inference.read_jsonl(self.prepared / "draw-checkpoints.jsonl")), 2520)
        self.assertEqual(len(inference.read_jsonl(self.prepared / "pools/11_only12/candidates.jsonl")), 70)

    def test_model_assets_and_generation_configuration_fail_before_draws(self):
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            support.prepare(self.baseline, self.prepared, protocol=self.specification,
                config=replace(self.config, temperature=0.9), tokenizer=Tokenizer())
        self.prepare()
        (self.model / "model.safetensors").write_bytes(b"changed weights")
        active = Backend(self.config)
        with self.assertRaisesRegex(ValueError, "model weights"):
            self.generate(active)
        self.assertEqual(active.calls, [])

    def test_all_seventy_prompts_preflighted_before_any_draw(self):
        class OverflowTokenizer(Tokenizer):
            def apply_chat_template(self, messages, **options):
                return [1] * 16000
        with self.assertRaisesRegex(ValueError, "context overflow"):
            self.prepare(tokenizer=OverflowTokenizer())
        self.assertFalse(self.prepared.exists())

    def test_preserve_final_assets_and_reject_output_symlinks(self):
        link = self.root / "symlink"
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Noncanonical/symlink"):
            support.prepare(self.baseline, link / "bad", protocol=self.specification, config=self.config, tokenizer=Tokenizer())
        self.prepare()
        with self.assertRaisesRegex(ValueError, "new directory"):
            self.prepare()
        self.generate()
        before = (self.prepared / "generation-manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "Final generation exists"):
            self.generate()
        self.assertEqual((self.prepared / "generation-manifest.json").read_bytes(), before)

    def test_nonfinite_scores_fail_without_success_and_preserve_checkpoint(self):
        self.prepare()
        self.generate()
        with self.assertRaises(ValueError):
            self.score(Backend(self.config, nonfinite=True))
        self.assertFalse((self.prepared / "SCORE_SUCCESS").exists())
        cached = inference.read_jsonl(self.prepared / "combined/scores.jsonl")
        self.assertEqual(len(cached), 318)

    def test_rehashed_final_pool_tamper_rejected_against_raw_draws(self):
        self.prepare()
        self.generate()
        path = self.prepared / "pools/balanced12/candidates.jsonl"
        records = inference.read_jsonl(path)
        records[0]["candidates"][0]["sources"] = ["copy_current", "sampled:01:999"]
        path.write_bytes(b"".join(support.encoded(item) for item in records))
        metadata = json.loads((self.prepared / "generation-manifest.json").read_bytes())
        metadata["files_sha256"]["pools/balanced12/candidates.jsonl"] = inference.file_hash(path)
        (self.prepared / "generation-manifest.json").write_bytes(support.encoded(metadata))
        (self.prepared / "GENERATION_SUCCESS").write_text(inference.file_hash(self.prepared / "generation-manifest.json") + "\n")
        with self.assertRaisesRegex(ValueError, "raw draw assignment"):
            self.score()

    def test_same_checkpoint_nonblocking_lock_prevents_concurrent_stages(self):
        self.prepare()
        with support.lease(self.prepared):
            with self.assertRaisesRegex(ValueError, "Another candidate-support stage"):
                self.generate()
        self.assertFalse((self.prepared / "draw-checkpoints.jsonl").exists())

    def test_score_resume_reuses_durable_prefix_without_duplicate_four_scores(self):
        self.prepare()
        self.generate()
        failed = Backend(self.config)
        normal = failed.sequence_logp
        def terminate_after_two_records(prompt, completion):
            if len(failed.scored) == 8:
                raise RuntimeError("synthetic interrupted scoring")
            return normal(prompt, completion)
        with patch.object(failed, "sequence_logp", side_effect=terminate_after_two_records):
            with self.assertRaises(RuntimeError):
                self.score(failed)
        self.assertEqual(len(inference.read_jsonl(self.prepared / "combined/scores.jsonl")), 320)
        resumed = Backend(self.config)
        report = self.score(resumed, resume=True)
        self.assertEqual(report["new_score_records"], 1260)
        self.assertEqual(len(resumed.scored), (1260 - 2) * 4)

    def test_reused_baseline_score_cannot_be_rehashed_into_different_logp(self):
        self.prepare()
        self.generate()
        with self.assertRaises(ValueError):
            self.score(Backend(self.config, nonfinite=True))
        path = self.prepared / "combined/scores.jsonl"
        rows = inference.read_jsonl(path)
        rows[0]["l11"] -= 1
        rows[0] = support._signed(rows[0], rows[0]["run_sha256"])
        path.write_bytes(b"".join(support.encoded(row) for row in rows))
        with self.assertRaisesRegex(ValueError, "Reused baseline scores changed"):
            self.score(resume=True)

    def test_raw_text_token_disagreement_and_duplicate_eos_are_rejected(self):
        original = Backend(self.config, original=True)
        prompts = original.prompt_tokens(inference.build_messages(self.rows[0], "11", []))
        raw = original.generate(prompts, "q0", 1)
        bad = dict(raw, raw_text="different untrusted text")
        with self.assertRaisesRegex(ValueError, "text differs"):
            support._raw(bad, "q0", 1, self.config, 256, Tokenizer())
        bad = dict(raw, generated_token_ids=raw["generated_token_ids"] + [256])
        with self.assertRaisesRegex(ValueError, "exactly one terminal EOS"):
            support._raw(bad, "q0", 1, self.config, 256)

    def test_import_and_cli_do_not_load_torch_or_allow_label_arguments(self):
        process = subprocess.run([sys.executable, "-B", "-c",
            "import sys; import student_sim_cd.candidate_support; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        process = subprocess.run([sys.executable, "-B", "-m", "student_sim_cd.candidate_support", "run", "--labels", "secret.jsonl"],
                                 capture_output=True, text=True)
        self.assertEqual(process.returncode, 2)

    def test_real_execution_requires_nonsynthetic_sources_and_bounded_process_group(self):
        self.prepare()
        bundle = support.load_prepared(self.prepared, self.config)
        preparation = dict(bundle["manifest"], synthetic_tokenizer_injected=False)
        old = list(bundle["baseline"])
        old[6] = copy.deepcopy(old[6])
        old[6]["cache_rescore"]["backend"] = "HFBackend"
        environment = {"STUDENT_SIM_CONTROLLED_PROCESS_GROUP": "1",
            "CUDA_VISIBLE_DEVICES": "GPU-12345678-1234-1234-1234-123456789abc",
            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
        with patch.dict(os.environ, environment, clear=True), patch.object(os, "sched_getaffinity", return_value={1, 2, 3, 4}, create=True):
            support._real_preconditions(preparation, old)
            with self.assertRaisesRegex(ValueError, "synthetic"):
                support._real_preconditions(bundle["manifest"], old)
            for key, value in (("STUDENT_SIM_CONTROLLED_PROCESS_GROUP", "0"), ("CUDA_VISIBLE_DEVICES", "0"),
                               ("CUDA_VISIBLE_DEVICES", environment["CUDA_VISIBLE_DEVICES"] + ",GPU-other"),
                               ("OMP_NUM_THREADS", "8"), ("MKL_NUM_THREADS", "8"), ("OPENBLAS_NUM_THREADS", "8")):
                with self.subTest(key=key, value=value), patch.dict(os.environ, {key: value}):
                    with self.assertRaisesRegex(ValueError, "bounded outer"):
                        support._real_preconditions(preparation, old)
            with patch.object(os, "sched_getaffinity", return_value=set(range(8)), create=True):
                with self.assertRaisesRegex(ValueError, "bounded outer"):
                    support._real_preconditions(preparation, old)


if __name__ == "__main__":
    unittest.main()
