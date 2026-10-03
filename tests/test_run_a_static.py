"""Meaningful input/output boundary tests for the static A CPU runner."""

from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_a_static", ROOT / "scripts/run_a_static.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def row(student="s1", number=1, *, feedback=True, history=False):
    result = {"schema_version": "student-sim-cd.progfeed.v1", "sample_id": student + "-" + str(number),
        "student_id": student, "lab": "lab02", "source_file": "x.py",
        "current_timestamp": "2020-01-02-00-00-%02d" % number,
        "current_code": "raise RuntimeError('this code must never execute')\n",
        "current_results": [], "history": [], "feedback": [], "problem_statement": "Task"}
    if feedback:
        result["feedback"] = [{"test_name": "t", "function_name": "f", "assigned_type": "nl", "text": "Please revise"}]
    if history:
        result["history"] = [{"timestamp": "2020-01-01-00-00-01", "code": "prior\n", "results": [], "feedback": []}]
    return result


def labels(rows):
    return [{"schema_version": "student-sim-cd.progfeed.v1", "sample_id": r["sample_id"],
        "target_timestamp": "2020-01-03-00-00-01", "target_code": "future\n", "target_results": []} for r in rows]


def jsonl(rows):
    return b"".join((json.dumps(r) + "\n").encode() for r in rows)


def directory(root, r):
    path = root / "all_labs" / r["lab"] / r["student_id"] / r["current_timestamp"]
    path.mkdir(parents=True)
    (path / r["source_file"]).write_text(r["current_code"])
    return path


class AStaticRunnerTests(unittest.TestCase):
    def test_current_only_complete_cohort_includes_empty_history_and_ignores_future_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            rows = [row("a", history=True), row("b"), row("c", feedback=False)]
            for r in rows:
                directory(root, r)
            # A future multi-file submission must not exclude this current input.
            future = root / "all_labs/lab02/a/2020-01-03-00-00-01"
            future.mkdir()
            (future / "x.py").write_text("future\n")
            (future / "another.py").write_text("future companion\n")
            selected, history, report = runner.build_train_cohort(rows, {("lab02", "x.py")}, root,
                {"train_samples": 3, "train_students": 3, "eligible_samples": 2, "eligible_students": 2, "history_samples": 1})
            self.assertEqual({r["student_id"] for r in selected}, {"a", "b"})
            self.assertEqual(len(history), 1)
            self.assertFalse(report["labels_used_for_selection"])
            self.assertFalse(report["future_directories_inspected"])
            self.assertEqual(report["empty_history_samples"], 1)

    def test_current_multifile_is_excluded_with_explicit_reason_and_count_mismatch_stops(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            rows = [row("a"), row("b")]
            directory(root, rows[0])
            current = directory(root, rows[1])
            (current / "other.py").write_text("extra\n")
            expected = {"train_samples": 2, "train_students": 2, "eligible_samples": 1, "eligible_students": 1, "history_samples": 0}
            selected, _, report = runner.build_train_cohort(rows, {("lab02", "x.py")}, root, expected)
            self.assertEqual(len(selected), 1)
            self.assertEqual(report["exclusions_by_primary_reason"], {"current_submission_not_exactly_one_declared_python_file": 1})
            with self.assertRaisesRegex(ValueError, "counts differ"):
                runner.build_train_cohort(rows, {("lab02", "x.py")}, root, {**expected, "eligible_samples": 2})

    def test_missing_raw_source_never_falls_back_to_historical_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                runner.build_train_cohort([row()], {("lab02", "x.py")}, Path(temporary).resolve() / "missing",
                    {"train_samples": 1, "train_students": 1, "eligible_samples": 1, "eligible_students": 1, "history_samples": 0})

    def test_current_symlink_is_rejected_without_reading_or_executing_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            r = row()
            current = directory(root, r)
            (current / "bad.py").symlink_to(root / "missing-outside.py")
            with self.assertRaisesRegex(ValueError, "symlink"):
                runner.current_file_listing(r, root)

    def test_input_validation_rejects_future_fields_and_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "forbidden or unknown"):
            runner.input_rows(jsonl([{**row(), "target_code": "not an input"}]))
        with self.assertRaisesRegex(ValueError, "Duplicate input"):
            runner.input_rows(jsonl([row(), row()]))

    def test_labels_require_complete_ids_and_strictly_later_time(self):
        rows = [row("a"), row("b")]
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            runner.label_rows(jsonl(labels(rows[:1])), rows)
        values = labels(rows)
        values[0]["target_timestamp"] = rows[0]["current_timestamp"]
        with self.assertRaisesRegex(ValueError, "not after"):
            runner.label_rows(jsonl(values), rows)

    def test_scope_rejects_wildcards_duplicates_and_implicit_names(self):
        for value in [{"pairs": [{"lab": "*", "source_file": "x.py"}]},
                      {"pairs": [{"lab": "lab02", "source_file": "x.py"}] * 2},
                      {"labs": ["lab02"]}]:
            with self.assertRaises(ValueError):
                runner.scope_pairs(json.dumps(value).encode())

    def test_byte_source_hash_mismatch_and_post_read_mutation_are_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "input.jsonl"
            path.write_bytes(b"original\n")
            audit = runner.ReadAudit()
            with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                audit.read(path, "training_inputs", "0" * 64)
            audit.read(path, "training_inputs", runner.sha(b"original\n"))
            path.write_bytes(b"changed\n")
            with self.assertRaisesRegex(ValueError, "source changed"):
                audit.verify_unchanged()

    def test_existing_empty_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / "existing"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "new and canonical"):
                runner.new_output(output)
            self.assertEqual(list(output.iterdir()), [])

    def test_public_artifacts_reject_recursive_private_text_case_insensitively(self):
        for value in [{"nested": [{"TaRgEt_CoDe": "private"}]}, {"code": "private"}, {"history": []}]:
            with self.assertRaisesRegex(ValueError, "Private content"):
                runner.code_free(value)

    def test_train_predict_parser_does_not_accept_dev_or_test_labels(self):
        with patch("sys.stderr"):
            for forbidden in ("--dev-labels", "--test-labels"):
                with self.assertRaises(SystemExit):
                    runner.parser().parse_args(["train-predict", forbidden, "/tmp/labels"])

    def test_expired_or_unbounded_deadlines_fail_before_output(self):
        past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        future = (datetime.now(timezone.utc) + timedelta(minutes=2)).isoformat()
        with self.assertRaisesRegex(ValueError, "unexpired"):
            runner.checked_deadline(past, 60)
        for cap in (0, 7201, True):
            with self.assertRaisesRegex(ValueError, "wall cap"):
                runner.checked_deadline(future, cap)
        with self.assertRaises(ValueError):
            runner.checked_deadline("2020-01-01T01:00:00", 60)

    def test_frozen_inventory_uses_original_serializer_and_reports_weaker_scope(self):
        r = row("a", history=True)
        prefix = "all_labs/lab02/a/" + r["current_timestamp"] + "/"
        inventory = [{"path": prefix + "x.py", "sha256": "1" * 64, "bytes": 12},
                     {"path": "all_labs/lab02/a/2020-01-03-00-00-01/other.py", "sha256": "2" * 64, "bytes": 5}]
        fingerprint = runner.sha(runner.progfeed._json(inventory).encode())
        value = {"schema_version": "student-sim-cd.progfeed.v1", "source_files": inventory,
                 "source_fingerprint_sha256": fingerprint, "counts": {"missing_or_unreadable_source_files": 0}}
        listings, audit = runner.source_inventory(json.dumps(value).encode(), fingerprint)
        self.assertNotEqual(fingerprint, runner.sha(runner.inference.canonical_json(inventory).encode()))
        selected, _, funnel = runner.build_train_cohort([r], {("lab02", "x.py")}, None,
            {"train_samples": 1, "train_students": 1, "eligible_samples": 1, "eligible_students": 1, "history_samples": 1}, inventory=listings)
        self.assertEqual(selected, [r])
        self.assertFalse(audit["physical_raw_directory_checked"])
        self.assertFalse(funnel["physical_raw_directory_checked"])
        self.assertIn("unrecorded nested companion", audit["limitation"])
        with self.assertRaisesRegex(ValueError, "fingerprint differs"):
            runner.source_inventory(json.dumps(value).encode(), "0" * 64)
        value["counts"]["missing_or_unreadable_source_files"] = 1
        with self.assertRaisesRegex(ValueError, "unreadable or missing"):
            runner.source_inventory(json.dumps(value).encode(), fingerprint)

    def test_inventory_companion_current_file_excludes_row_without_target_filter(self):
        r = row("a")
        prefix = "all_labs/lab02/a/" + r["current_timestamp"]
        reason, record = runner.inventory_current_listing(r, {prefix: ["other.py", "x.py"]})
        self.assertEqual(reason, "current_inventory_not_exactly_one_declared_python_file")
        self.assertEqual(record["files"], ["other.py", "x.py"])
        with self.assertRaisesRegex(ValueError, "One explicit"):
            runner.build_train_cohort([r], {("lab02", "x.py")}, "/tmp",
                {"train_samples": 1, "train_students": 1}, inventory={})

    def test_blank_jsonl_is_rejected_instead_of_silently_changing_cohort(self):
        with self.assertRaisesRegex(ValueError, "Empty JSONL"):
            runner.input_rows(jsonl([row()]) + b"\n")
        with self.assertRaisesRegex(ValueError, "objects"):
            runner.jsonl_rows(b"[]\n")

    def test_prelabel_distributions_cover_all_rankers_and_fail_on_missing_q_group(self):
        r = row()
        current, revised = r["current_code"], "changed\n"
        ids = [runner.sha(current.encode()), runner.sha(revised.encode())]
        pools = {r["sample_id"]: {ids[0]: {"code": current}, ids[1]: {"code": revised}}}
        scores = {(r["sample_id"], cid): {method: index for method in ("base", "cd", "b", "history_d0")}
                  for index, cid in enumerate(ids)}
        q = [{"sample_id": r["sample_id"], "student_id": r["student_id"], "p_changed": .7}]
        sealed = runner.seal_distributions([r], pools, scores, q)
        self.assertEqual(len(sealed), 8)
        runner.code_free(sealed)
        for record in sealed:
            if record["method"].startswith("a_"):
                changed = sum(p for group, p in zip(record["groups"], record["probabilities"]) if group == "changed")
                self.assertAlmostEqual(changed, .7)
        copy_only = {r["sample_id"]: {ids[0]: {"code": current}}}
        with self.assertRaisesRegex(ValueError, "no candidate"):
            runner.seal_distributions([r], copy_only, {key: val for key, val in scores.items() if key[1] == ids[0]}, q)

    def test_analysis_is_bound_to_sealed_probabilities_before_labels(self):
        from student_sim_cd.a_static_analysis import analyze_pool
        r = row()
        bodies = [r["current_code"], "changed\n"]
        pools = {r["sample_id"]: {runner.sha(body.encode()): {"code": body} for body in bodies}}
        scores = {(r["sample_id"], cid): {method: 0. for method in ("base", "cd", "b", "history_d0")}
                  for cid in pools[r["sample_id"]]}
        q = [{"sample_id": r["sample_id"], "student_id": r["student_id"], "p_changed": .4}]
        sealed = runner.seal_distributions([r], pools, scores, q)
        report = analyze_pool([r], labels([r]), pools, scores, q, "fixture", bootstrap=20)
        runner.verify_distribution_seal(report, sealed)
        report["distributions"][0]["probabilities"] = [.9, .1]
        with self.assertRaisesRegex(ValueError, "pre-label distribution"):
            runner.verify_distribution_seal(report, sealed)

    def test_real_protocol_requires_all_frozen_sources_and_no_generation(self):
        path = ROOT / "configs/a_static_v1.json"
        protocol, digest = runner.read_protocol(path, runner.ReadAudit())
        self.assertEqual(protocol["eligibility_mode"], "frozen_source_inventory_current_submission_only")
        self.assertEqual(len(digest), 64)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary).resolve() / "protocol.json"
            altered = {**protocol, "new_generation": True}
            destination.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "prohibits"):
                runner.read_protocol(destination, runner.ReadAudit())

    def pipeline_fixture(self, root):
        from student_sim_cd import change_q
        train = [row("train-a", history=True), row("train-b", history=True)]
        dev = [row("dev-a"), row("dev-b")]
        (root / "inputs.dev.jsonl").write_bytes(jsonl(dev))
        (root / "labels.dev.jsonl").write_bytes(jsonl(labels(dev)))
        protocol = json.loads((ROOT / "configs/a_static_v1.json").read_text())
        protocol.update(expected_dev_samples=2, expected_dev_students=2,
            dev_inputs_sha256=runner.sha(jsonl(dev)), dev_labels_sha256=runner.sha(jsonl(labels(dev))),
            stop_by=(datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat())
        protocol["bootstrap"] = {**protocol["bootstrap"], "repetitions": 20}
        args = SimpleNamespace(command="run", deadline=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
            max_seconds=60, protocol=ROOT / "configs/a_static_v1.json", output=root / "new-result",
            train_inputs=root / "never-opened.train-inputs", train_labels=root / "never-opened.train-labels",
            dev_inputs=root / "inputs.dev.jsonl", dev_labels=root / "labels.dev.jsonl",
            source_audit=root / "never-opened.source-audit", source_root=None, scope=root / "scope", splits=root / "splits")
        pools = {r["sample_id"]: {runner.sha(body.encode()): {"code": body}
                 for body in (r["current_code"], "revision\n")} for r in dev}
        scores = {(sid, cid): {method: 0. for method in ("base", "cd", "b", "history_d0")}
                  for sid, pool in pools.items() for cid in pool}
        return args, protocol, train, dev, pools, scores, change_q

    def test_full_pipeline_seals_all_choices_before_first_dev_label_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            args, protocol, train, dev, pools, scores, change_q = self.pipeline_fixture(root)
            original_read = runner.ReadAudit.read
            observed = []

            def read(audit, path, role, expected=None):
                if role == "dev_labels":
                    names = ["q-fit-seal.json", "q-prediction-seal.json", "candidate-distribution-seal.json"]
                    names += ["distributions." + name + ".pre-label.jsonl" for name in
                              ("balanced12", "11_only12", "legacy318", "balanced12-history-sensitivity")]
                    self.assertTrue(all((args.output / name).is_file() for name in names))
                    observed.append(role)
                return original_read(audit, path, role, expected)

            def fit(inputs, labels, *positional, **keywords):
                self.assertEqual(keywords["split"], "train")
                self.assertEqual(keywords["folds"], 5)
                return {"samples": len(inputs)}, {"fixed_method": "feedback_numeric"}, []

            def prediction(models, inputs, **keywords):
                return [{"sample_id": r["sample_id"], "student_id": r["student_id"], "p_changed": .6} for r in inputs]

            with patch.object(runner, "read_protocol", return_value=(protocol, "a" * 64)), \
                 patch.object(runner, "prepare_training", return_value=(train, labels(train), train, labels(train),
                     {"eligible_samples": 2}, {r["student_id"]: "dev" for r in dev})), \
                 patch.object(change_q, "run_train_experiment", side_effect=fit), \
                 patch.object(change_q, "predict", side_effect=prediction), \
                 patch.object(runner, "validated_pools", return_value=(
                     {name: (pools, scores) for name in ("balanced12", "11_only12", "legacy318")}, {})), \
                 patch.object(runner, "independent_verify", return_value={"status": "verified", "scope": "synthetic fixture only"}), \
                 patch.object(runner.ReadAudit, "read", read):
                manifest = runner.execute(args, enforce_server=False)
            self.assertEqual(observed, ["dev_labels"])
            self.assertTrue(manifest["dev_labels_read"])
            self.assertFalse(manifest["full_progress_A_completed"])
            self.assertTrue(manifest["independent_verification_completed"])
            self.assertTrue((args.output / "calculation-manifest.json").exists())
            self.assertTrue((args.output / "independent-verification.json").exists())
            self.assertEqual((args.output / "SUCCESS").read_text().strip(), runner.sha((args.output / "manifest.json").read_bytes()))
            self.assertFalse((args.output / "FAILURE.json").exists())
            for path in args.output.iterdir():
                if path.suffix in (".json", ".jsonl"):
                    self.assertNotIn("this code must never execute", path.read_text())
                    self.assertNotIn('"target_code"', path.read_text())

    def test_pipeline_missing_positive_q_group_stops_before_dev_labels_and_keeps_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            args, protocol, train, dev, pools, scores, change_q = self.pipeline_fixture(root)
            pools = {r["sample_id"]: {runner.sha(r["current_code"].encode()): {"code": r["current_code"]}} for r in dev}
            scores = {key: value for key, value in scores.items() if key[1] in pools[key[0]]}
            with patch.object(runner, "read_protocol", return_value=(protocol, "a" * 64)), \
                 patch.object(runner, "prepare_training", return_value=(train, labels(train), train, labels(train),
                     {}, {r["student_id"]: "dev" for r in dev})), \
                 patch.object(change_q, "run_train_experiment", return_value=({}, {}, [])), \
                 patch.object(change_q, "predict", return_value=[{"sample_id": r["sample_id"], "student_id": r["student_id"], "p_changed": .6} for r in dev]), \
                 patch.object(runner, "validated_pools", return_value=(
                     {name: (pools, scores) for name in ("balanced12", "11_only12", "legacy318")}, {})):
                with self.assertRaisesRegex(ValueError, "no candidate"):
                    runner.execute(args, enforce_server=False)
            failure = json.loads((args.output / "FAILURE.json").read_text())
            self.assertFalse(failure["dev_labels_read"])
            self.assertFalse(any(item["role"] == "dev_labels" for item in failure["actual_read_files"]))
            self.assertFalse((args.output / "SUCCESS").exists())
            self.assertTrue((args.output / "q-prediction-seal.json").exists())

    def test_independent_verification_failure_never_writes_final_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            args, protocol, train, dev, pools, scores, change_q = self.pipeline_fixture(root)
            with patch.object(runner, "read_protocol", return_value=(protocol, "a" * 64)), \
                 patch.object(runner, "prepare_training", return_value=(train, labels(train), train, labels(train),
                     {}, {r["student_id"]: "dev" for r in dev})), \
                 patch.object(change_q, "run_train_experiment", return_value=({}, {}, [])), \
                 patch.object(change_q, "predict", return_value=[{"sample_id": r["sample_id"], "student_id": r["student_id"], "p_changed": .6} for r in dev]), \
                 patch.object(runner, "validated_pools", return_value=(
                     {name: (pools, scores) for name in ("balanced12", "11_only12", "legacy318")}, {})), \
                 patch.object(runner, "independent_verify", return_value={"status": "failed"}):
                with self.assertRaisesRegex(ValueError, "verification did not pass"):
                    runner.execute(args, enforce_server=False)
            failure = json.loads((args.output / "FAILURE.json").read_text())
            self.assertTrue(failure["dev_labels_read"])
            self.assertTrue((args.output / "calculation-manifest.json").exists())
            self.assertTrue((args.output / "analysis.balanced12.json").exists())
            self.assertFalse((args.output / "manifest.json").exists())
            self.assertFalse((args.output / "SUCCESS").exists())

    def test_real_q_fit_and_real_analysis_pipeline_on_synthetic_students(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            args, protocol, _, dev, pools, scores, _ = self.pipeline_fixture(root)
            train = [row("train-" + str(student), number, history=True) for student in range(10) for number in (1, 2)]
            train_labels = labels(train)
            for current, outcome in zip(train, train_labels):
                if current["sample_id"].endswith("-2"):
                    outcome["target_code"] = current["current_code"]
            registry = {r["student_id"]: "train" for r in train} | {r["student_id"]: "dev" for r in dev}
            with patch.object(runner, "read_protocol", return_value=(protocol, "a" * 64)), \
                 patch.object(runner, "prepare_training", return_value=(train, train_labels, train, train_labels,
                     {"eligible_samples": 20, "eligible_students": 10}, registry)), \
                 patch.object(runner, "validated_pools", return_value=(
                     {name: (pools, scores) for name in ("balanced12", "11_only12", "legacy318")}, {})), \
                 patch.object(runner, "independent_verify", return_value={"status": "verified", "scope": "synthetic fixture only"}):
                manifest = runner.execute(args, enforce_server=False)
            cv = json.loads((args.output / "train-cv.json").read_text())
            self.assertEqual(cv["sample_count"], 20)
            self.assertEqual(cv["student_count"], 10)
            self.assertEqual(cv["folds"], 5)
            oof = runner.jsonl_rows((args.output / "train-oof.jsonl").read_bytes())
            self.assertEqual(len(oof), 20)
            predictions = runner.jsonl_rows((args.output / "q.dev.jsonl").read_bytes())
            self.assertEqual(len(predictions), 2)
            self.assertTrue(all(0 < r["p_changed"] < 1 for r in predictions))
            self.assertTrue(manifest["independent_verification_completed"])
            self.assertFalse(manifest["full_progress_A_completed"])


if __name__ == "__main__":
    unittest.main()
