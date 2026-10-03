"""Select fixed-weight predictions from a complete, verified inference run.

Example: python -m student_sim_cd.predict --run-dir outputs/run --inputs
data/prepared/inputs.test.jsonl --output outputs/predictions --include-copy

Reads only the three named inference artifacts and the explicit input file.
No labels, model loading, candidate execution, parameter fitting or A calibration.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from . import inference, scoring
from .inference import canonical_json, file_hash, object_hash, validate_input
from .scoring import LogProbs, components, method_scores


SCHEMA_VERSION = "student-sim-cd.predictions.v1"
INPUT_SCHEMA = "student-sim-cd.progfeed.v1"
METHOD_PARAMETERS = {
    "base": {"alpha": 0, "beta": 0},
    "history_d0": {"alpha": 1, "beta": 0},
    "cd": {"alpha": 1, "beta": 1},
    "b": {"alpha": 0, "beta": 1},
}
CANDIDATE_RECORD_FIELDS = {
    "sample_id", "candidates", "attempts", "run_sha256", "reference_kind",
    "reference_provenance", "record_sha256",
}
CANDIDATE_FIELDS = {
    "candidate_id", "code", "sources", "transform", "completion_token_ids", "token_count", "eos_included",
}
SCORE_FIELDS = {
    "sample_id", "candidate_id", "run_sha256", "l11", "l01", "l10", "l00",
    "token_count", "eos_included", "prompt_token_counts", "reference_kind",
    "components", "scores_weight_1", "record_sha256",
}


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate JSON field: " + name)
        result[name] = value
    return result


def _json(content):
    value = json.loads(content, object_pairs_hook=_unique_object)
    canonical_json(value)  # Reject NaN/Infinity anywhere, including ignored metadata.
    return value


def _records(content, name):
    rows = []
    # JSON strings may contain U+0085/U+2028/U+2029. Only LF delimits JSONL;
    # str.splitlines() would split those valid characters inside candidate code.
    lines = content.decode("utf-8").split("\n")
    if lines[-1] == "":
        lines.pop()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(f"{name}:{number}: empty JSONL record")
        row = _json(line)
        if not isinstance(row, dict):
            raise ValueError(f"{name}:{number}: record must be an object")
        rows.append(row)
    return rows


def _record(record, allowed, run_hash):
    if set(record) != allowed:
        raise ValueError("unexpected or missing inference record fields")
    if record["run_sha256"] != run_hash:
        raise ValueError("inference record run hash mismatch")
    if record["record_sha256"] != object_hash({k: v for k, v in record.items() if k != "record_sha256"}):
        raise ValueError("inference record content hash mismatch")


def _same_numbers(stored, expected, name):
    if (not isinstance(stored, dict) or set(stored) != set(expected)
            or any(type(stored[key]) not in (int, float) or stored[key] != value
                   for key, value in expected.items())):
        raise ValueError(name + " does not match the four raw log probabilities")


def _validated_run(sources):
    manifest = _json(sources["manifest"])
    if not isinstance(manifest, dict) or manifest.get("schema_version") != inference.SCHEMA_VERSION:
        raise ValueError("unsupported inference manifest schema")
    if manifest.get("prompt_version") != inference.PROMPT_VERSION:
        raise ValueError("unsupported inference prompt version")
    if manifest.get("config_sha256") != object_hash(manifest.get("config")):
        raise ValueError("inference configuration hash mismatch")
    input_hash = hashlib.sha256(sources["inputs"]).hexdigest()
    if manifest.get("inputs_sha256") != input_hash:
        raise ValueError("input file hash differs from the inference manifest")
    run_hash = object_hash(manifest)
    inputs = {}
    for row in _records(sources["inputs"], "inputs"):
        validate_input(row)
        if row.get("schema_version") != INPUT_SCHEMA:
            raise ValueError("unsupported input schema")
        if row["sample_id"] in inputs:
            raise ValueError("duplicate input sample ID")
        inputs[row["sample_id"]] = row
    if not inputs:
        raise ValueError("inputs must contain at least one sample")
    pools, copies, reference_kinds = {}, {}, {}
    for record in _records(sources["candidates"], "candidates"):
        _record(record, CANDIDATE_RECORD_FIELDS, run_hash)
        sample_id = record["sample_id"]
        if not isinstance(sample_id, str) or sample_id not in inputs or sample_id in pools:
            raise ValueError("unknown or duplicate candidate sample ID")
        if not isinstance(record["candidates"], list) or not record["candidates"]:
            raise ValueError("sample has no candidates")
        if (not isinstance(record["reference_kind"], str) or record["reference_kind"] not in
                {"matched", "empty_reference_diagnostic", "shuffled_reference_diagnostic"}):
            raise ValueError("unsupported reference kind")
        pool = {}
        for candidate in record["candidates"]:
            if not isinstance(candidate, dict) or set(candidate) != CANDIDATE_FIELDS:
                raise ValueError("unexpected or missing candidate fields")
            code, identifier = candidate["code"], candidate["candidate_id"]
            if not isinstance(code, str) or identifier != hashlib.sha256(code.encode("utf-8")).hexdigest():
                raise ValueError("candidate ID does not match code content hash")
            if identifier in pool:
                raise ValueError("duplicate candidate ID within sample")
            tokens, count = candidate["completion_token_ids"], candidate["token_count"]
            if (not isinstance(tokens, list) or not tokens or type(count) is not int
                    or count != len(tokens) or candidate["eos_included"] is not True
                    or any(type(token) is not int or token < 0 for token in tokens)):
                raise ValueError("invalid candidate token/EOS protocol")
            origins = candidate["sources"]
            if not isinstance(origins, list) or not origins or any(not isinstance(origin, str) for origin in origins):
                raise ValueError("candidate sources must be nonempty strings")
            if "copy_current" in origins:
                if sample_id in copies or code != inputs[sample_id]["current_code"]:
                    raise ValueError("copy candidate does not uniquely match the current input")
                copies[sample_id] = identifier
            pool[identifier] = candidate
        if sample_id not in copies:
            raise ValueError("candidate pool is missing its copy baseline")
        pools[sample_id] = pool
        reference_kinds[sample_id] = record["reference_kind"]
    if set(pools) != set(inputs):
        raise ValueError("candidate records do not cover every input sample")
    scores = {}
    for record in _records(sources["scores"], "scores"):
        _record(record, SCORE_FIELDS, run_hash)
        sample_id, identifier = record["sample_id"], record["candidate_id"]
        if (not isinstance(sample_id, str) or not isinstance(identifier, str)
                or sample_id not in pools or identifier not in pools[sample_id]):
            raise ValueError("score references an unknown sample or candidate")
        key = (sample_id, identifier)
        if key in scores:
            raise ValueError("duplicate score sample/candidate ID")
        candidate = pools[sample_id][identifier]
        if (type(record["token_count"]) is not int or record["token_count"] != candidate["token_count"]
                or record["eos_included"] is not True or record["reference_kind"] != reference_kinds[sample_id]):
            raise ValueError("score/candidate protocol mismatch")
        raw = LogProbs(**{name: record[name] for name in ("l11", "l01", "l10", "l00")})
        derived = method_scores(raw, weight=1.0)
        _same_numbers(record["components"], components(raw), "stored components")
        _same_numbers(record["scores_weight_1"], derived, "stored method scores")
        scores[key] = derived
    expected = {(sample_id, identifier) for sample_id, pool in pools.items() for identifier in pool}
    if set(scores) != expected:
        raise ValueError("missing scores; every method requires the same complete candidate pool")
    return manifest, run_hash, pools, scores, copies, reference_kinds


def predict(run_dir, inputs, output, include_copy=False):
    """Write one evaluator-compatible JSONL per method into a new directory."""
    run_dir, output = Path(run_dir), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("prediction output already exists; no overwrite or resume")
    if output.resolve() != output:
        raise ValueError("prediction output must not contain symlinks or noncanonical parents")
    paths = {"inputs": Path(inputs), "manifest": run_dir / "manifest.json",
             "candidates": run_dir / "candidates.jsonl", "scores": run_dir / "scores.jsonl"}
    # Hash exactly the bytes parsed, even if another process is still appending.
    sources = {name: path.read_bytes() for name, path in paths.items()}
    manifest, run_hash, pools, scores, copies, reference_kinds = _validated_run(sources)
    methods = list(METHOD_PARAMETERS) + (["copy"] if include_copy else [])
    predictions, selections = {method: [] for method in methods}, []
    for sample_id, pool in sorted(pools.items()):
        choices = {}
        for method in methods:
            identifier = (copies[sample_id] if method == "copy" else min(
                pool, key=lambda candidate_id: (-scores[sample_id, candidate_id][method], candidate_id)))
            predictions[method].append({"sample_id": sample_id, "method": method,
                                        "predicted_code": pool[identifier]["code"]})
            choices[method] = {"candidate_id": identifier,
                               "score": None if method == "copy" else scores[sample_id, identifier][method]}
        selections.append({"sample_id": sample_id, "candidate_count": len(pool),
                           "reference_kind": reference_kinds[sample_id], "choices": choices})
    report = {
        "schema_version": SCHEMA_VERSION, "input_schema_version": INPUT_SCHEMA,
        "inference_run_sha256": run_hash, "inference_manifest": manifest,
        "source_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()},
        "source_paths": {name: str(path.resolve()) for name, path in paths.items()},
        "implementation_sha256": {"predict": file_hash(Path(__file__)),
                                   "scoring": file_hash(Path(scoring.__file__)),
                                   "input_validation": file_hash(Path(inference.__file__))},
        "method_parameters": {method: dict(parameters) for method, parameters in METHOD_PARAMETERS.items()},
        "contrast_weight": 1.0,
        "include_copy": include_copy, "selection": "argmax; ties use lexicographically smallest candidate_id",
        "common_candidate_pool": True, "samples": len(pools), "candidates": len(scores),
        "copy_only_samples": sum(len(pool) == 1 for pool in pools.values()),
        "reference_kinds": dict(sorted(Counter(reference_kinds.values()).items())),
        "labels_read": False, "parameters_fitted": False, "candidate_execution": False,
        "limitations": ["Fixed-weight shared-candidate diagnostic; not online token CD.",
                        "No A calibration or execution-based candidate groups are applied.",
                        "Argmax predictions do not guarantee group marginals.",
                        "Copy-only pools and empty references must be reported as diagnostics."],
    }
    output.mkdir(parents=True)
    marker = output / ".incomplete"
    with marker.open("x", encoding="utf-8") as stream:
        stream.write("Prediction export is incomplete until manifest.json is written.\n")
    artifacts = {"predictions." + method + ".jsonl": rows for method, rows in predictions.items()}
    artifacts["selections.jsonl"] = selections
    report["output_sha256"] = {}
    for name, rows in artifacts.items():
        with (output / name).open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(canonical_json(row) + "\n")
        report["output_sha256"][name] = file_hash(output / name)
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(report) + "\n")
    marker.unlink()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="exact inputs file recorded by inference")
    parser.add_argument("--output", type=Path, required=True, help="new prediction directory")
    parser.add_argument("--include-copy", action="store_true")
    options = parser.parse_args(argv)
    try:
        report = predict(options.run_dir, options.inputs, options.output, options.include_copy)
        print(canonical_json({"schema_version": SCHEMA_VERSION, "samples": report["samples"],
                              "candidates": report["candidates"], "outputs": report["output_sha256"]}))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
