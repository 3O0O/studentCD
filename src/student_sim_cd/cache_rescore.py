"""Extract and rescore an immutable, complete candidate cache without generation.

Preparation is standard-library only. Scoring reuses inference.HFBackend and
run_inference, including their tokenizer, EOS, context and cache checks. No
student code is parsed or executed, and no label path is accepted.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

from . import inference, predict


SCHEMA_VERSION = "student-sim-cd.cache-rescore.v1"
EXTRACTION_PROTOCOL = "whole-response-single-python-or-py-fence-v1"
PREPARED_NAME = "prepared.jsonl"
PREPARED_FIELDS = {
    "sample_id", "candidates", "attempts", "mappings", "reference_kind",
    "reference_provenance", "run_sha256", "record_sha256",
}
ATTEMPT_FIELDS = {
    "attempt", "source", "seed", "raw_text", "generated_token_ids",
    "eos_reached", "finish_reason", "candidate_id", "transform", "exclusion_reason",
}


def extract_code(raw_text: str) -> tuple[str, str, str]:
    """Unwrap only one complete Python/py response; preserve all other bytes.

    This deliberately retains empty strings, invalid syntax, whitespace,
    unlabelled fences, extra prose and multiple fences. It is not code repair.
    """
    if not isinstance(raw_text, str):
        raise ValueError("raw_text must be a string")
    if raw_text.startswith(("```python\n", "```py\n", "```python\r\n", "```py\r\n")):
        code, transform = inference.transform_code(raw_text, "unwrap-single")
        if transform != "none":
            return code, transform, "single_python_fence_extracted"
        return raw_text, "none", "python_fence_not_single_complete_response"
    if raw_text.startswith("```"):
        return raw_text, "none", "unsupported_or_unlabelled_fence_preserved"
    if "```" in raw_text:
        return raw_text, "none", "non_whole_or_multiple_fence_preserved"
    return raw_text, "none", "original_text_preserved"


def _identifier(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _canonical_destination(output: Path, source: Path | None = None) -> Path:
    output = Path(output).absolute()
    if output.is_symlink() or output.resolve() != output:
        raise ValueError("output must not contain symlinks or noncanonical parents")
    if source is not None and (output == source or source in output.parents):
        raise ValueError("rescore output must be outside the original completed run")
    return output


def _completion(sources: dict[str, bytes]) -> dict:
    manifest = predict._json(sources["completion_manifest"])
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != "student-sim-cd.experiment.v1"
            or manifest.get("status") != "success"
            or type(manifest.get("exit_code")) is not int or manifest["exit_code"] != 0):
        raise ValueError("source experiment is not a successful completed run")
    expected_stages = {"inference", "predict", "evaluate-base", "evaluate-history_d0",
                       "evaluate-cd", "evaluate-b", "evaluate-copy"}
    stages = manifest.get("stages")
    if (not isinstance(stages, list) or len(stages) != len(expected_stages)
            or any(not isinstance(stage, dict) or stage.get("status") != "success"
                   or type(stage.get("exit_code")) is not int or stage["exit_code"] != 0
                   for stage in stages)
            or {stage.get("name") for stage in stages} != expected_stages):
        raise ValueError("source experiment must have all seven successful stages")
    digest = hashlib.sha256(sources["completion_manifest"]).hexdigest()
    if sources["success"].decode("ascii").strip() != digest:
        raise ValueError("source SUCCESS does not match the completed manifest bytes")
    return manifest


def _prepared_sample(row: dict, original: dict, old_pool: dict, config: dict) -> dict:
    attempts = original["attempts"]
    if not isinstance(attempts, list) or len(attempts) != config["num_generations"]:
        raise ValueError("source attempts must cover the full original generation budget")
    pool: dict[str, dict] = {}
    mappings = []
    original_sources: dict[str, list[str]] = {}

    def add(code, source, transform):
        identifier = _identifier(code)
        if identifier not in pool:
            pool[identifier] = {"candidate_id": identifier, "code": code,
                                "sources": [], "transform": transform}
        pool[identifier]["sources"].append(source)
        return identifier

    copy_id = _identifier(row["current_code"])
    original_sources[copy_id] = ["copy_current"]
    mappings.append({"source": "copy_current", "attempt": None,
                     "original_candidate_id": copy_id, "candidate_id": add(row["current_code"], "copy_current", "none"),
                     "transform": "none", "extraction_reason": "copy_current_preserved", "eligible": True})
    for index, attempt in enumerate(attempts):
        if (not isinstance(attempt, dict) or set(attempt) - ATTEMPT_FIELDS
                or type(attempt.get("attempt")) is not int or attempt["attempt"] != index
                or attempt.get("source") != ("greedy" if index == 0 else "sampled")
                or type(attempt.get("seed")) is not int or not isinstance(attempt.get("raw_text"), str)
                or type(attempt.get("eos_reached")) is not bool):
            raise ValueError("invalid or unordered source generation attempt")
        tokens = attempt.get("generated_token_ids")
        if (not isinstance(tokens, list) or (attempt["eos_reached"] and not tokens)
                or any(type(token) is not int or token < 0 for token in tokens)
                or attempt.get("finish_reason") != ("eos" if attempt["eos_reached"] else "length_limit")):
            raise ValueError("invalid source generation token/EOS metadata")
        old_id = attempt.get("candidate_id")
        origin = f"{attempt['source']}:{index}"
        if old_id is not None:
            old_code, old_transform = inference.transform_code(attempt["raw_text"], config["fence_policy"])
            if (not attempt["eos_reached"] or old_id != _identifier(old_code) or old_id not in old_pool
                    or attempt.get("transform") != old_transform):
                raise ValueError("source raw attempt does not match its original candidate")
            original_sources.setdefault(old_id, []).append(origin)
        elif not isinstance(attempt.get("exclusion_reason"), str) or not attempt["exclusion_reason"]:
            raise ValueError("source excluded attempt is missing its explicit reason")
        code, transform, reason = extract_code(attempt["raw_text"])
        eligible = attempt["eos_reached"]
        mappings.append({"source": origin, "attempt": index, "original_candidate_id": old_id,
                         "candidate_id": add(code, origin, transform) if eligible else None,
                         "transform": transform, "extraction_reason": reason, "eligible": eligible,
                         "original_exclusion_reason": attempt.get("exclusion_reason")})
    if (set(original_sources) != set(old_pool)
            or any(old_pool[key]["sources"] != origins for key, origins in original_sources.items())):
        raise ValueError("source pool/source mapping does not match every original attempt")
    return {"sample_id": row["sample_id"], "candidates": list(pool.values()),
            "attempts": attempts, "mappings": mappings,
            "reference_kind": original["reference_kind"],
            "reference_provenance": original["reference_provenance"]}


def _write_exact(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"prepared artifact content mismatch: {path.name}; no overwrite")
    else:
        with path.open("xb") as handle:
            handle.write(content)


def prepare(source_run: Path, inputs: Path, references: Path | None, output: Path,
            resume: bool = False) -> dict:
    """Verify all original caches, then write a new model-free preparation.

    source_run is the completed experiment's inference/ directory. Its parent
    must contain the final seven-stage manifest.json and matching SUCCESS.
    Only named inputs/references/cache files are read, never future labels.
    """
    source_run = Path(source_run).resolve()
    output = _canonical_destination(output, source_run.parent)
    paths = {"manifest": source_run / "manifest.json", "candidates": source_run / "candidates.jsonl",
             "scores": source_run / "scores.jsonl", "inputs": Path(inputs),
             "completion_manifest": source_run.parent / "manifest.json", "success": source_run.parent / "SUCCESS"}
    if references is not None:
        paths["references"] = Path(references)
    sources = {name: path.read_bytes() for name, path in paths.items()}
    completed = _completion(sources)
    source_manifest, source_hash, pools, _, _, _ = predict._validated_run(sources)
    reference_hash = hashlib.sha256(sources["references"]).hexdigest() if references is not None else None
    if source_manifest.get("references_sha256") != reference_hash:
        raise ValueError("reference bytes do not match the source inference manifest")
    if (completed.get("source_sha256") != {"inputs": source_manifest["inputs_sha256"], "references": reference_hash}
            or any(completed.get("configuration", {}).get(key) != value
                   for key, value in source_manifest["config"].items())):
        raise ValueError("completed experiment configuration/input hashes differ from source inference")
    implementations = completed.get("implementation_sha256", {})
    if (implementations.get("inference.py") != source_manifest["implementation_sha256"]
            or implementations.get("scoring.py") != source_manifest["scoring_sha256"]):
        raise ValueError("completed experiment code hashes differ from source inference manifest")
    rows = predict._records(sources["inputs"], "inputs")  # Already validated from exactly these bytes.
    inference.load_references(Path(references) if references is not None else None, set(pools))
    if references is not None and inference.file_hash(Path(references)) != reference_hash:
        raise ValueError("reference file changed while preparation was reading it")
    records = {record["sample_id"]: record for record in predict._records(sources["candidates"], "candidates")}
    prepared = [_prepared_sample(row, records[row["sample_id"]], pools[row["sample_id"]],
                                 source_manifest["config"]) for row in rows]
    manifest = {
        "schema_version": SCHEMA_VERSION, "extraction_protocol": EXTRACTION_PROTOCOL,
        "source_manifest": source_manifest, "source_run_sha256": source_hash,
        "source_completion_manifest": completed,
        "source_paths": {key: str(path.resolve()) for key, path in paths.items()},
        "source_sha256": {key: hashlib.sha256(value).hexdigest() for key, value in sources.items()},
        "implementation_sha256": {"cache_rescore": inference.file_hash(Path(__file__)),
                                   "inference": inference.file_hash(Path(inference.__file__)),
                                   "source_validation": inference.file_hash(Path(predict.__file__))},
        "records_payload_sha256": inference.object_hash(prepared), "references_present": references is not None,
        "samples": len(prepared), "original_candidates": sum(len(pool) for pool in pools.values()),
        "candidates": sum(len(record["candidates"]) for record in prepared),
        "attempts": sum(len(record["attempts"]) for record in prepared),
        "generation_performed": False, "labels_read": False, "candidate_execution": False,
        "greedy_definition": "attempt 0 from original actual 11-condition generation; distinct from pool base reranking",
        "sequence_protocol": source_manifest["sequence_protocol"],
    }
    run_hash = inference.prepare_output(output, manifest, resume)
    serialized = []
    for record in prepared:
        record = dict(record, run_sha256=run_hash)
        record["record_sha256"] = inference.object_hash(record)
        serialized.append(inference.canonical_json(record) + "\n")
    _write_exact(output / PREPARED_NAME, "".join(serialized).encode("utf-8"))
    _write_exact(output / "inputs.jsonl", sources["inputs"])
    if references is not None:
        _write_exact(output / "references.jsonl", sources["references"])
    return {"prepared_dir": str(output), "run_sha256": run_hash,
            **{key: manifest[key] for key in ("samples", "original_candidates", "candidates", "attempts", "generation_performed")}}


def _load_prepared(prepared_dir: Path) -> tuple[dict, str, list, list, dict]:
    prepared_dir = Path(prepared_dir)
    manifest = predict._json((prepared_dir / "manifest.json").read_bytes())
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("extraction_protocol") != EXTRACTION_PROTOCOL):
        raise ValueError("unsupported cache rescore preparation protocol")
    run_hash = inference.object_hash(manifest)
    source = manifest["source_manifest"]
    if inference.object_hash(source) != manifest["source_run_sha256"]:
        raise ValueError("prepared source manifest hash mismatch")
    paths = {"inputs": prepared_dir / "inputs.jsonl"}
    if manifest["references_present"]:
        paths["references"] = prepared_dir / "references.jsonl"
    for name, path in paths.items():
        if inference.file_hash(path) != manifest["source_sha256"][name]:
            raise ValueError(f"prepared {name} bytes differ from source")
    rows = inference.load_inputs(paths["inputs"])
    references = inference.load_references(paths.get("references"), {row["sample_id"] for row in rows})
    records = predict._records((prepared_dir / PREPARED_NAME).read_bytes(), PREPARED_NAME)
    for record in records:
        predict._record(record, PREPARED_FIELDS, run_hash)
    payloads = [{key: value for key, value in record.items() if key not in {"run_sha256", "record_sha256"}}
                for record in records]
    if inference.object_hash(payloads) != manifest["records_payload_sha256"]:
        raise ValueError("prepared candidate/source mapping payload hash mismatch")
    if ([record["sample_id"] for record in records] != [row["sample_id"] for row in rows]
            or manifest["samples"] != len(records)
            or manifest["candidates"] != sum(len(record["candidates"]) for record in records)
            or manifest["attempts"] != sum(len(record["attempts"]) for record in records)):
        raise ValueError("prepared samples do not match the complete source input order")
    return manifest, run_hash, records, rows, references


class _NoGenerationBackend:
    def __init__(self, backend: Any):
        self.backend = backend

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def generate(self, *args, **kwargs):
        raise ValueError("cache rescoring must never generate a replacement candidate")


def score(prepared_dir: Path, config: inference.InferenceConfig, backend: Any = None,
          resume: bool = False) -> dict:
    """Retokenize every new candidate and compute four fresh, finite sumlogp.

    Injecting a backend is only for synthetic offline tests. Normal execution
    loads the existing local model via HFBackend, with downloads disabled.
    """
    prepared_dir = Path(prepared_dir).resolve()
    preparation, preparation_hash, records, rows, references = _load_prepared(prepared_dir)
    source_config = preparation["source_manifest"]["config"]
    current = asdict(config)
    if (config.fence_policy != "unwrap-single" or any(current[key] != value
            for key, value in source_config.items() if key not in {"model_path", "device", "fence_policy"})):
        raise ValueError("rescore config must preserve the original model/scoring settings and use unwrap-single")
    model = inference.model_fingerprint(Path(config.model_path))
    if model != preparation["source_manifest"]["model"]:
        raise ValueError("rescore model/tokenizer assets differ from the original model fingerprint")
    refs_path = prepared_dir / "references.jsonl" if preparation["references_present"] else None
    manifest = inference.make_manifest(config, prepared_dir / "inputs.jsonl", refs_path, model)
    manifest["cache_rescore"] = {
        "schema_version": SCHEMA_VERSION, "extraction_protocol": EXTRACTION_PROTOCOL,
        "implementation_sha256": inference.file_hash(Path(__file__)),
        "prepared_run_sha256": preparation_hash,
        "prepared_records_sha256": inference.file_hash(prepared_dir / PREPARED_NAME),
        "source_run_sha256": preparation["source_run_sha256"],
        "source_sha256": preparation["source_sha256"],
        "generation_performed": False, "old_scores_reused": False,
        "backend": "HFBackend" if backend is None else "injected_synthetic_backend",
    }
    output = _canonical_destination(prepared_dir / "inference")
    run_hash = inference.prepare_output(output, manifest, resume)
    active = inference.HFBackend(config) if backend is None else backend
    candidate_records = []
    for record in records:
        candidates = []
        for candidate in record["candidates"]:
            tokens = active.code_tokens(candidate["code"])
            candidates.append(dict(candidate, completion_token_ids=tokens, token_count=len(tokens), eos_included=True))
        candidate_record = {"sample_id": record["sample_id"], "candidates": candidates,
                            "attempts": record["attempts"], "run_sha256": run_hash,
                            "reference_kind": record["reference_kind"],
                            "reference_provenance": record["reference_provenance"]}
        candidate_record["record_sha256"] = inference.object_hash(candidate_record)
        candidate_records.append(candidate_record)
    content = "".join(inference.canonical_json(record) + "\n" for record in candidate_records).encode("utf-8")
    _write_exact(output / "candidates.jsonl", content)
    result = inference.run_inference(rows, references, config, output, run_hash, _NoGenerationBackend(active))
    # The same consumer validation used by prediction proves complete common
    # candidate coverage, raw finite scores and all derived-score identities.
    predict._validated_run({"manifest": (output / "manifest.json").read_bytes(),
                            "inputs": (prepared_dir / "inputs.jsonl").read_bytes(),
                            "candidates": (output / "candidates.jsonl").read_bytes(),
                            "scores": (output / "scores.jsonl").read_bytes()})
    return dict(result, inference_dir=str(output), run_sha256=run_hash, generation_performed=False)


def export_greedy(prepared_dir: Path, output: Path) -> dict:
    """Export the actual original first generation after complete fresh scoring.

    Requiring a verified scored run also checks token/EOS eligibility. A missing
    or truncated greedy attempt fails the entire export instead of dropping a
    sample. The pool's base argmax is a different baseline.
    """
    prepared_dir = Path(prepared_dir).resolve()
    _, prepared_hash, records, _, _ = _load_prepared(prepared_dir)
    run_dir = prepared_dir / "inference"
    manifest, _, pools, _, _, _ = predict._validated_run({
        "manifest": (run_dir / "manifest.json").read_bytes(),
        "inputs": (prepared_dir / "inputs.jsonl").read_bytes(),
        "candidates": (run_dir / "candidates.jsonl").read_bytes(),
        "scores": (run_dir / "scores.jsonl").read_bytes(),
    })
    if manifest.get("cache_rescore", {}).get("prepared_run_sha256") != prepared_hash:
        raise ValueError("greedy export scored run differs from preparation fingerprint")
    predictions = []
    for record in records:
        mappings = [mapping for mapping in record["mappings"] if mapping["attempt"] == 0]
        if (len(mappings) != 1 or mappings[0]["source"] != "greedy:0" or not mappings[0]["eligible"]
                or mappings[0]["candidate_id"] not in pools[record["sample_id"]]):
            raise ValueError(f"{record['sample_id']}: actual greedy attempt has no complete eligible candidate")
        code = pools[record["sample_id"]][mappings[0]["candidate_id"]]["code"]
        predictions.append({"sample_id": record["sample_id"], "method": "greedy", "predicted_code": code})
    output = _canonical_destination(output)
    if output.exists():
        raise ValueError("greedy output already exists; no overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(inference.canonical_json(row) + "\n")
    return {"samples": len(predictions), "method": "greedy", "output": str(output), "generation_performed": False}
