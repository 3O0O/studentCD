"""Measure all four inference prompts with a local tokenizer, without model weights.

No label file, student program, model inference, network request, sample filter,
or truncation is part of this command. Coverage is reported, never enforced by
dropping a sample or changing the inference configuration.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import platform

from . import inference
from .inference import (
    CONDITIONS, PROMPT_VERSION, _reference, build_messages, canonical_json,
    file_hash, load_inputs, load_references, object_hash,
)


SCHEMA_VERSION = "student-sim-cd.preflight.v1"
DEFAULT_CONTEXT_LIMITS = (8192, 16384, 32768)
DEFAULT_MAX_NEW_TOKENS = (512, 1024, 2048, 4096)


@dataclass(frozen=True)
class PreflightConfig:
    model_path: str
    context_limits: tuple[int, ...] = DEFAULT_CONTEXT_LIMITS
    max_new_tokens: tuple[int, ...] = DEFAULT_MAX_NEW_TOKENS

    def __post_init__(self):
        for name in ("context_limits", "max_new_tokens"):
            values = getattr(self, name)
            if not values or any(type(value) is not int or value < 1 for value in values):
                raise ValueError("%s must contain positive integers" % name)
            if len(set(values)) != len(values):
                raise ValueError("%s must not contain duplicate values" % name)


def tokenizer_fingerprint(model_path: Path) -> dict:
    """Hash tokenizer artifacts and its model-config fallback, never weight files."""
    model_path = Path(model_path)
    if not model_path.is_dir():
        raise ValueError("--model-path must be an existing local tokenizer directory")
    patterns = (
        "tokenizer*.json", "tokenizer*.model", "special_tokens_map.json", "added_tokens.json",
        "vocab.json", "vocab*.txt", "merges.txt", "spiece.model", "sentencepiece*.model", "*.tiktoken",
        "chat_template*.jinja", "chat_templates/*.jinja", "config.json",
    )
    paths = sorted({path for pattern in patterns for path in model_path.glob(pattern) if path.is_file()})
    if not any(path.name != "config.json" for path in paths):
        raise ValueError("local directory has no recognized tokenizer artifacts")
    files = {path.relative_to(model_path).as_posix(): file_hash(path) for path in paths}
    model_limit = None
    config_path = model_path / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        value = config.get("max_position_embeddings")
        if type(value) is int and value > 0:
            model_limit = value
    return {"sha256": object_hash(files), "files": files,
            "model_config_max_position_embeddings": model_limit,
            "weight_files_read": False}


def load_local_tokenizer(model_path: Path):
    """Keep loading local even when called outside the CLI process."""
    if not Path(model_path).is_dir():
        raise ValueError("--model-path must be an existing local directory, not a Hub ID")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        str(Path(model_path).resolve()), local_files_only=True, trust_remote_code=False)


def measure(rows: list[dict], references: dict, tokenizer) -> list[dict]:
    """Use the same chat template and canonical code-plus-EOS convention as inference."""
    eos_id = tokenizer.eos_token_id
    if type(eos_id) is not int:
        raise ValueError("tokenizer must have a single integer EOS token")
    if not tokenizer.chat_template:
        raise ValueError("tokenizer must supply the inference chat template")
    special_ids = set(tokenizer.all_special_ids)
    records = []
    for row in rows:
        reference_history, reference_kind, _ = _reference(row, references)
        lengths = {}
        for condition in CONDITIONS:
            tokens = tokenizer.apply_chat_template(
                build_messages(row, condition, reference_history), tokenize=True,
                add_generation_prompt=True, truncation=False)
            if not isinstance(tokens, list) or not tokens or any(type(token) is not int for token in tokens):
                raise ValueError("chat template must return a nonempty flat integer token list")
            lengths[condition] = len(tokens)
        code_tokens = tokenizer.encode(row["current_code"], add_special_tokens=False, truncation=False)
        if not isinstance(code_tokens, list) or any(type(token) is not int for token in code_tokens):
            raise ValueError("code tokenizer must return a flat integer token list")
        reserved = sorted(set(code_tokens) & special_ids)
        copy_length = len(code_tokens) + 1
        records.append({
            "sample_id": row["sample_id"], "student_id": row["student_id"],
            "lab": row["lab"], "source_file": row["source_file"],
            "reference_kind": reference_kind,
            "input_record_sha256": object_hash(row),
            "reference_record_sha256": object_hash(references[row["sample_id"]]) if references else None,
            "prompt_tokens": lengths, "max_prompt_tokens": max(lengths.values()),
            "copy_code_tokens": len(code_tokens), "copy_code_plus_eos_tokens": copy_length,
            "prompt_plus_copy_tokens": {condition: count + copy_length for condition, count in lengths.items()},
            "copy_protocol_valid": not reserved, "copy_reserved_special_token_ids": reserved,
        })
    if not records:
        raise ValueError("inputs file has no samples")
    return records


def summarize(records: list[dict], config: PreflightConfig, model_limit: int | None) -> dict:
    total = len(records)
    if not total:
        raise ValueError("cannot summarize empty inputs")
    maximum_prompt = max(record["max_prompt_tokens"] for record in records)
    maximum_copy_context = max(max(record["prompt_plus_copy_tokens"].values()) for record in records)
    coverage = []
    for context in config.context_limits:
        for budget in config.max_new_tokens:
            generations = [r["max_prompt_tokens"] + budget <= context for r in records]
            copies = [max(r["prompt_plus_copy_tokens"].values()) <= context for r in records]
            both = [g and c for g, c in zip(generations, copies)]
            valid = [fit and r["copy_protocol_valid"] for fit, r in zip(both, records)]
            coverage.append({
                "max_context_tokens": context, "max_new_tokens": budget, "samples": total,
                "generation_budget_fits_all_branches": sum(generations),
                "copy_candidate_fits_all_branches": sum(copies),
                "both_fit_all_branches": sum(both), "both_fit_fraction": sum(both) / total,
                "both_fit_and_copy_protocol_valid": sum(valid),
                "generation_budget_fits_by_branch": {
                    condition: sum(r["prompt_tokens"][condition] + budget <= context for r in records)
                    for condition in CONDITIONS},
                "within_declared_model_capacity": context <= model_limit if model_limit is not None else None,
                "overflow_sample_ids": [r["sample_id"] for r, fits in zip(records, both) if not fits],
            })
    return {
        "samples": total, "branches_per_sample": len(CONDITIONS),
        "reference_modes": sorted({r["reference_kind"] for r in records}),
        "max_prompt_tokens": maximum_prompt,
        "max_prompt_tokens_by_branch": {condition: max(r["prompt_tokens"][condition] for r in records)
                                         for condition in CONDITIONS},
        "max_copy_code_plus_eos_tokens": max(r["copy_code_plus_eos_tokens"] for r in records),
        "minimum_context_for_all_copy_candidates": maximum_copy_context,
        "minimum_context_by_generation_budget": [{
            "max_new_tokens": budget,
            "generation_only": maximum_prompt + budget,
            "generation_and_copy_candidate": max(maximum_prompt + budget, maximum_copy_context),
            "within_declared_model_capacity": (
                max(maximum_prompt + budget, maximum_copy_context) <= model_limit
                if model_limit is not None else None),
        } for budget in config.max_new_tokens],
        "copy_protocol_invalid_sample_ids": [r["sample_id"] for r in records if not r["copy_protocol_valid"]],
        "coverage": coverage,
        "interpretation": {
            "model_run": False, "model_weights_loaded": False, "labels_read": False,
            "samples_filtered": False, "history_or_prompt_truncated": False,
            "inference_config_modified": False,
            "counts_are_tokenizer_specific": True,
            "unknown_generated_candidate_lengths": "not_observed;canonical_rescoring_must_still_check_each_candidate",
            "memory_or_runtime_feasibility": "not_measured_by_tokenizer_preflight",
        },
    }


def run_preflight(inputs: Path, references_path: Path | None, output: Path,
                  config: PreflightConfig, tokenizer=None) -> dict:
    """Injected tokenizers support synthetic unit tests, not a CLI model shortcut."""
    inputs, output = Path(inputs), Path(output)
    references_path = Path(references_path) if references_path is not None else None
    if output.exists():
        raise ValueError("output already exists; preflight only creates a new directory")
    rows = load_inputs(inputs)
    references = load_references(references_path, {row["sample_id"] for row in rows})
    artifacts = tokenizer_fingerprint(Path(config.model_path))
    tokenizer_injected = tokenizer is not None
    tokenizer = tokenizer if tokenizer_injected else load_local_tokenizer(Path(config.model_path))
    records = measure(rows, references, tokenizer)
    report = summarize(records, config, artifacts["model_config_max_position_embeddings"])
    versions = {"python": platform.python_version()}
    for name in ("transformers", "tokenizers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    manifest = {
        "schema_version": SCHEMA_VERSION, "prompt_version": PROMPT_VERSION,
        "preflight_implementation_sha256": file_hash(Path(__file__)),
        "inference_implementation_sha256": file_hash(Path(inference.__file__)),
        "inputs_path": str(inputs.resolve()), "inputs_sha256": file_hash(inputs),
        "references_path": str(references_path.resolve()) if references_path else None,
        "references_sha256": file_hash(references_path) if references_path else None,
        "config": asdict(config), "config_sha256": object_hash(asdict(config)),
        "tokenizer_artifacts": artifacts, "runtime_versions": versions,
        "tokenizer": {"class": type(tokenizer).__name__, "eos_token_id": tokenizer.eos_token_id,
                      "chat_template_sha256": object_hash(tokenizer.chat_template),
                      "all_special_ids": sorted(tokenizer.all_special_ids)},
        "loading": {"local_files_only": True, "trust_remote_code": False,
                    "model_weights_loaded": False, "synthetic_tokenizer_injected": tokenizer_injected},
    }
    run_hash = object_hash(manifest)
    report.update(schema_version=SCHEMA_VERSION, run_sha256=run_hash)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("manifest.json", manifest), ("report.json", report)):
        with (output / name).open("x", encoding="utf-8") as handle:
            handle.write(canonical_json(value) + "\n")
    with (output / "lengths.jsonl").open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical_json(dict(record, run_sha256=run_hash)) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-limits", type=int, nargs="+", default=list(DEFAULT_CONTEXT_LIMITS))
    parser.add_argument("--max-new-tokens", type=int, nargs="+", default=list(DEFAULT_MAX_NEW_TOKENS))
    args = parser.parse_args(argv)
    try:
        config = PreflightConfig(str(args.model_path.resolve()), tuple(args.context_limits), tuple(args.max_new_tokens))
        report = run_preflight(args.inputs, args.references, args.output, config)
        print(canonical_json({"samples": report["samples"], "model_run": False, "output": str(args.output),
                              "max_prompt_tokens": report["max_prompt_tokens"],
                              "minimum_context_by_generation_budget": report["minimum_context_by_generation_budget"]}))
        return 0
    except (ValueError, OSError, ImportError) as error:
        parser.exit(2, "preflight: %s\n" % error)


if __name__ == "__main__":
    raise SystemExit(main())
