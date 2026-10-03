"""Frozen text-only prompt controls; never evaluate or execute student programs."""

from dataclasses import asdict
import fcntl
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

from . import cache_rescore, inference


PROMPTS = {
    "student_revision": inference.SYSTEM_PROMPT + (
        " Infer the student's likely next action from the patterns in their past revisions: "
        "which errors they address, how much they change, and which errors they leave unresolved. "
        "Predict one plausible next submission at their observed stage. Do not replace their "
        "approach with an expert solution merely because that solution is correct."
    ),
    "conservative_edit": inference.SYSTEM_PROMPT + (
        " Prefer retaining the student's structure and unaffected source lines. Make only the "
        "small local changes this student is likely to attempt in response to the current feedback. "
        "If they are likely to leave the code unchanged, reproduce it. Avoid wholesale refactoring "
        "or fixing every issue unless their past revision behavior supports that action."
    ),
}


def messages(row, references, method):
    if method not in PROMPTS:
        raise ValueError("unknown frozen prompt baseline")
    history, _, _ = inference._reference(row, references)
    result = inference.build_messages(row, "11", history)
    result[0] = {"role": "system", "content": PROMPTS[method]}
    return result


def audit(rows, references, tokenizer, config):
    result = []
    for row in rows:
        for method in PROMPTS:
            tokens = tokenizer.apply_chat_template(messages(row, references, method),
                tokenize=True, add_generation_prompt=True, truncation=False)
            inference.ensure_context(len(tokens), config.max_new_tokens, config.max_context_tokens)
            result.append({"sample_id": row["sample_id"], "method": method,
                           "prompt_tokens": len(tokens), "prompt_sha256": inference.object_hash(tokens)})
    return result


def _write(path, value):
    data = (inference.canonical_json(value) + "\n").encode("utf-8")
    with Path(path).open("xb") as stream:
        stream.write(data)


def _checkpoint(path, keys, run_hash):
    records = {}
    for item in inference.read_jsonl(path) if path.exists() else []:
        key = tuple(item[name] for name in keys)
        if (key in records or item.get("run_sha256") != run_hash
                or item.get("record_sha256") != inference.object_hash(
                    {name: value for name, value in item.items() if name != "record_sha256"})):
            raise ValueError("prompt baseline checkpoint hash/key mismatch; preserve existing bytes")
        records[key] = item
    return records


def _append(path, item, run_hash):
    item = dict(item, run_sha256=run_hash)
    item["record_sha256"] = inference.object_hash(item)
    inference.append_record(path, item)
    return item


def _validate_raw(item, reservation, prompt_hash, sample_id, attempt, config, backend):
    tokens, complete = item.get('generated_token_ids'), item.get('eos_reached')
    if (type(item.get('attempt')) is not int or item['attempt'] != attempt
            or item.get('prompt_sha256') != prompt_hash
            or reservation.get('prompt_sha256') != prompt_hash
            or type(item.get('seed')) is not int
            or item['seed'] != int(inference.object_hash([config.seed, sample_id, attempt])[:8], 16)
            or item.get('source') != ('greedy' if attempt == 0 else 'sampled')
            or type(complete) is not bool or not isinstance(item.get('raw_text'), str)
            or not isinstance(tokens, list) or not tokens or len(tokens) > config.max_new_tokens
            or any(type(token) is not int or token < 0 for token in tokens)
            or item.get('finish_reason') != ('eos' if complete else 'length_limit')
            or (complete and (tokens[-1] != backend.eos_id or backend.eos_id in tokens[:-1]))
            or (not complete and (backend.eos_id in tokens or len(tokens) != config.max_new_tokens))):
        raise ValueError('baseline raw generation provenance differs')
    tokenizer = getattr(backend, 'tokenizer', None)
    if tokenizer is not None:
        decoded = tokenizer.decode(tokens[:-1] if complete else tokens,
            skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if decoded != item['raw_text']:
            raise ValueError('baseline raw tokens do not decode to raw text')


def run(rows, references, config, backend, output, *, protocol_sha256, resume=False, model_identity=None):
    """Each prompt gets one greedy and twelve sampled attempts on every query.

    Models/inputs are preflighted by the parent. Durable checkpoints prohibit
    silent overwrites or retries that create extra draws. All candidate scores
    use that baseline's own full 11 prompt, canonical code plus one EOS.
    """
    if not rows or len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('empty/duplicate prompt baseline cohort')
    for row in rows:
        inference.validate_input(row)
    real = isinstance(backend, inference.HFBackend)
    if real and (not isinstance(model_identity, dict) or not model_identity.get('files') or not model_identity.get('sha256')):
        raise ValueError('real prompt baseline requires parent-verified model identity')
    versions = {}
    for name in ('torch', 'transformers', 'tokenizers', 'safetensors'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not-installed'
    output = Path(output).absolute()
    if output.is_symlink() or output.resolve() != output:
        raise ValueError("baseline output must be canonical")
    if output.exists() and not resume:
        raise ValueError("baseline output exists; preserve it or explicitly resume")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fingerprint = {"schema_version": "student-sim-cd.prompt-baselines.v1",
            "protocol_sha256": protocol_sha256, "config": asdict(config),
            "prompts": PROMPTS, "inputs_sha256": inference.object_hash(rows),
            "references_sha256": inference.object_hash(references),
            "implementation_sha256": inference.file_hash(Path(__file__)),
            "attempts_per_prompt": 13, "random_attempts": 12,
            "sequence_protocol": "canonical code tokens + exactly one tokenizer EOS; sum logp; no length normalization",
            "student_execution": False, "labels_read": False}
        fingerprint.update(backend='HFBackend' if real else 'injected_synthetic_backend',
                           model=model_identity, runtime_versions=versions)
        run_hash = inference.object_hash(fingerprint)
        marker = output / "checkpoint-manifest.json"
        if marker.exists():
            if json.loads(marker.read_text()) != fingerprint:
                raise ValueError("baseline resume provenance differs")
        else:
            if any(path.name != ".lock" for path in output.iterdir()):
                raise ValueError("unidentified baseline output; preserve it")
            _write(marker, fingerprint)
        if (output / "SUCCESS").exists():
            manifest = json.loads((output / "manifest.json").read_text())
            if ((output / "SUCCESS").read_text() != inference.file_hash(output / "manifest.json") + "\n"
                    or manifest.get("run_sha256") != run_hash
                    or any(inference.file_hash(output / name) != digest
                           for name, digest in manifest["files_sha256"].items())):
                raise ValueError("completed baseline changed")
            return manifest
        raw_path, score_path = output / "raw.jsonl", output / "scores.jsonl"
        reservation_path = output / 'reservations.jsonl'
        reserved = _checkpoint(reservation_path, ("sample_id", "method", "attempt"), run_hash)
        raw = _checkpoint(raw_path, ("sample_id", "method", "attempt"), run_hash)
        scored = _checkpoint(score_path, ("sample_id", "method", "candidate_id"), run_hash)
        expected = {(row["sample_id"], method, attempt)
                    for row in rows for method in PROMPTS for attempt in range(13)}
        if not set(raw) <= set(reserved) <= expected:
            raise ValueError("unexpected baseline generation checkpoint")
        if set(reserved) - set(raw):
            raise ValueError('unresolved baseline generation reservation; preserve it, no automatic redraw')
        predictions = {method + suffix: [] for method in PROMPTS for suffix in ("_greedy", "_pool")}
        summaries = []
        for row in rows:
            for method in PROMPTS:
                prompt = backend.prompt_tokens(messages(row, references, method))
                inference.ensure_context(len(prompt), config.max_new_tokens, config.max_context_tokens)
                prompt_hash = inference.object_hash(prompt)
                pool = {}
                def add(code, origin):
                    identifier = hashlib.sha256(code.encode("utf-8")).hexdigest()
                    if identifier not in pool:
                        tokens = backend.code_tokens(code)
                        inference.ensure_context(len(prompt), len(tokens), config.max_context_tokens)
                        pool[identifier] = {"code": code, "tokens": tokens, "sources": []}
                    pool[identifier]["sources"].append(origin)
                    return identifier
                add(row["current_code"], "copy_current")
                eligible = []
                greedy_id = None
                for attempt in range(13):
                    key = (row["sample_id"], method, attempt)
                    item = raw.get(key)
                    if item is None:
                        reserved[key] = _append(reservation_path, {'sample_id': row['sample_id'],
                            'method': method, 'attempt': attempt, 'prompt_sha256': prompt_hash}, run_hash)
                        started = time.monotonic()
                        generated = backend.generate(prompt, row["sample_id"], attempt)
                        item = _append(raw_path, dict(generated, sample_id=row["sample_id"],
                            method=method, prompt_sha256=prompt_hash,
                            generation_seconds=time.monotonic()-started), run_hash)
                        raw[key] = item
                    _validate_raw(item, reserved[key], prompt_hash, row['sample_id'], attempt, config, backend)
                    identifier, reason = None, None
                    if item["eos_reached"]:
                        if not item["generated_token_ids"] or item["generated_token_ids"][-1] != backend.eos_id:
                            raise ValueError("baseline raw generation EOS mismatch")
                        code, _, extraction = cache_rescore.extract_code(item["raw_text"])
                        try:
                            identifier = add(code, "greedy:0" if attempt == 0 else "sampled:" + str(attempt))
                        except ValueError as exc:
                            reason = str(exc)
                    else:
                        extraction, reason = "raw_preserved", "no EOS; incomplete generation"
                    if attempt == 0:
                        greedy_id = identifier
                        if identifier is None:
                            raise ValueError("baseline greedy is incomplete/ineligible; no substitute prediction")
                    eligible.append({"attempt": attempt, "candidate_id": identifier,
                                     "extraction": extraction, "exclusion_reason": reason})
                for identifier, candidate in sorted(pool.items()):
                    key = (row["sample_id"], method, identifier)
                    item = scored.get(key)
                    if item is None:
                        started = time.monotonic()
                        value = backend.sequence_logp(prompt, candidate["tokens"])
                        if not math.isfinite(value) or value > 0:
                            raise ValueError("invalid baseline log probability")
                        item = _append(score_path, {"sample_id": row["sample_id"], "method": method,
                            "candidate_id": identifier, "logp": value,
                            "prompt_sha256": prompt_hash, "token_count": len(candidate["tokens"]),
                            "completion_sha256": inference.object_hash(candidate["tokens"]),
                            'scoring_seconds': time.monotonic()-started}, run_hash)
                        scored[key] = item
                    if (not math.isfinite(item["logp"]) or item["logp"] > 0
                            or item["prompt_sha256"] != prompt_hash
                            or item["completion_sha256"] != inference.object_hash(candidate["tokens"])):
                        raise ValueError("baseline score provenance differs")
                best = min(pool, key=lambda identifier: (-scored[(row["sample_id"], method, identifier)]["logp"], identifier))
                for suffix, identifier in (("_greedy", greedy_id), ("_pool", best)):
                    predictions[method + suffix].append({"sample_id": row["sample_id"],
                        "method": method + suffix, "predicted_code": pool[identifier]["code"]})
                summaries.append({"sample_id": row["sample_id"], "method": method,
                    "candidates": len(pool), "eligible_attempts": eligible,
                    "prompt_token_count": len(prompt), "prompt_sha256": prompt_hash,
                    "sources": {identifier: candidate["sources"] for identifier, candidate in pool.items()}})
                print(inference.canonical_json({"stage": "prompt_baselines", "sample_id": row["sample_id"],
                    "method": method, "raw_attempts": 13, "candidates": len(pool)}), flush=True)
        if set(raw) != expected or len(scored) != sum(item["candidates"] for item in summaries):
            raise ValueError("baseline checkpoint coverage differs")
        files = {}
        for name, records in predictions.items():
            path = output / (name + ".jsonl")
            with path.open("x", encoding="utf-8") as stream:
                for item in records:
                    stream.write(inference.canonical_json(item) + "\n")
            files[path.name] = inference.file_hash(path)
        _write(output / "summary.json", summaries)
        for name in ("raw.jsonl", "scores.jsonl", "summary.json", "checkpoint-manifest.json", 'reservations.jsonl'):
            files[name] = inference.file_hash(output / name)
        manifest = {**fingerprint, "status": "success", "run_sha256": run_hash,
            "samples": len(rows), "attempts": len(raw), "scores": len(scored), "files_sha256": files,
            "generated_tokens": sum(len(item["generated_token_ids"]) for item in raw.values()),
            'generation_seconds': math.fsum(item['generation_seconds'] for item in raw.values()),
            'scoring_seconds': math.fsum(item['scoring_seconds'] for item in scored.values())}
        _write(output / "manifest.json", manifest)
        with (output / "SUCCESS").open("x", encoding="ascii") as stream:
            stream.write(inference.file_hash(output / "manifest.json") + "\n")
        return manifest
