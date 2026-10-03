# Candidate generation and four-condition scoring

This is the first **shared-candidate, sequence-rescoring prototype** of the
research proposal. It is not online token-level contrastive decoding, a trained
student simulator, or an executable grading pipeline. Generated and historical
student programs are handled only as text. This module never runs them.

## Inputs and labels

Use the prepared `inputs.train.jsonl`, `inputs.dev.jsonl`, or `inputs.test.jsonl`.
Keep `labels.*.jsonl` in separate files. There is no labels argument and the
inference process does not scan adjacent files. Never merge labels into inputs.

Each input contains:

| Field | Meaning |
| --- | --- |
| `schema_version` | Data schema identifier, currently `student-sim-cd.progfeed.v1` |
| `sample_id`, `student_id`, `lab`, `source_file` | Stable sample and provenance identifiers |
| `current_timestamp` | Time of the current submission |
| `problem_statement` | Nonempty task description; mandatory for actual inference |
| `current_code` | Complete current source file |
| `history` | Earlier `{timestamp, code, results, feedback}` records |
| `feedback` | Actual current `{test_name, function_name, assigned_type, text}` messages |
| `current_results` | Basic `{test_name, function_name, status, score, max_score, testcase_mask}` records |

Top-level and nested history/result/feedback fields use explicit allowlists.
Fields such as `target_code`, `target_timestamp`, `target_results`, or
`next_timestamp` are rejected, rather than ignored. When `current_timestamp` is
present, real-history times must strictly increase and precede it. Official
ProgFeed `YYYY-MM-DD-HH-MM-SS` and ISO timestamps are supported; no missing timezone
is guessed. Data preparation remains responsible for provenance and student splits; field validation cannot
prove that arbitrary text does not contain future information. The current
format represents one source file, so it does not automatically provide companion
files for a multi-file assignment.

The model sees the task description, source-file name, current code, historical
code/results/feedback, basic current results, and current feedback. Student and
sample identifiers are kept in provenance, not included in the prompt.

The following check needs only Python's standard library:

```sh
PYTHONPATH=src python3 -B -m student_sim_cd.inference \
  --inputs /path/to/prepared/inputs.test.jsonl --validate-only
```

`--validate-only --allow-missing-problem` permits a schema-only engineering
diagnostic and reports missing descriptions. This option is forbidden during
real inference. Validation success is not a model result.

## Reference history and the four conditions

For `l11`, `l01`, `l10`, and `l00`, the first index selects real/reference history
and the second selects present/absent **current specific feedback**. The same
task, current code, and basic results remain in all four prompts. Past feedback
remains part of each chosen history. The feedback-absent branch supplies an
empty current-feedback list; it does not erase basic test information.

Optional `--references /path/to/references.jsonl` reads one record per input:

```json
{
  "sample_id": "the-input-sample-id",
  "reference_kind": "matched",
  "reference_history": [
    {"timestamp": "2020-01-01T09:00:00", "code": "x = 0\n", "results": [], "feedback": []}
  ],
  "provenance": {"split": "train", "matching_rule": "document the past-only rule here"}
}
```

The allowed kinds are `matched`, `shuffled`, and `empty`. Matching itself is not
implemented here: the supplied history and provenance must come from an explicit,
audited, prediction-time-available construction. An explicit references file must
cover every input exactly once. Missing rows never trigger a fallback.

The separate [selection module](../src/student_sim_cd/select_inputs.py) now
implements the declared training-input nearest-neighbor protocol described below.

Without `--references`, all records are labelled
`empty_reference_diagnostic`, with the explanation `not matched main`. This is
a reference ablation, not the proposal's matched-reference main setting. Merely
labelling a supplied record `matched` does not independently validate the match.

## Real model run

The suggested first backbone is
[Qwen/Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct).
Its official model card provides the Transformers loading and chat-template
interface. The code targets PyTorch 2.6 and Transformers 4.51, using a single model
instance on one explicit device. The four branches are scored sequentially with
batch size one. No model is loaded four times.

Download and provision a model **separately** in an approved environment. The
inference command requires an existing local directory with safetensors weights,
tokenizer files, and configuration. It sets `local_files_only=True`,
`trust_remote_code=False`, and `use_safetensors=True`; it neither downloads a model
nor installs packages. PyTorch/Transformers imports are delayed until a real run.
GPU availability, laboratory rules, and memory use must be checked by the caller.

```sh
PYTHONPATH=src python -B -m student_sim_cd.inference \
  --inputs /path/to/prepared/inputs.test.jsonl \
  --model-path /path/to/models/Qwen2.5-Coder-7B-Instruct \
  --output /path/to/outputs/first-run \
  --device cuda:0 --dtype bfloat16 \
  --num-generations 4 --max-new-tokens 1024 --max-context-tokens 8192
```

These counts and lengths are explicit prototype defaults, not a claim that they
adequately cover the data. Freeze the eventual scientific configuration on
development data before test evaluation. A run requires every input to have a
description. Prepare and document any evaluation subset outside this module.

Each sample includes the unchanged current code as a copy baseline plus one
greedy attempt and `num_generations - 1` sampled attempts under condition `11`.
Sampling defaults are temperature `0.8`, top-p `0.95`, with top-k filtering disabled.
A fresh generation configuration avoids inheriting hidden sampling or forced-EOS
defaults from a model package. Attempt seeds are derived
from the configured seed, sample ID, and attempt index, so resuming a sample does
not depend on how many previous samples ran. Exact source text is deduplicated;
duplicate attempts and source identities remain recorded. The actual next
submission is never supplied or deliberately inserted into the candidate pool.

All methods rank the same completed candidates. Candidate coverage and selection
quality therefore need separate evaluation. A pool containing only the copy
baseline, for example after all generated attempts are truncated, is not evidence
that methods agree on good student behavior.

## Code text, length, and EOS

The default fence policy is `preserve`: no `.strip()`, indentation fixes, syntax
repair, or heuristic code extraction is applied. If a predeclared experiment uses
`--fence-policy unwrap-single`, only a single exact outer Python/unlabelled fence
is removed. Interior indentation, blank lines, and the newline before the closing
fence are preserved. Raw text and the named transformation are saved. Prose,
multiple fences, or malformed wrappers are not silently repaired.

The generator stores raw generated token IDs and decoded text. Scoring uses one
canonical tokenization of each candidate's stored `code`, followed by exactly one
tokenizer EOS token, identical in all four branches. `token_count` **includes EOS**.
Scores sum log probabilities over code and EOS, excluding all prompt tokens.
There is no length normalization. Re-tokenizing decoded text can differ from the
original generated token path; both the original path and canonical scored path
are saved. The score is for the declared canonical representation, not a sum over
all tokenizations of a program.

When the model explicitly supports `logits_to_keep`, scoring requests only the
hidden-state positions predicting completion tokens (including EOS), avoiding
unused prompt vocabulary logits. Models without this explicit parameter keep the
full-forward path. This is an equivalent computation, with no change to prompts,
four conditions, or summed scores; CPU tests on a tiny random Qwen2 compare both
paths for EOS-only and multi-token completions. It is not a measured performance
claim for a particular GPU or model size.

Generation stops on the same tokenizer EOS used for scoring. An attempt that hits
its token budget without EOS remains in `attempts` with `eos_reached=false` and an
exclusion reason; it is not presented as a complete-code candidate. Reserved special
tokens inside code are explicitly excluded. Empty source files are retained and
scored as an EOS-only completion, including the unchanged empty-file baseline. Other complete
but invalid Python remains in the pool: code validity is not inferred by running
it. A malformed/fenced candidate may therefore remain textually valid input to
the scorer while being invalid source code for a later isolated grader.

No branch is truncated. The command checks each prompt plus the full generation
budget, and separately each prompt plus its scored candidate including EOS,
against `max_context_tokens`. The configured context must not exceed the model's
declared `max_position_embeddings`. Overlength inputs stop the run with an
explicit error; overlength generated candidates retain an explicit exclusion
reason. Any history-window policy or larger context configuration must be a new,
documented input/configuration, not a silent fallback. Nonfinite log probabilities
are errors. No result is replaced with an arbitrary sentinel score.

## Output and resume

### Cached development protocol correction (2026-10-02)

`configs/cache_rescore_v1.json` freezes `cached-python-fence-v1` before the new
development rescore. The existing 70-sample run is complete; its 280 attempts and
original scores remain intact. This correction generates no new candidates.
Only a response consisting entirely of one lower-case `python` or `py` outer
fence is unwrapped, with LF/CRLF and all payload whitespace retained. Unlabelled,
multiple, malformed or prose-containing wrappers retain their exact raw text and
an explicit reason. The unchanged current-code candidate is never transformed.
Syntax errors do not remove candidates. Every original attempt and old-to-new
candidate mapping is kept, and source SUCCESS, input/reference bytes and model
fingerprints must match before scoring.

The new candidate pool is deduplicated by exact UTF-8 content per sample and all
four conditions are recomputed with the same backbone, prompts, EOS and sum-logp
protocol. Original greedy attempt 0 is separately evaluated using the same
extraction; it is distinct from shared-pool base reranking. All 70 samples and
17 students remain present. Fixed comparisons and paired student bootstrap are
recorded in the protocol. This is a development-set protocol correction after
the first-run format audit, not a preregistration before the original experiment
or a confirmatory test result. Static metrics cannot establish correctness,
feedback uptake or student causal effects.

| File | Contents |
| --- | --- |
| `manifest.json` | Full configuration; SHA-256 of input/reference files, implementation, scoring code, model artifacts; package versions; sequence protocol |
| `candidates.jsonl` | One record per sample: code, exact-text candidate ID, sources, canonical token IDs/count, all raw attempts, reference mode/provenance |
| `scores.jsonl` | One record per sample/candidate: `l11/l01/l10/l00`, EOS-inclusive length, prompt lengths, contrasts and named scores |

Every cache record carries a manifest hash and its own content hash. Resume with
the same command plus `--resume`. Input, reference, model artifact, configuration,
implementation, or dependency-version changes cause an explicit refusal. Model
fingerprinting reads the local weight bytes; this costs disk I/O but detects a
changed model behind the same directory name. A different experiment needs a new
output directory.

Completed candidate sets are written after a sample's attempts, and score records
after all four branches of one candidate. An interrupted partial sample/score is
recomputed, without redoing completed records. Complete records are flushed and
synced. A torn JSONL line or edited cache fails loudly; no partial line or score is
silently discarded. Preserve the damaged output and recover deliberately before
resuming. Only one process may write an output directory.

## Scoring API and A's coverage requirement

`student_sim_cd.scoring` has only standard-library dependencies:

```python
from student_sim_cd.scoring import LogProbs, method_scores, softmax, calibrate_groups

logp = LogProbs(l11=-7, l01=-10, l10=-12, l00=-14)
scores = method_scores(logp, weight=1)
# base=-7, history_d0=-5, cd=-4, b=-6, joint=-6 (default alpha=0, beta=1)
```

`components` computes `D0=l10-l00`, `D1=l11-l01`, and `Gamma=D1-D0`.
`score(logp, alpha, beta)` computes `l11 + alpha*D0 + beta*Gamma`.
`method_scores` provides base, historical-D0 control, ordinary CD, B, and an
explicitly weighted joint score. `scores_weight_1` in inference output is a
convenient fixed diagnostic; final parameter selection must follow the evaluation
protocol. `softmax` produces a distribution over the common pool.

`calibrate_groups(scores, groups, q)` imposes exactly the supplied group masses
through groupwise softmax. Every candidate group must have an explicit q value;
q must be finite, nonnegative, and sum to one. If a positive-q group has no
candidate, `MissingGroupError.missing_groups` names the missing groups. It does
**not** renormalize q over available groups. Report coverage and apply a separately
declared resampling or fallback protocol. A missing zero-mass group is harmless.

Inference does not create candidate execution labels, fit q, or use true next
results to choose a group. A evaluations need independently supplied, valid
candidate groups and a shared prediction-time q. A separate static prototype
has now fitted text-change q on training students and evaluated shared-pool
distributions; see the [research summary](RESEARCH_SUMMARY.md). It labels exact
text equality rather than executable progress. Until grading isolation is
accepted, there are no real execution-based A results. The group-mass guarantee
concerns the probability distribution, not frequencies of an argmax-only output.

## Validation status

After a **complete** run, export fixed-weight predictions without loading the
model again or reading labels:

```sh
PYTHONPATH=src python3 -B -m student_sim_cd.predict \
  --run-dir /path/to/outputs/first-run \
  --inputs /path/to/the/exact-inputs-used.jsonl \
  --output /path/to/outputs/first-run-predictions --include-copy
```

The exporter validates input/manifest/record hashes, all sample and candidate
coverage, and the raw four-condition scores. It writes separate
`predictions.base.jsonl`, `predictions.history_d0.jsonl`, `predictions.cd.jsonl`,
`predictions.b.jsonl`, and optional `predictions.copy.jsonl`. Coefficients are
respectively `(alpha,beta)=(0,0),(1,0),(1,1),(0,1)`; ties use the smallest candidate
content hash. Each file can be evaluated separately with `student_sim_cd.evaluate`.
`base` here is a shared-pool likelihood reranker, not a claim that its selected
code equals the single greedy generation. No A calibration is performed by this
exporter. Existing output directories are never overwritten.

Local standard-library tests use synthetic fixtures and a fake inference backend:

```sh
PYTHONPATH=src python3 -B -m unittest tests.test_scoring tests.test_inference -v
```

They check the CD/zero-weight/reference degeneracies, natural-q identity, exact
group marginals, missing-group failure, finite-value rejection, prompt conditions,
label separation, length/EOS protocol, duplicate/truncated attempts, and strict
resume. They do not establish GPU correctness, memory capacity, data quality,
candidate execution results, or method effectiveness. Real model runs and their
results must be recorded separately by the experiment driver.

## Frozen selection and statistical interpretation

`select_inputs` freezes exact `(lab, source_file)` pairs from
`configs/progfeed-single-file-scope.json`, requires nonempty actual feedback and a
task description, and retains **all** eligible dev/test samples. Optional
`--require-history` declares a narrower cohort. Labels are copied by sample ID
only; label contents and model results do not enter filtering or matching.
`--source-root` checks that the **current** submission directory contains exactly
the one declared Python file, without examining future submission file counts.

```sh
PYTHONPATH=src python3 -B -m student_sim_cd.select_inputs \
  --prepared data/prepared/progfeed-v2 \
  --scope configs/progfeed-single-file-scope.json \
  --source-root data/raw/progfeed \
  --output data/prepared/progfeed-selected-v1
```

The 2026-09-29 local run produced the following static counts, not model results:

| Cohort | Dev samples / students | Test samples / students |
| --- | --- | --- |
| All input-eligible samples, root output | 91 / 19 | 48 / 15 |
| Matched-reference subset, `matched/` | 70 / 17 | 27 / 10 |
| Excluded from matched: empty real history | 21 | 17 |
| Excluded from matched: no eligible earlier train donor | 0 | 4 |

The scope is lab02/to_do_list.py, lab03/triangle_class.py, lab06/nested_loops.py,
lab07/dictionaries.py, and lab09/files.py, following the data preparation review.
All four output input files passed inference schema validation with zero missing
descriptions. Original prepared files were not changed. `selection.json` records
source/output hashes, filters, scope, per-file counts, exclusions, and donor reuse;
the per-sample exclusion files preserve every reason. There is no sample count cap.

Matched references come only from training inputs with a different student and the
same lab/file. A donor's current timestamp and every history timestamp must precede
the query's current timestamp globally. Matching chooses lexicographically by
absolute history-count difference, current logged score-fraction difference,
known-result-count difference, then stable sample ID. The reference is the donor's
existing history; the donor's current code/feedback is not appended. Missing or
conflicting logged scores do not become zero. This logged fraction is not executed
test accuracy or a validated ability estimate. There is no tuned distance threshold.
The run used 60/24 distinct donor samples with maximum reuse 3/2 in dev/test.
Unmatched samples have no silent empty-reference fallback.

Interpretation requires these distinctions:

- Raw summed sequence log probabilities favor some lengths. Report selected token
  counts and edit sizes alongside behavior metrics for every method.
- Candidates come from the `l11` proposal plus a copy baseline. Report unique pool
  size, truncation, and copy-only pools; distinguish missing candidate coverage from
  a ranking failure. This comparison cannot establish free-generation superiority.
- With empty real and reference histories, both history contrasts vanish and CD/B
  reduce to base. Report the real-history-present stratum while retaining the full
  declared cohort.
- Deduplicated pool probabilities are not empirical generation frequencies, and
  argmax selection does not preserve A's q marginals.
- The matched cohort selects for available history, logged performance, and earlier
  training donors. Compare all baselines on those **same matched IDs**; do not read
  differences against the full empty-reference cohort as a method gain. Future
  empty/shuffled-reference ablations should also use the same IDs. Cluster evaluation
  by query student and disclose donor reuse. Neither four-condition differences nor
  these observational comparisons identify a causal feedback effect.
