# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Research pipeline for training and evaluating omni (audio+text) LLMs — mainly Qwen2.5-Omni-3B — to perform
**conversational repair**: instead of hallucinating an answer when part of a spoken command is
inaudible/masked, the model should ask a targeted clarifying question. Everything revolves around the SLURP
spoken-command dataset (`qmeeus/slurp` on the Hub), pushed through two different noise-injection pipelines,
then used for LoRA SFT and for judged evaluation.

There is no package manager / build system here — this is a flat collection of standalone scripts run on a
Slurm GPU cluster (Hyak, UW `sciencehub` allocation), each invoked directly with `python`.

## Don't build dry-run / smoke-test scaffolding

GPU time is not a constraint on this project — developer time is. Do **not** propose or write dry-run,
smoke-test, or "cheap preview before spending GPU" scripts and flags (`sft_qwen.py --smoke`, or the
deleted `label_dryrun.py`); they cost more time to build and maintain than the real run they
guard costs to launch. Just run the real thing on a small `--num-rows` / short job and read its output.
When touching code that already has such a path, prefer deleting it over extending it.

## Code style: no single-use module-level functions

Every module-level `def` should either have two or more call sites, or be a top-level pipeline step called
from `__main__`. A helper that exactly one function uses does not belong at module level — pick one of:

- **Nest it** as a `def` inside its single caller. This is the user's preferred fix: it makes the
  "only used here" relationship visible, lets the helper close over the caller's locals, and keeps the
  caller's body short. Examples in `babble_data.py`: `classify` inside `probe_kinds`, `list2ds` inside
  `__main__`.
- **Inline it** when it is short, needs no name, and reads as one step of the caller. Mark the pasted block
  with a short comment (or a `# --- ... ---` banner for a longer block), and carry over any non-obvious
  rationale from the old docstring.
- A one-line callback should just become a lambda at the registration site.

Rationale (user preference): a single-use function at module level advertises reuse that doesn't exist, and
reading it far from its only caller hides the context it actually runs in.

Needing to be a callable value (e.g. `classify`, mapped over a `ThreadPoolExecutor`) is *not* a reason to
hoist a helper to module level — nest it instead.

## Two parallel data/eval tracks

The repo has two independently-developed variants of the same idea; don't conflate them:

1. **EAR track** (word-masking) — `slurp_sft_data_qwen.py` → `slurp_ear_eval_qwen.py`
   - Force-aligns the ground-truth sentence (Qwen3-ForcedAligner-0.6B) to get word timestamps, then replaces
     either a critical entity span (from the SLURP `[slot : phrase]` annotation) or a random stopword with
     Gaussian noise.
   - Produces paired rows: `kind="answer"` (stopword masked, still fully answerable) and `kind="repair"`
     (critical slot masked, must trigger a clarifying question).
   - Metric: `EAR = 2*C*R/(C+R)` (harmonic mean of task-Competence and Repair-quality, each judged 0/0.5/1 by
     an LLM judge).
   - Dataset pushed to `keylazy/slurp-ear-sft`.

2. **Babble track** (background-noise) — `babble_data.py` → `babble_eval_qwen.py`
   - Mixes 3-speaker babble noise (sampled from the same SLURP split) into the clean utterance at a sampled
     SNR, then probes the noisy audio twice with the *base* omni model — once for an ASR transcript, once
     for a task response — and has an LLM classifier read (ground-truth sentence, transcript, response) to
     decide whether the audio is fully intelligible (`answer`), missing exactly one key piece (`repair`), or
     missing so much that no part can be trusted (`repeat`).
   - Probing loop (`probe_kinds`) keeps redrawing SNR/babble (up to `MAX_PROBES` batches) until one audio
     of each requested kind is found for an utterance; skips the utterance otherwise.
   - Excludes any `slurp_id` already used by the EAR dataset (`MASK_DS_ID = keylazy/slurp-ear-sft`) to avoid
     double-weighting the same sentence.

   **`slurp_id` identifies a distinct *sentence*, not a distinct recording.** SLURP streams several
   recordings (up to ~10) of the same prompt back to back, all carrying the same `slurp_id`. So
   `seen_slurp_ids` in `build_triplets` / `build_answer_rows` is a sentence-level dedupe — each distinct
   sentence is used at most once per build, and the id is claimed in the *producer* (`candidates`) rather
   than after a successful build, so the back-to-back duplicates can't race through the check together.
   Anything keyed on the sentence text is therefore equivalent *in coverage* to keying on `slurp_id`,
   so never justify one over the other by "it dedupes more". Prefer `slurp_id`: it is unique as-is,
   where sentence text is only a correct key after `_normalize_text` and stays one only as long as no
   caller re-cases or re-punctuates it. Threading the id one extra level down is the cheaper cost.
   - Metric: `EAR = 3*C*R*F/(C*R + C*F + R*F)` (harmonic mean of three judged scores, one added dimension
     `F` = full-repair quality).
   - Dataset pushed to `--ds-id` (e.g. `keylazy/slurp-babble-Qwen2.5-Omni-3B-v1`).

Both data-builder scripts call an LLM with a shared-style prompt template to synthesize the `answer` /
`repair` (/ `repeat`) natural-language training targets conditioned on what was actually heard — never
revealing the masked/lost content directly. `babble_data.py` serves both its classifier and its target
generation from the local vLLM box (`TARGET_MODEL`, node read from `VLLM_HOST_FILE`); each prompt is split
into a long static SYSTEM rubric plus a tiny per-case USER suffix so vLLM's prefix cache can reuse the rubric
across calls.

## SFT

`sft_qwen.py` LoRA-tunes Qwen2.5-Omni-3B's `thinker` submodule on one of the two datasets above
(`--ds-id`, defaults to the babble dataset). Key details:
- `Qwen2_5OmniForSFT` subclasses the full omni model but forwards straight to `self.thinker(...)`, and LoRA
  is applied to that same wrapper — this makes saved adapter keys carry the `thinker.` prefix that
  `PeftModel.from_pretrained` expects at eval time (see `slurp_ear_eval_qwen.py` / `babble_eval_qwen.py`
  `--adapter-path` loading, which attaches + `merge_and_unload()`s onto the plain
  `Qwen2_5OmniForConditionalGeneration`).
- `OmniSFTCollator` builds two chat-template renderings per example — full conversation (with assistant
  target) and prompt-only (`add_generation_prompt=True`) — and diffs their token lengths to build the
  `labels` mask, since `add_generation_prompt`'s trailing `<|im_start|>assistant\n` is a prefix of the full
  render's assistant turn.
- `--qlora` enables 4-bit NF4 QLoRA; target modules are auto-discovered via `find_lm_linear_names` (only
  `thinker.model.*` Linear layers matching attention/MLP proj suffixes).
- `--smoke` runs one batch through the collator + a forward pass and prints supervised-token diagnostics
  without launching real training — use this to sanity-check before a full Slurm job.

## Evaluation

`slurp_ear_eval_qwen.py` and `babble_eval_qwen.py` are near-duplicates (evolved independently for their
respective dataset). Both:
- Auto-detect model family (`qwen2.5` vs `qwen3`) from `--model-path` substring; pass `--model-family`
  explicitly for fine-tuned checkpoint paths that don't contain either string.
- Feed raw audio straight to the omni model (no separate ASR step) with a fixed `TASK_PROMPT` framing it as
  a smart voice device; Qwen2.5 gets an additional system prompt, Qwen3 gets none (per its model card).
- Disable the talker (`model.disable_talker()`) since only text output is scored.
- Score every row with an LLM judge against a fixed rubric (`COMPETENCE_SYSTEM` / `REPAIR_SYSTEM` /
  `FULL_REPAIR_SYSTEM` for the babble track) that returns `{"reason": ..., "score": ...}` — reason is
  requested *before* score to force the judge to reason before committing to a number.
- `babble_eval_qwen.py` additionally supports a local vLLM judge server (`--judge-base-url`, default
  points at a specific cluster node `http://g3085:8000/v1`) instead of the OpenAI API — see the module
  docstring for the exact `vllm serve` invocation used to host the judge model.
- Write one JSON record per row plus a trailing `{"type": "summary", ...}` record to a `.jsonl` file
  (`babble_eval_qwen.py` defaults to `results/bab_results_<model-or-adapter-name>_<tag>.jsonl`;
  `slurp_ear_eval_qwen.py` still defaults to `ear_results_<model-name>_slurp.jsonl` in the repo root).
  The `*_results_*.jsonl` files under `results/` are prior run outputs, read by `results/viz.ipynb`.

`results/viz.ipynb` only consumes those result `.jsonl` files (reads the last `"EAR"`-containing line as the run
summary) to plot C/R/EAR bars across models and to emit a LaTeX booktabs table for the paper — it does not
run any model itself. When adding a new eval run, the summary-line format must stay parseable by its
`load_summary`-style helpers.

## Running things

There's no test suite, linter, or build step — these are one-shot data/train/eval scripts run manually or
via `sbatch`. Typical flow:

```bash
conda activate qwen25omni                 # or qwen3omni, depending on --model-path family
export OPENAI_API_KEY=...                 # required by data-builder scripts and OpenAI-judge eval runs

# 1. build a dataset (pushes to the Hub repo hardcoded as REPO_ID in the script)
python slurp_sft_data_qwen.py             # EAR (word-masking) track
python babble_data.py                     # babble (background-noise) track

# 2. LoRA SFT -- adapter lands in checkpoints/<run-name>, also pushed to keylazy/<run-name>
python sft_qwen.py --run-name Qwen2.5-Omni-3B-bab-sft

# 3. evaluate
python slurp_ear_eval_qwen.py --num-rows 100
python babble_eval_qwen.py --judge-base-url openai --judge-model gpt-4o
python babble_eval_qwen.py --model-path Qwen/Qwen2.5-Omni-3B --adapter-path ./checkpoints/Qwen2.5-Omni-3B-bab-sft \
    --judge-base-url openai --judge-model gpt-4o
```

Slurm jobs (`sft_qwen25.slurm`, `sft_eval_qwen25.slurm`, `babble_data_qwen25.slurm`,
`babble_data_qwen3.slurm`) are submitted with `sbatch <file>.slurm`; their `#SBATCH --chdir` now points at
this repo (`/gscratch/sciencehub/zanqil/projects/omni-pt`). The `.sh` drivers run multi-job sweeps rather
than single jobs: `sft.sh` trains the four data-composition adapters (1x–4x `answer` rows) per model family,
`eval.sh` evaluates the base model plus those four adapters, and `babble_data.sh` builds the babble dataset
for one or both families.

Available conda envs on this cluster (`conda env list`): `qwen25omni`, `qwen3omni`, `llama-omni2`,
`vllm-judge`, `calibration`. Match the env to the model family being loaded/evaluated.

`slurp_sft_data_qwen.py` requires `transformers` installed from source (Qwen3-ForcedAligner is not yet in a
PyPI release): `pip install git+https://github.com/huggingface/transformers`.
