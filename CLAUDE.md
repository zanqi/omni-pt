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

2. **Babble track** (background-noise) — `slurp_babble_data.py` → `slurp_bab_eval_qwen.py`
   - Mixes 3-speaker babble noise (sampled from the same SLURP split) into the clean utterance at a sampled
     SNR, using the *base* Qwen2.5-Omni-3B itself as an ASR probe to decide, per-SNR draw, whether the
     resulting audio is fully intelligible (`answer`), missing one detail (`repair`), or missing so much
     that no part can be trusted (`repair_full`). Diffing is done between the ASR transcript and ground
     truth (`difflib` on stemmed/Whisper-normalized tokens).
   - Probing loop (`probe_triplet`) keeps redrawing SNR/babble (up to `MAX_PROBES` batches) until one audio
     of each kind (`answer`/`repair`/`repair_full`) is found for an utterance; skips the utterance otherwise.
   - Excludes any `slurp_id` already used by the EAR dataset (`MASK_DS_ID = keylazy/slurp-ear-sft`) to avoid
     double-weighting the same sentence.
   - Metric: `EAR = 3*C*R*F/(C*R + C*F + R*F)` (harmonic mean of three judged scores, one added dimension
     `F` = full-repair quality).
   - Dataset pushed to `keylazy/slurp-babble-Qwen2.5-Omni-3B`.

Both data-builder scripts call GPT-4o (`OPENAI_MODEL = "gpt-4o"`, via `OpenAI()` client) with a shared-style
prompt template to synthesize the `answer` / `repair` (/ `repair_full`) natural-language training targets
conditioned on what was actually heard — never revealing the masked/lost content directly.

## SFT

`slurp_sft_qwen25.py` LoRA-tunes Qwen2.5-Omni-3B's `thinker` submodule on one of the two datasets above
(`--ds-id`, defaults to the babble dataset). Key details:
- `Qwen2_5OmniForSFT` subclasses the full omni model but forwards straight to `self.thinker(...)`, and LoRA
  is applied to that same wrapper — this makes saved adapter keys carry the `thinker.` prefix that
  `PeftModel.from_pretrained` expects at eval time (see `slurp_ear_eval_qwen.py` / `slurp_bab_eval_qwen.py`
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

`slurp_ear_eval_qwen.py` and `slurp_bab_eval_qwen.py` are near-duplicates (evolved independently for their
respective dataset). Both:
- Auto-detect model family (`qwen2.5` vs `qwen3`) from `--model-path` substring; pass `--model-family`
  explicitly for fine-tuned checkpoint paths that don't contain either string.
- Feed raw audio straight to the omni model (no separate ASR step) with a fixed `TASK_PROMPT` framing it as
  a smart voice device; Qwen2.5 gets an additional system prompt, Qwen3 gets none (per its model card).
- Disable the talker (`model.disable_talker()`) since only text output is scored.
- Score every row with an LLM judge against a fixed rubric (`COMPETENCE_SYSTEM` / `REPAIR_SYSTEM` /
  `FULL_REPAIR_SYSTEM` for the babble track) that returns `{"reason": ..., "score": ...}` — reason is
  requested *before* score to force the judge to reason before committing to a number.
- `slurp_bab_eval_qwen.py` additionally supports a local vLLM judge server (`--judge-base-url`, default
  points at a specific cluster node `http://g3085:8000/v1`) instead of the OpenAI API — see the module
  docstring for the exact `vllm serve` invocation used to host the judge model.
- Write one JSON record per row plus a trailing `{"type": "summary", ...}` record to a `.jsonl` file
  (defaults to `{ear,bab}_results_<model-or-adapter-name>_slurp.jsonl` in the repo root — the existing
  `*_results_*.jsonl` files here are prior run outputs, read by `eval.ipynb`).

`eval.ipynb` only consumes those result `.jsonl` files (reads the last `"EAR"`-containing line as the run
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
python slurp_babble_data.py               # babble (background-noise) track

# 2. LoRA SFT
python slurp_sft_qwen25.py --smoke        # sanity check collator/labels before a real run
python slurp_sft_qwen25.py --push         # full run, pushes adapter to --hub-id

# 3. evaluate
python slurp_ear_eval_qwen.py --num-rows 100
python slurp_bab_eval_qwen.py --judge-base-url openai --judge-model gpt-4o
python slurp_bab_eval_qwen.py --model-path Qwen/Qwen2.5-Omni-3B --adapter-path ./Qwen2.5-Omni-3B-bab-sft \
    --judge-base-url openai --judge-model gpt-4o
```

Slurm jobs (`sft_qwen25.slurm`, `slurp_babble_data.slurm`) are submitted with `sbatch <file>.slurm`; note
their `#SBATCH --chdir` points at an older path (`/gscratch/sciencehub/zanqil/asr-calibration/qwen_omni/sft`)
that predates this repo's current location — check/update that path before resubmitting.

Available conda envs on this cluster (`conda env list`): `qwen25omni`, `qwen3omni`, `llama-omni2`,
`vllm-judge`, `calibration`. Match the env to the model family being loaded/evaluated.

`slurp_sft_data_qwen.py` requires `transformers` installed from source (Qwen3-ForcedAligner is not yet in a
PyPI release): `pip install git+https://github.com/huggingface/transformers`.
