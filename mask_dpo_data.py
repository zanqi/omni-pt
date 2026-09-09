"""
Build DPO preference pairs for the mask track (steps/mask.html, step 9).

On-policy: every candidate is sampled from the SFT checkpoint itself, under the
exact prompt mask_eval_qwen.py generates with, and ranked by the exact judge it
scores with. The preference signal and the reported metric therefore agree by
construction -- a pair that teaches the model something is a pair that moves
C or R.

  1. sample K responses per train row at temperature 1.0
  2. score all K with the per-kind rubric (ANSWER_JUDGE_SYSTEM / REPAIR_JUDGE_SYSTEM)
  3. chosen = best, rejected = worst, kept only if the margin is >= MIN_MARGIN
  4. all-K-perfect rows are dropped (no gradient); all-K-zero rows fall back to
     the dataset's written target as `chosen`, counted separately because a set
     dominated by those is really just more SFT
  5. reference log-probs for both sides, computed here while the SFT model is
     already resident -- that is exactly the DPO reference, so dpo_qwen.py
     never has to hold a second model

Writes a JSONL keyed by the dataset's row `id`; dpo_qwen.py joins it back onto
the audio. Deliberately not a pushed dataset: the audio is already on the Hub
under --ds-id and re-uploading it per DPO run buys nothing.

  python mask_dpo_data.py --config configs/mask-crf.yaml
"""

import argparse
import json
import os
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import soundfile as sf
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm

from mask_eval_qwen import JUDGE_BY_KIND, get_audio, judge_user, make_judge
from prompts import QWEN25_SYSTEM_PROMPT, task_prompt
from util import (
    detect_model_family,
    load_config,
    load_model,
    resolve_judge,
    seq_logprobs,
)

AUDIO_SAMPLING_RATE = 16000
MAX_AUDIO_SECONDS = 30
# a pair whose two sides score the same teaches nothing and still costs a
# forward pass on both; 0.5 is one rubric step
MIN_MARGIN = 0.5
JUDGE_WORKERS = 8


@dataclass
class Config:
    """The track YAML's key names, shared with the other four mask stages --
    hence `omni_path` for the base model and `sft_adapter` for the checkpoint
    being sampled from, which is dpo_qwen.py's key for the same path. `prefs`
    is spelled here too: this stage writes the file that stage names."""

    omni_path: str = "Qwen/Qwen2.5-Omni-3B"
    ds_id: str = "keylazy/slurp-mask-v1"
    train_split: str = "train"
    sft_adapter: str | None = None
    prefs: str | None = None
    dpo_samples: int = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        help="Track YAML (configs/*.yaml). Its keys become parser defaults, so "
        "any flag also given on the command line still wins.",
    )
    ap.add_argument("--ds-id", default="keylazy/slurp-mask-v1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument(
        "--adapter-path",
        help="The SFT checkpoint to sample from. It is also the DPO reference "
        "model, which is why the reference log-probs are computed here.",
    )
    ap.add_argument("--model-family", default=None, choices=["qwen2.5", "qwen3"])
    ap.add_argument("--out", default=None)
    # K=4 left two thirds of the rows with every sample already scoring 1.0
    # and no pair to make; each extra sample is another chance to catch the
    # policy failing, at one more judge call
    ap.add_argument("-k", "--samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--num-rows", type=int, default=-1)
    ap.add_argument(
        "--judge-model",
        help="from vllm --served-model-name; from openai, gpt-4o. Left unset "
        "on a vLLM judge, the served name is read off the server.",
    )
    ap.add_argument(
        "--judge-base-url",
        default="auto",
        help="'auto' reads the judge node out of VLLM_HOST_FILE, 'openai' uses "
        "the OpenAI API, or give a URL like 'http://g3085:8000/v1'.",
    )
    ap.add_argument("--judge-max-tokens", type=int, default=4096)
    ap.add_argument(
        "--plain-prompt",
        action="store_true",
        help="Sample under TASK_PROMPT. Must match how the adapter was trained "
        "and how it will be evaluated, or the pairs teach the wrong conditional.",
    )
    # --config has to be read before parse_args, because the file supplies
    # defaults rather than overrides -- a flag on the command line has to stay
    # able to beat it, and after parse_args an explicit flag is
    # indistinguishable from the default it happens to equal.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    config_path = pre.parse_known_args()[0].config
    if config_path:
        cfg = load_config(config_path, Config)
        print(f"config: {cfg}")
        ap.set_defaults(
            model_path=cfg.omni_path,
            ds_id=cfg.ds_id,
            split=cfg.train_split,
            adapter_path=cfg.sft_adapter,
            out=cfg.prefs,
            samples=cfg.dpo_samples,
        )
    args = ap.parse_args()
    # not required=True: the config supplies it, and argparse checks required
    # flags before set_defaults has been given a chance to fill them
    if not args.adapter_path:
        raise SystemExit("no SFT checkpoint: pass --adapter-path or sft_adapter:")

    # before the omni model is loaded: an unreachable box fails in seconds
    judge_url, judge_model = resolve_judge(args.judge_base_url, args.judge_model)

    family = args.model_family or detect_model_family(args.model_path)
    name = os.path.basename(args.adapter_path.rstrip("/"))
    out_path = args.out or os.path.join("results", f"mask_prefs_{name}.jsonl")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    ds = load_dataset(args.ds_id, split=args.split)
    ds = ds.cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE))
    if args.num_rows != -1:
        ds = ds.select(range(min(args.num_rows, len(ds))))
    print(f"sampling {args.samples}x over {len(ds)} rows of {args.ds_id}:{args.split}")

    model, processor = load_model(args.model_path, family, args.adapter_path)
    judge_fn = make_judge(
        judge_model, base_url=judge_url, max_tokens=args.judge_max_tokens
    )
    system_prompt = QWEN25_SYSTEM_PROMPT if family == "qwen2.5" else None
    prompt_text = task_prompt(False, args.plain_prompt)

    def conversation(wav_path, answer=None):
        conv = []
        if system_prompt is not None:
            conv.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
        conv.append(
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": wav_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        )
        if answer is not None:
            conv.append(
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            )
        return conv

    @torch.inference_mode()
    def sample_k(wav_path):
        """K sampled replies in one generate call -- the audio encoder runs once."""
        from qwen_omni_utils import process_mm_info

        conv = conversation(wav_path)
        text = processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos, *_ = process_mm_info(conv, use_audio_in_video=False)
        inputs = processor(
            text=text, audio=audios, images=images, videos=videos, return_tensors="pt"
        ).to(model.device).to(model.dtype)
        ids = model.generate(
            **inputs,
            return_audio=False,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            num_return_sequences=args.samples,
            thinker_max_new_tokens=args.max_new_tokens,
        )
        gen = ids[:, inputs["input_ids"].shape[1] :]
        return [
            t.strip()
            for t in processor.batch_decode(
                gen, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        ]

    @torch.inference_mode()
    def ref_logprob(wav_path, answer):
        """log P(answer | audio, prompt) under the SFT model.

        The label mask is built the way OmniSFTCollator builds it: render the
        conversation with and without the assistant turn and diff the token
        counts, since add_generation_prompt's trailing '<|im_start|>assistant\\n'
        is a prefix of the full render.
        """
        from qwen_omni_utils import process_mm_info

        full_conv = conversation(wav_path, answer)
        full_text = processor.apply_chat_template(
            full_conv, add_generation_prompt=False, tokenize=False
        )
        prompt_text_only = processor.apply_chat_template(
            conversation(wav_path), add_generation_prompt=True, tokenize=False
        )
        audios, images, videos, *_ = process_mm_info(full_conv, use_audio_in_video=False)
        full = processor(
            text=full_text, audio=audios, images=images, videos=videos,
            return_tensors="pt",
        )
        prompt = processor(
            text=prompt_text_only, audio=audios, images=images, videos=videos,
            return_tensors="pt",
        )
        ans_len = int(
            full["attention_mask"].sum() - prompt["attention_mask"].sum()
        )
        if ans_len <= 0:
            return None
        full = full.to(model.device).to(model.dtype)
        labels = torch.full_like(full["input_ids"], -100)
        labels[:, -ans_len:] = full["input_ids"][:, -ans_len:]
        logits = model.thinker(**full).logits
        return float(seq_logprobs(logits, labels)[0])

    kept, stats = 0, Counter()
    with open(out_path, "w", encoding="utf-8") as fout:
        for row in tqdm(ds, desc="pairs", unit="row", dynamic_ncols=True):
            arr, sr = get_audio(row["audio"])
            arr = arr[: MAX_AUDIO_SECONDS * AUDIO_SAMPLING_RATE]
            fd, wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                sf.write(wav_path, arr, sr)
                samples = sample_k(wav_path)
                # dedupe before judging: identical samples cost a judge call
                # each and can never form a pair with each other
                uniq = list(dict.fromkeys(s for s in samples if s))
                if not uniq:
                    stats["empty"] += 1
                    continue
                with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as ex:
                    judged = list(
                        ex.map(
                            lambda s: judge_fn(
                                JUDGE_BY_KIND[row["kind"]], judge_user(row, s)
                            ),
                            uniq,
                        )
                    )
                # score only, and a stable sort, so ties keep sampling order.
                # Breaking them on length made `chosen` the shortest sample of
                # the top score and `rejected` the longest of the bottom one,
                # which teaches brevity as much as it teaches repair.
                scored = sorted(
                    ({"text": s, "score": j[0], "reason": j[1]} for s, j in zip(uniq, judged)),
                    key=lambda d: d["score"],
                )
                best, worst = scored[-1], scored[0]

                pair_source = "sampled"
                if best["score"] - worst["score"] < MIN_MARGIN:
                    if best["score"] >= 1.0:
                        # the policy already gets this row right every time
                        stats["all-good"] += 1
                        continue
                    if worst["score"] > 0.0:
                        stats["flat"] += 1
                        continue
                    # every sample failed: the row is worth keeping, but the
                    # only better answer available is the written target
                    if not row.get("target"):
                        stats["all-bad-no-target"] += 1
                        continue
                    best = {"text": row["target"], "score": 1.0, "reason": "dataset target"}
                    pair_source = "gold-chosen"

                ref_c = ref_logprob(wav_path, best["text"])
                ref_r = ref_logprob(wav_path, worst["text"])
                if ref_c is None or ref_r is None:
                    stats["logprob"] += 1
                    continue
            finally:
                os.remove(wav_path)

            fout.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "slurp_id": row["slurp_id"],
                        "kind": row["kind"],
                        "mask": row["mask"],
                        "snr_db": row["snr_db"],
                        "sentence": row["sentence"],
                        "chosen": best["text"],
                        "rejected": worst["text"],
                        "chosen_score": best["score"],
                        "rejected_score": worst["score"],
                        "chosen_reason": best["reason"],
                        "rejected_reason": worst["reason"],
                        "ref_logp_chosen": ref_c,
                        "ref_logp_rejected": ref_r,
                        "pair_source": pair_source,
                        "n_sampled": len(uniq),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fout.flush()
            kept += 1
            stats[pair_source] += 1
            stats[f"{row['kind']}-pair"] += 1

    print(f"\nkept {kept}/{len(ds)} pairs -> {out_path}")
    print(f"breakdown: {dict(stats)}")
    if stats["gold-chosen"] > kept * 0.5:
        print(
            "warning: over half the pairs use the written target as `chosen`. "
            "That is off-policy, and a set dominated by it is closer to more "
            "SFT than to preference optimization."
        )


if __name__ == "__main__":
    main()
