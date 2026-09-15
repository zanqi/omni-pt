"""
Build DPO preference pairs for the mask track (steps/mask.html, step 9).

On-policy: every candidate is sampled from the SFT checkpoint itself, under the
exact prompt mask_eval_qwen.py generates with, and ranked by the exact judge it
scores with. The preference signal and the reported metric therefore agree by
construction -- a pair that teaches the model something is a pair that moves
C or R.

  1. pass 1 samples K responses per train row at temperature 1.0 and scores
     all K with the per-kind rubric (ANSWER_/REPAIR_/REPEAT_JUDGE_SYSTEM).
     Nothing is dropped here: a row the policy never fails on still holds the
     reply a sibling row needs, and that is only known once every row is in.
  2. pass 2 pairs them. chosen = best, rejected = the HARDEST losing sample --
     the best one still a full rubric step (MIN_MARGIN) below `chosen`, so a
     repair pair is 1-vs-0.5 rather than 1-vs-0 wherever the policy produced
     both
  3. a row whose every sample scores 1.0 borrows instead of dropping: the
     negative is the policy's own best reply to a SIBLING row -- same sentence
     and slurp_id, a different kind -- which is right there and wrong here (see
     BORROW_ORDER). Only the text crosses over; it is re-judged under this
     row's rubric, so `rejected_score` is real and MIN_MARGIN is enforced the
     same way it is on a sampled pair. Every row then makes a pair and the
     three kinds come out equal by construction, which is why dpo_qwen.py no
     longer weights the loss by kind (steps/dpo-avoid-weight.html).
  4. rows where every sample tied below 1.0 fall back to the dataset's written
     target as `chosen`, counted separately because a set dominated by those is
     really just more SFT
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
import random
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import soundfile as sf
import torch
from datasets import Audio, load_dataset
from qwen_omni_utils import process_mm_info
from tqdm import tqdm

from mask_eval_qwen import JUDGE_BY_KIND, get_audio, judge_user, make_judge
from prompts import QWEN25_SYSTEM_PROMPT, get_task_prompt
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
# which sibling kind a borrowed negative is taken from, primary first. Each
# sibling's correct behaviour is this row's failure: acting confidently on a
# command with a hole in it (repair borrows from answer), asking when nothing
# critical was missing (answer borrows from repeat), asking one targeted
# question when nothing in the utterance can be trusted (repeat borrows from
# repair).
BORROW_ORDER = {
    "answer": ("repeat", "repair"),
    "repair": ("answer", "repeat"),
    "repeat": ("repair", "answer"),
}
# how often the primary sibling wins; the secondary keeps the negatives of one
# kind from collapsing into a single register. Was REPEAT_SHARE, which said the
# same thing for `answer` rows alone.
PRIMARY_SHARE = 0.7
# what the borrowed text may cost: the previous recipe capped minted negatives
# at 25% of the all-good `answer` rows because every one of them was off-policy
# dataset text, and the -hf run separated those to a 3 nats/token margin while
# repair@1 moved by 3 rows out of 400. A sibling's SAMPLE is text this policy
# actually emits, so there is nothing left to cap -- the written target is only
# the fallback for a sibling that produced nothing usable.


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


def conversation(wav_path: str, task_prompt: str, sys_prompt=None, resp=None):
    """concat sys prompt, audio, task prompt, and assistant resp into an openai conv array"""

    conv = []
    if sys_prompt is not None:
        conv.append(
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]}
        )

    conv.append(
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": wav_path},
                {"type": "text", "text": task_prompt},
            ],
        }
    )

    if resp is not None:
        conv.append({"role": "assistant", "content": [{"type": "text", "text": resp}]})
    return conv


def pre_process(processor, conv, add_generation_prompt: bool):
    text = processor.apply_chat_template(
        conv, add_generation_prompt=add_generation_prompt, tokenize=False
    )
    audios, images, videos, *_ = process_mm_info(conv, use_audio_in_video=False)
    return processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
    )


@torch.inference_mode()
def ref_logprob(model, processor, wav_path, task_prompt, resp, sys_prompt):
    """log P(resp | audio, prompts) under the SFT model.

    The label mask is built the way OmniSFTCollator builds it: render the
    conversation with and without the assistant turn and diff the token
    counts, Only consider text after '<|im_start|>assistant\\n'.
    """

    # full text is sys+audio+resp
    full_conv = conversation(wav_path, task_prompt, sys_prompt, resp)
    # prompt text is sys+audio+"<|im_start|>assistant\\n"
    prefix_conv = conversation(wav_path, task_prompt, sys_prompt)

    # TODO: refactor it to util and make OmniSFTCollator use it too
    full = pre_process(processor, full_conv, add_generation_prompt=False)
    prefix = pre_process(processor, prefix_conv, add_generation_prompt=True)
    ans_len = int(full["attention_mask"].sum() - prefix["attention_mask"].sum())
    if ans_len <= 0:
        return None
    full = full.to(model.device).to(model.dtype)
    labels = torch.full_like(full["input_ids"], -100)
    labels[:, -ans_len:] = full["input_ids"][:, -ans_len:]
    logits = model.thinker(**full).logits

    return float(seq_logprobs(logits, labels)[0])


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
    sys_prompt = QWEN25_SYSTEM_PROMPT if family == "qwen2.5" else None
    task_prompt = get_task_prompt(False, args.plain_prompt)

    @torch.inference_mode()
    def sample_k(wav_path, k):
        """sample K replies in one generate call -- the audio encoder runs once."""

        conv = conversation(wav_path, task_prompt, sys_prompt)
        inputs = (
            pre_process(processor, conv, add_generation_prompt=True)
            .to(model.device)
            .to(model.dtype)
        )
        ids = model.generate(
            **inputs,
            return_audio=False,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            num_return_sequences=k,
            thinker_max_new_tokens=args.max_new_tokens,
        )
        gen = ids[:, inputs["input_ids"].shape[1] :]

        return [
            txt.strip()
            for txt in processor.batch_decode(
                gen, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        ]

    def row_wav(row):
        """row audio -> a temp wav path the omni processor can read; the caller
        removes it. Pass 2 re-decodes rather than keeping every wav from pass 1
        alive, which would be ~3 GB on disk for the length of the run."""

        arr, sr = get_audio(row["audio"])
        arr = arr[: MAX_AUDIO_SECONDS * AUDIO_SAMPLING_RATE]
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(wav_path, arr, sr)
        return wav_path

    # === pass 1: sample and judge every row ===
    #
    # Nothing is dropped for being easy here. A row the policy never fails on
    # makes no pair of its own, but its best reply is exactly the negative a
    # sibling row needs, and which rows those are is not known until the whole
    # split has been sampled.
    scored_by_row, stats = {}, Counter()
    for row in tqdm(ds, desc="sample", unit="row", dynamic_ncols=True):
        wav_path = row_wav(row)
        try:
            samples = sample_k(wav_path, args.samples)
        finally:
            os.remove(wav_path)

        # dedupe before judging: identical samples cost a judge call each and
        # can never form a pair with each other
        uniq = list(dict.fromkeys(s for s in samples if s))
        if not uniq:
            stats["empty"] += 1
            continue

        with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as ex:
            judged = list(
                ex.map(
                    lambda s: judge_fn(JUDGE_BY_KIND[row["kind"]], judge_user(row, s)),
                    uniq,
                )
            )

        # score only, and a stable sort, so ties keep sampling order. Breaking
        # them on length made `chosen` the shortest sample of the top score and
        # `rejected` the longest of the bottom one, which teaches brevity as
        # much as it teaches repair.
        scored_by_row[row["id"]] = sorted(
            ({"text": s, "score": j[0], "reason": j[1]} for s, j in zip(uniq, judged)),
            key=lambda d: d["score"],
        )

    # === pass 2: pair each row, borrowing where the policy never failed ===
    #
    # slurp_id -> kind -> row id, plus the written target behind each id: a
    # borrow falls back to the sibling's dataset text when the sibling itself
    # sampled empty.
    sibling, target_of = defaultdict(dict), {}
    for r in ds.select_columns(["id", "slurp_id", "kind", "target"]):
        sibling[r["slurp_id"]][r["kind"]] = r["id"]
        target_of[r["id"]] = r["target"]
    rng = random.Random(0)

    def borrow_negative(row, chosen_text):
        """(text, pair_source) for a row the policy gets right every time.

        Only the text crosses over. The caller re-judges it under THIS row's
        rubric, so a borrow that happens to work here too is thrown out rather
        than written as a 1-vs-1 pair, and one that lands on 0.5 -- a repair
        question read under the repeat rubric, say -- is kept as the hard
        negative it is.
        """

        sibs = sibling[row["slurp_id"]]
        order = BORROW_ORDER[row["kind"]]
        if rng.random() >= PRIMARY_SHARE:
            order = order[::-1]

        # the sibling's own best sample first: on-policy text, and the right
        # reply on its own row, so the pair is "right behaviour, wrong row"
        # rather than "good reply vs bad reply"
        for kind in order:
            cand = scored_by_row.get(sibs.get(kind))
            if cand and cand[-1]["text"] != chosen_text:
                return cand[-1]["text"], f"borrow-{kind}"
        for kind in order:
            text = target_of.get(sibs.get(kind))
            if text and text != chosen_text:
                return text, "borrow-target"
        return None, None

    kept = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for row in tqdm(ds, desc="pairs", unit="row", dynamic_ncols=True):
            scored = scored_by_row.get(row["id"])
            if not scored:
                continue  # sampled empty in pass 1, already counted there

            best = scored[-1]
            # the hardest negative, not the worst sample. On the rubric's
            # {0, 0.5, 1} steps `scored[0]` made most repair pairs 1-vs-0,
            # a distinction the SFT policy already makes; the pair that is
            # still open is 1-vs-0.5 -- asks for clarification, but not
            # about the piece that went missing -- which is where 37% of
            # the eval's repair rows sit. One rubric step down is still
            # MIN_MARGIN, so the filter is what picks the side.
            losing = [s for s in scored if best["score"] - s["score"] >= MIN_MARGIN]
            worst = losing[-1] if losing else scored[0]

            pair_source = "sampled"
            if best["score"] - worst["score"] < MIN_MARGIN:
                # this means best == worst when min margin is 0.5

                if worst["score"] >= 1.0:
                    # worst == best == 1.0: the policy already gets this audio
                    # right every time, so the negative has to come from a
                    # sibling row rather than from this one
                    text, pair_source = borrow_negative(row, best["text"])
                    if text is None:
                        stats["no-sibling"] += 1
                        continue

                    score, reason = judge_fn(
                        JUDGE_BY_KIND[row["kind"]], judge_user(row, text)
                    )
                    if best["score"] - score < MIN_MARGIN:
                        # the borrowed reply serves this row too -- not a
                        # negative, whatever it was on its own row
                        stats["borrow-too-good"] += 1
                        continue
                    worst = {"text": text, "score": score, "reason": reason}

                elif not row.get("target"):
                    stats["all-bad-no-target"] += 1
                    continue
                else:
                    # every sample landed on the same sub-perfect score:
                    # all-0 (nothing usable was sampled) or all-0.5 (the
                    # policy always asks and never on target -- again the
                    # repair@0.5 bucket the eval is stuck in, which used
                    # to be dropped as "flat"). Either way the written
                    # target is the only better reply on hand.
                    best = {
                        "text": row["target"],
                        "score": 1.0,
                        "reason": "dataset target",
                    }
                    pair_source = "gold-chosen"

            # the audio is only needed from here: both sides are settled, and a
            # row dropped above never pays for a decode
            wav_path = row_wav(row)
            try:
                ref_c = ref_logprob(
                    model, processor, wav_path, task_prompt, best["text"], sys_prompt
                )
                ref_r = ref_logprob(
                    model, processor, wav_path, task_prompt, worst["text"], sys_prompt
                )
            finally:
                os.remove(wav_path)

            if ref_c is None or ref_r is None:
                stats["logprob"] += 1
                continue

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
                        "n_sampled": len(scored),
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
    # the two numbers that say what kind of run this was. dpo_qwen.py no longer
    # weights by kind, so a set that is not thirds is not balanced by anything
    # downstream -- and the borrowed share is the recipe itself: near zero means
    # the sibling index never joined (a slurp_id left with one kind after the
    # build's skips), near 100% means the policy failed nowhere on its own.
    if kept:
        borrowed = sum(v for k, v in stats.items() if k.startswith("borrow-"))
        by_kind = {k: stats[f"{k}-pair"] for k in BORROW_ORDER}
        print(
            f"borrowed: {borrowed}/{kept} ({borrowed / kept:.0%}) of pairs take "
            "their negative from a sibling row the policy never fails on"
        )
        print(f"kinds: {by_kind}")
        if max(by_kind.values()) - min(by_kind.values()) > 0.1 * kept / 3:
            print(
                "warning: the kinds are more than 10% apart. DPO weights them "
                "equally per pair, so an imbalance here is an imbalance in the "
                "loss -- check the no-sibling / borrow-too-good / empty counts."
            )
    if stats["gold-chosen"] > kept * 0.5:
        print(
            "warning: over half the pairs use the written target as `chosen`. "
            "That is off-policy, and a set dominated by it is closer to more "
            "SFT than to preference optimization."
        )


if __name__ == "__main__":
    main()
