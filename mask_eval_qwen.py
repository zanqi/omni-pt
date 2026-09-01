"""
Evaluate a Qwen Omni model on the mask track (steps/mask.html).

  C   = mean task-competence over the kind="answer" rows
  R   = mean conversational-repair over the kind="repair" rows
  EAR = 2*C*R / (C+R)

A third near-duplicate of babble_eval_qwen.py, deliberately: this track has
only two kinds, no probe columns and no reply parsing, so the type judge, the
score matrices and the heard-reply path are all gone. What is left is the
per-kind judge -- the row's own label picks the rubric and the rubric returns
the score, so no cell can hand a repair row 1.0 for a confident answer.

What it adds is the breakdown the four masks exist for: by_mask (silence /
white / splice / burst), by_bg (clean vs babble, split on snr_db) and by_snr in
5 dB bins. The summary line keeps `model`, `C`, `R` and `EAR` at the top level
and last in the file, so results/viz.ipynb reads these runs unchanged.

  python mask_eval_qwen.py --dataset keylazy/slurp-mask-v1 \
      --adapter-path checkpoints/Qwen2.5-Omni-3B-mask-sft \
      --judge-base-url http://g3061:8000/v1 --judge-model Qwen/Qwen3.8-27B
"""

import argparse
import json
import os
import tempfile
import threading
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

import soundfile as sf
import torch
from datasets import Audio, load_dataset
from openai import OpenAI

from prompts import ANSWER_JUDGE_SYSTEM, QWEN25_SYSTEM_PROMPT, REPAIR_JUDGE_SYSTEM, task_prompt
from util import detect_model_family, load_model

AUDIO_SAMPLING_RATE = 16000
KINDS = ("answer", "repair")
JUDGE_BY_KIND = {"answer": ANSWER_JUDGE_SYSTEM, "repair": REPAIR_JUDGE_SYSTEM}
METRIC_NAME = {"answer": "C", "repair": "R"}
VALID_SCORES = (0.0, 0.5, 1.0)
PARSE_FAIL_REASON = "Error parsing judge output"
SNR_BIN = 5.0

# GPU inference stays single-row and serialized behind this lock; worker
# threads overlap it with the (network-bound) judge call, so the judge latency
# for row N hides behind row N+1's generation
GPU_LOCK = threading.Lock()
ROW_WORKERS = 4


def get_audio(field):
    """Decoded Audio field -> (float32 array, sample rate)"""
    samples = field.get_all_samples()
    arr = samples.data  # (C, T)
    if arr.ndim > 1:
        arr = arr.mean(dim=0)
    return arr.numpy().astype("float32"), samples.sample_rate


@torch.inference_mode()
def run_model(model, processor, family, audio_array, sr, max_new_tokens, plain):
    """audio + task prompt -> the model's text reply"""
    from qwen_omni_utils import process_mm_info

    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(wav_path, audio_array, sr)

        conversation = []
        if family == "qwen2.5":
            conversation.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": QWEN25_SYSTEM_PROMPT}],
                }
            )
        conversation.append(
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": wav_path},
                    # must match what the adapter was trained under: scoring a
                    # restate-trained adapter under the plain prompt is a
                    # train/test mismatch that reads as a regression
                    {"type": "text", "text": task_prompt(False, plain)},
                ],
            }
        )

        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = processor(
            text=text, audio=audios, images=images, videos=videos, return_tensors="pt"
        )
        inputs = inputs.to(model.device).to(model.dtype)

        text_ids = model.generate(
            **inputs,
            return_audio=False,
            do_sample=False,
            thinker_max_new_tokens=max_new_tokens,
        )
        gen_ids = text_ids[:, inputs["input_ids"].shape[1] :]
        return processor.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
    finally:
        os.remove(wav_path)


def make_judge(judge_model, base_url="", max_tokens=4096):
    """-> _judge(system, user) -> (score in VALID_SCORES, reason).

    An unparseable reply scores 0 and says so in the reason, which the caller
    counts -- a judge that quietly fails looks exactly like a model that is bad
    at the task. Imported by mask_dpo_data.py so preference pairs are ranked by
    the same rubric the metric is computed from.
    """
    is_openai = base_url in (None, "", "openai")
    client = OpenAI() if is_openai else OpenAI(base_url=base_url, api_key="EMPTY")

    def _judge(system, user, retries=3):
        resp = None
        for attempt in range(retries + 1):
            try:
                kwargs = {}
                if not is_openai:
                    # vLLM reasoning models otherwise burn the whole token
                    # budget on <think> before reaching the JSON; the rubric
                    # already asks for "reason" before "score" inside it
                    kwargs["extra_body"] = {
                        "chat_template_kwargs": {"enable_thinking": False}
                    }
                resp = client.chat.completions.create(
                    model=judge_model,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    **kwargs,
                )
                break
            except Exception as e:
                if attempt == retries:
                    raise
                print(f"judge API error (attempt {attempt + 1}): {e}")
                time.sleep(2**attempt)

        msg = resp.choices[0].message
        text = (msg.content or "").strip()
        if not text:
            text = (getattr(msg, "reasoning_content", None) or "").strip()
        try:
            data = json.loads(text)
            score = float(data["score"])
            if score in VALID_SCORES:
                return score, data.get("reason")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
        return 0.0, PARSE_FAIL_REASON

    return _judge


def judge_user(row, reply):
    """The user half of a judge call. Shared with mask_dpo_data.py so a
    preference pair is scored on exactly the text the metric would see."""
    user = f'COMMAND: {row["sentence"]}\n'
    if row["kind"] == "repair":
        # the repair rubric needs the piece the row says was masked to decide
        # whether the question is targeted at it
        lost = row.get("lost") or []
        user += "LOST PIECE: " + ("; ".join(f'"{s}"' for s in lost) or "(none)") + "\n"
    return user + f"REPLY: {reply}\n"


def harmonic(*vals):
    """2*C*R/(C+R), and 0.0 if either is 0."""
    if any(v == 0 for v in vals):
        return 0.0
    return len(vals) / sum(1.0 / v for v in vals)


def imap_ordered(items, work, workers):
    """(item, work(item)) in submission order, `workers` in flight."""
    it = iter(items)
    pending = deque()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        try:
            while True:
                while len(pending) < workers:
                    item = next(it, None)
                    if item is None:
                        break
                    pending.append((item, ex.submit(work, item)))
                if not pending:
                    return
                item, fut = pending.popleft()
                yield item, fut.result()
        finally:
            for _, fut in pending:
                fut.cancel()


def breakdown(bucket_scores):
    """{bucket: {kind: [scores]}} -> {bucket: {C, R, EAR, n}}"""
    out = {}
    for name, by_kind in sorted(bucket_scores.items()):
        means = {}
        for kind in KINDS:
            vals = by_kind.get(kind, [])
            means[METRIC_NAME[kind]] = sum(vals) / len(vals) if vals else None
        n = sum(len(v) for v in by_kind.values())
        scored = [means[METRIC_NAME[k]] for k in KINDS if means[METRIC_NAME[k]] is not None]
        out[name] = {
            **means,
            "EAR": harmonic(*scored) if len(scored) == len(KINDS) else None,
            "n": n,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="keylazy/slurp-mask-v1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument(
        "--adapter-path",
        default=None,
        help="LoRA adapter, or a comma-separated stack merged left to right "
        "(e.g. <sft>,<dpo>) for an adapter trained on top of another.",
    )
    ap.add_argument(
        "--model-family",
        default=None,
        choices=["qwen2.5", "qwen3"],
        help="Override the family auto-detected from --model-path. Needed for "
        "checkpoint paths that carry neither family string.",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--tag",
        default="mask",
        help="Suffix of the default output name, so two runs of one model on "
        "different splits do not overwrite each other.",
    )
    ap.add_argument("--num-rows", type=int, default=-1)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--judge-model", default="gpt-4o")
    ap.add_argument(
        "--judge-base-url",
        default="openai",
        help="'openai' for the API, or a vLLM box's /v1 URL. Take the model "
        "name from that box's /v1/models rather than trusting the default: "
        "a mismatch is a 404 on every row.",
    )
    ap.add_argument("--judge-max-tokens", type=int, default=4096)
    ap.add_argument(
        "--plain-prompt",
        action="store_true",
        help="Generate under TASK_PROMPT instead of the restate prompt. Only "
        "for scoring an adapter trained back when sft_qwen.py still had "
        "--plain-prompt.",
    )
    args = ap.parse_args()

    family = args.model_family or detect_model_family(args.model_path)
    # a stack is named after its last adapter -- that is the run being scored
    last = (args.adapter_path or args.model_path).split(",")[-1]
    name = os.path.basename(last.rstrip("/"))
    out_path = args.out or os.path.join(
        "results", f"mask_results_{name}_{args.tag}.jsonl"
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split)
    ds = ds.cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE))
    if args.num_rows != -1:
        ds = ds.select(range(min(args.num_rows, len(ds))))
    print(
        f"eval {len(ds)} rows from {args.dataset}:{args.split} | model {name} "
        f"({family}) | prompt {'plain' if args.plain_prompt else 'restate'} | "
        f"judge {args.judge_model} @ {args.judge_base_url}"
    )

    model, processor = load_model(args.model_path, family, args.adapter_path)
    judge_fn = make_judge(
        args.judge_model, base_url=args.judge_base_url, max_tokens=args.judge_max_tokens
    )

    def process_row(row):
        arr, sr = get_audio(row["audio"])
        with GPU_LOCK:
            resp = run_model(
                model, processor, family, arr, sr, args.max_new_tokens, args.plain_prompt
            )
        score, reason = judge_fn(JUDGE_BY_KIND[row["kind"]], judge_user(row, resp))
        return {"resp": resp, "score": score, "reason": reason}

    scores = defaultdict(list)
    by_mask = defaultdict(lambda: defaultdict(list))
    by_bg = defaultdict(lambda: defaultdict(list))
    by_snr = defaultdict(lambda: defaultdict(list))
    hist = Counter()
    judge_failures = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for i, (row, result) in enumerate(imap_ordered(ds, process_row, ROW_WORKERS)):
            kind, score = row["kind"], result["score"]
            scores[kind].append(score)
            by_mask[row["mask"]][kind].append(score)
            snr = row["snr_db"]
            by_bg["clean" if snr is None else "babble"][kind].append(score)
            if snr is not None:
                lo = int(snr // SNR_BIN) * int(SNR_BIN)
                by_snr[f"{lo}-{lo + int(SNR_BIN)}"][kind].append(score)
            hist[f"{kind}@{score:g}"] += 1
            if result["reason"] == PARSE_FAIL_REASON:
                judge_failures += 1

            fout.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "slurp_id": row["slurp_id"],
                        "kind": kind,
                        "mask": row["mask"],
                        "snr_db": snr,
                        "mask_snr_db": row.get("mask_snr_db"),
                        "sentence": row["sentence"],
                        "lost": row.get("lost") or [],
                        "slot_type": row.get("slot_type", ""),
                        "bed_asr": row.get("bed_asr", ""),
                        "target": row.get("target", ""),
                        "response": result["resp"],
                        "score": score,
                        "reason": result["reason"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fout.flush()

            print(
                f"[{i+1}/{len(ds)}] id={row['id']} kind={kind} mask={row['mask']} "
                f"snr={snr} {METRIC_NAME[kind]}={score}"
            )
            print(f"    CMD : {row['sentence']}")
            print(f"    LOST: {'; '.join(row.get('lost') or []) or '(none)'}")
            print(f"    LLM : {result['resp']}")
            print(f"    JUD : {result['reason']}")

        if not any(scores.values()):
            print("No rows evaluated.")
            return

        means = {
            k: (sum(scores[k]) / len(scores[k]) if scores[k] else None) for k in KINDS
        }
        C, R = means["answer"], means["repair"]
        EAR = harmonic(*[v for v in (C, R) if v is not None]) if C and R else 0.0

        summary = {
            "type": "summary",
            "model": name,
            "model_path": args.model_path,
            "adapter_path": args.adapter_path,
            "dataset": args.dataset,
            "split": args.split,
            "judge_model": args.judge_model,
            "judge_base_url": args.judge_base_url,
            "plain_prompt": args.plain_prompt,
            "n": {k: len(scores[k]) for k in KINDS},
            "hist": dict(sorted(hist.items())),
            "judge_failures": judge_failures,
            "by_mask": breakdown(by_mask),
            "by_bg": breakdown(by_bg),
            "by_snr": breakdown(by_snr),
            # last, and named as viz.ipynb's load_summary expects
            "C": C,
            "R": R,
            "EAR": EAR,
        }
        fout.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"\n=== {name} on {args.dataset}:{args.split} ===")
    print(f"C (answer): {C if C is None else round(C, 3)}  n={len(scores['answer'])}")
    print(f"R (repair): {R if R is None else round(R, 3)}  n={len(scores['repair'])}")
    print(f"EAR:        {EAR:.3f}")
    for title, block in (
        ("by mask", summary["by_mask"]),
        ("by background", summary["by_bg"]),
        ("by snr", summary["by_snr"]),
    ):
        print(f"\n{title}:")
        for bucket, v in block.items():
            c = "  n/a" if v["C"] is None else f"{v['C']:.3f}"
            r = "  n/a" if v["R"] is None else f"{v['R']:.3f}"
            e = "  n/a" if v["EAR"] is None else f"{v['EAR']:.3f}"
            print(f"  {bucket:>10}  C={c}  R={r}  EAR={e}  n={v['n']}")
    if judge_failures:
        print(f"\njudge parse failures: {judge_failures} (scored 0)")
    print(f"\nper-row results + summary -> {out_path}")


if __name__ == "__main__":
    main()
