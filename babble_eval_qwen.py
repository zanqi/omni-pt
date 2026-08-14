"""
Evaluate a Qwen Omni model (Qwen2.5-Omni or Qwen3-Omni, base or fine-tuned
checkpoint) on the slurp EAR benchmark by feeding the raw audio directly to
the omni model.

  C   = mean task-competence over the answerable audio
  R   = mean conversational-repair over the unanswerable audio
  F   = mean full-repair score over the `repair_full` rows
  EAR = harmonic mean of the scores of the evaluated kinds
        (3 * C*R*F / (C*R + C*F + R*F) with all three;
         2 * C*R / (C+R) under --kinds answer,repair)

Judging is done by a local vLLM server (default) or the OpenAI API.


The model family (qwen2.5 vs qwen3) is auto-detected from --model-path.
For fine-tuned checkpoints whose path doesn't contain "qwen2.5"/"qwen3",
pass --model-family explicitly.
"""

import argparse
import json
import os
import tempfile
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import soundfile as sf
import torch
from datasets import load_dataset, Audio
from openai import OpenAI
from util import QWEN25_SYSTEM_PROMPT, detect_model_family, load_model
from prompts import (
    ANSWER_JUDGE_SYSTEM,
    REPAIR_JUDGE_SYSTEM,
    REPAIR_ON_TARGET_SYSTEM,
    REPEAT_JUDGE_SYSTEM,
    RESPONSE_TYPE_SYSTEM,
    split_heard_reply,
    task_prompt,
)

AUDIO_SAMPLING_RATE = 16000

# GPU inference stays single-row and serialized behind this lock; worker
# threads overlap that with the (network-bound) judge call, which needs no
# lock, so judge latency for row N hides behind GPU inference for row N+1.
GPU_LOCK = threading.Lock()
ROW_WORKERS = 4

JUDGED_TYPES = ("answer", "repair", "repair_off", "repeat", "bad")
# score = SCORE_MATRICES[--score-matrix][target kind][judged type]
SCORE_MATRICES = {
    "legacy": {
        "answer": {"answer": 1.0, "repair": 0.0, "repeat": 0.0, "bad": 0.0},
        "repair": {"answer": 1.0, "repair": 1.0, "repeat": 0.5, "bad": 0.0},
        "repeat": {"answer": 1.0, "repair": 0.5, "repeat": 1.0, "bad": 0.0},
    },
    "tree": {
        "answer": {
            "answer": 1.0,
            "repair": 0.0,
            "repair_off": 0.0,
            "repeat": 0.0,
            "bad": 0.0,
        },
        "repair": {
            "answer": 1.0,
            "repair": 1.0,
            "repair_off": 0.0,
            "repeat": 0.5,
            "bad": 0.0,
        },
        "repeat": {
            "answer": 1.0,
            "repair": 0.0,
            "repair_off": 0.0,
            "repeat": 1.0,
            "bad": 0.0,
        },
    },
}

# --judge-mode per-kind: the row's label picks the rubric and the judge returns
# the score itself, so there is no type to convert and no cell that can hand a
# repair row 1.0 for a confident answer.
JUDGE_BY_KIND = {
    "answer": ANSWER_JUDGE_SYSTEM,
    "repair": REPAIR_JUDGE_SYSTEM,
    "repeat": REPEAT_JUDGE_SYSTEM,
}
VALID_SCORES = (0.0, 0.5, 1.0)
PARSE_FAIL_REASON = "Error parsing judge output"


def get_audio(field):
    """Decoded Audio field -> (float32 array, sample rate)"""

    samples = field.get_all_samples()
    # arr: (C, T), C is num channels, 1 mono, 2 stereo
    arr = samples.data
    if arr.ndim > 1:
        # if stereo, average the 2 channels to get 1 array
        # (downmix to mono)
        arr = arr.mean(dim=0)
    return arr.numpy().astype("float32"), samples.sample_rate


@torch.inference_mode()
def run_model(
    model, processor, family, audio_array, sr, max_new_tokens, heard_reply, restate
):
    """input: audio + task prompt => return: model's text reply"""
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
                    # must match the prompt the adapter was trained under --
                    # evaluating an hr adapter under the plain prompt is a
                    # train/test mismatch that reads as a regression
                    {"type": "text", "text": task_prompt(heard_reply, restate)},
                ],
            }
        )

        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device).to(model.dtype)

        text_ids = model.generate(
            **inputs,
            return_audio=False,
            do_sample=False,
            thinker_max_new_tokens=max_new_tokens,
        )  # (B=1, T) tensor

        gen_ids = text_ids[:, inputs["input_ids"].shape[1] :]
        resp = processor.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return resp.strip()
    finally:
        os.remove(wav_path)


def make_judge(
    judge_model: str,
    base_url: str = "",
    max_tokens: int = 4096,
    per_kind: bool = False,
):
    """-> _judge(system, user) -> (judged type, reason).

    Under per_kind the rubric emits the score directly, so the first element is
    a float from VALID_SCORES instead of a type name. Either way an unparseable
    reply scores 0 and says so in the reason, which the caller counts.
    """
    is_openai = base_url in (None, "", "openai")
    if is_openai:
        client = OpenAI()
    else:
        client = OpenAI(base_url=base_url, api_key="EMPTY")

    def _judge(system: str, user: str, retries: int = 3):
        resp = None
        for attempt in range(retries + 1):
            try:
                kwargs = {}
                if not is_openai:
                    # vLLM reasoning models (e.g. Qwen3.5) otherwise burn the
                    # whole max_tokens budget on <think> before ever reaching
                    # the JSON payload; the rubric already asks for "reason"
                    # before "score" inside that JSON, so extra thinking is redundant.
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
            if per_kind:
                score = float(data["score"])
                if score in VALID_SCORES:
                    return score, data.get("reason")
            else:
                jtype = str(data.get("type", "bad")).strip().lower()
                if jtype in JUDGED_TYPES:
                    return jtype, data.get("reason")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
        return (0.0 if per_kind else "bad"), PARSE_FAIL_REASON

    return _judge


# ---
# Judge rubrics
# ---


def _fmt_lost(lost):
    if not lost:
        return "(none)"
    return "; ".join(f'"{s}"' for s in lost)


def harmonic(*vals):
    """Harmonic mean of the per-kind scores, 0.0 if any of them is 0.
    n=3 is 3*C*R*F/(C*R + C*F + R*F); n=2 is 2*C*R/(C+R)."""
    if any(v == 0 for v in vals):
        return 0.0
    return len(vals) / sum(1.0 / v for v in vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="keylazy/slurp-babble-Qwen2.5-Omni-3B-v3")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--model-family",
        default="auto",
        choices=["auto", "qwen2.5", "qwen3"],
        help="Which Qwen Omni family to load. 'auto' infers from --model-path.",
    )
    ap.add_argument(
        "--judge-model",
        default="Qwen/Qwen3.5-122B-A10B-FP8",
        help="from vllm --served-model-name; from openai, gpt-4o",
    )
    ap.add_argument(
        "--judge-base-url",
        default="http://g3085:8000/v1",
        help="'http://g3085:8000/v1' for vllm, 'openai' to use openai",
    )
    ap.add_argument(
        "--judge-max-tokens",
        type=int,
        default=4096,
    )
    ap.add_argument("--num-rows", type=int, default=150)
    ap.add_argument(
        "--kinds",
        default="answer,repair,repeat",
        help="Which row kinds to evaluate, comma-separated. Dropping a kind "
        "drops its judge call and its factor in EAR, so 'answer,repair' scores "
        "C and R only (EAR = 2*C*R/(C+R)) on a dataset built with all three. "
        "Filtering happens before --num-rows, so the row budget is spent "
        "entirely on the kinds asked for.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument(
        "--heard-reply",
        action="store_true",
        help="Prompt with TASK_PROMPT_HR and judge only the parsed 'Reply:' "
        "half. Required for adapters trained with sft_qwen.py --heard-reply.",
    )
    ap.add_argument(
        "--restate-prompt",
        action="store_true",
        help="Prompt with TASK_PROMPT_TREE, which asks the model to restate "
        "every piece of the request it caught. Required for adapters trained "
        "with sft_qwen.py --restate-prompt.",
    )
    ap.add_argument(
        "--score-matrix",
        default="legacy",
        choices=list(SCORE_MATRICES),
        help="How a judged type scores against the row's target kind. Only "
        "the repeat-row/repair-judgment cell differs (0.5 -> 0.0), so this is "
        "independent of how the dataset was labeled: rescoring an old run "
        "under 'tree' measures the cell change on its own.",
    )
    ap.add_argument(
        "--judge-mode",
        default="type",
        choices=["type", "per-kind"],
        help="'type' classifies the reply into one of four types and converts "
        "it with --score-matrix. 'per-kind' picks a rubric from the row's own "
        "label and has it score directly -- the repair rubric is told which "
        "piece was lost, and a confident answer on a repair row scores 0 "
        "instead of 1.0, so numbers are NOT comparable with 'type' runs. "
        "Independent of the data track, so it can be ablated on any dataset.",
    )
    args = ap.parse_args()

    # if not os.environ.get("OPENAI_API_KEY"):
    #     raise SystemExit("OPENAI_API_KEY is not set.")

    family = (
        detect_model_family(args.model_path)
        if args.model_family == "auto"
        else args.model_family
    )

    name_src = args.adapter_path or args.model_path
    model_name = name_src.rstrip("/").split("/")[-1]
    # the default name is keyed only on the model/adapter, so fold the track in
    # -- otherwise an hr run silently overwrites the baseline result file
    # eval.ipynb reads
    per_kind = args.judge_mode == "per-kind"
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    unknown = [k for k in kinds if k not in JUDGE_BY_KIND]
    if unknown:
        raise SystemExit(f"--kinds: unknown kind(s) {unknown}")
    tag = "hr" if args.heard_reply else "beam" if per_kind else "v2"
    if kinds != ["answer", "repair", "repeat"]:
        tag = "-".join(kinds)
    out_path = args.out or f"results/bab_results_{model_name}_{tag}.jsonl"
    judge_system = RESPONSE_TYPE_SYSTEM
    score_matrix = SCORE_MATRICES[args.score_matrix]
    print(
        f"prompt: {'heard-reply' if args.heard_reply else 'restate' if args.restate_prompt else 'plain'} | "
        f"judge: {'per-kind' if per_kind else 'rules'} | "
        f"scores: {'direct' if per_kind else args.score_matrix} | "
        f"kinds: {','.join(kinds)}"
    )

    ds = load_dataset(args.dataset, split=args.split)

    ds = ds.cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE))
    if len(kinds) < 3:
        # before the --num-rows slice, so the budget buys only wanted kinds
        keep = [i for i, k in enumerate(ds["kind"]) if k in kinds]
        print(f"kind filter: {len(ds)} -> {len(keep)} rows")
        ds = ds.select(keep)
    if args.num_rows != -1:
        ds = ds.select(range(min(args.num_rows, len(ds))))

    print(f"Eval {len(ds)} rows from {args.dataset}:{args.split}")

    model, processor = load_model(args.model_path, family, args.adapter_path)
    judge_fn = make_judge(
        args.judge_model,
        base_url=args.judge_base_url,
        max_tokens=args.judge_max_tokens,
        per_kind=per_kind,
    )

    scores = {"answer": 0.0, "repair": 0.0, "repeat": 0.0}
    counts = {"answer": 0, "repair": 0, "repeat": 0}
    confusion = Counter()  # (target kind, judge type) -> n; type mode only
    hist = Counter()  # "<kind>@<score>" -> n; per-kind mode only
    metric_name = {"answer": "C", "repair": "R", "repeat": "F"}

    def imap_ordered(items, work, workers):
        """Run `work` over `items` on a thread pool, yielding (item, result)
        pairs in submission order even though work finishes out of order --
        so row N+1's GPU inference can run while row N's judge call is
        in flight, without reordering the output stream."""
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

    def process_row(row):
        kind = row["kind"]
        if kind not in score_matrix:
            raise ValueError(f"unknown kind in dataset: {kind!r}")
        arr, sr = get_audio(row["audio"])
        with GPU_LOCK:
            resp = run_model(
                model,
                processor,
                family,
                arr,
                sr,
                args.max_new_tokens,
                args.heard_reply,
                args.restate_prompt,
            )

        heard, reply = ("", "")
        if args.heard_reply:
            heard, reply = split_heard_reply(resp)
        # On the plain track, judge the whole resp
        # On the heard-reply track, judge the reply line
        judged_reply = reply if args.heard_reply else resp

        if per_kind:
            # the row's label is the ground truth, so the repair rubric gets
            # the piece it says was lost -- the type classifier had no slot for
            # it. The other two kinds carry no such information: on an answer
            # row nothing was lost, and on a repeat row essentially everything
            # was.
            user = f'COMMAND: {row["sentence"]}\n'
            if kind == "repair":
                # per-kind judge on repair need the lost
                # piece to decide if the repair question is
                # targeted
                user += f"LOST PIECE: {_fmt_lost(row['lost'])}\n"
            user += f"REPLY: {judged_reply}\n"
            score, reason = judge_fn(JUDGE_BY_KIND[kind], user)
            judged_type = ""
        else:
            # non-per-kind judge, judge see the command and
            # resp and classify the resp
            user = f'COMMAND: {row["sentence"]}\nREPLY: {judged_reply}\n'
            judged_type, reason = judge_fn(judge_system, user)
            # second stage: a repair question is only worth crediting if it is targeted
            cell = score_matrix[kind]
            if (
                judged_type == "repair"
                and row["lost"]
                and cell.get("repair_off", cell["repair"]) < cell["repair"]
            ):
                judged_type, reason = judge_fn(
                    REPAIR_ON_TARGET_SYSTEM,
                    f'COMMAND: {row["sentence"]}\n'
                    f"LOST PIECE: {_fmt_lost(row['lost'])}\n"
                    f"REPLY: {judged_reply}\n",
                )
            score = score_matrix[kind][judged_type]

        return {
            "resp": resp,
            "heard": heard,
            "reply": reply,
            "judged_type": judged_type,
            "reason": reason,
            "score": score,
        }

    parse_failures = 0
    judge_failures = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for i, (row, result) in enumerate(imap_ordered(ds, process_row, ROW_WORKERS)):
            kind = row["kind"]
            sentence = row["sentence"]
            # a dataset built without the probe columns still evaluates
            asr_transcript = row.get("asr_transcript", "")
            lost = row["lost"]

            resp = result["resp"]
            judged_type = result["judged_type"]
            reason = result["reason"]
            score = result["score"]

            scores[kind] += score
            counts[kind] += 1
            if per_kind:
                hist[f"{kind}@{score:g}"] += 1
            else:
                confusion[(kind, judged_type)] += 1
            if reason == PARSE_FAIL_REASON:
                judge_failures += 1
            if args.heard_reply and not result["heard"]:
                parse_failures += 1

            rec = {
                "id": row["id"],
                "slurp_id": row["slurp_id"],
                "kind": kind,
                "sentence": sentence,
                "snr_db": row["snr_db"],
                "asr_transcript": asr_transcript,
                "base_response": row.get("omni_response", ""),
                "lost": lost,
                "target": row["target"],
                "response": resp,
                "heard": result["heard"],
                "reply": result["reply"],
                "judged_type": judged_type,
                "score": score,
                "reason": reason,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            print(
                f"[{i+1}/{len(ds)}] id={row['id']} slurp_id={row['slurp_id']} "
                f"kind={kind} snr={row['snr_db']} "
                + ("" if per_kind else f"judged={judged_type} ")
                + f"{metric_name[kind]}={score}"
            )
            print(f"    CMD : {sentence}")
            print(f"    ASR : {asr_transcript}")
            print(f"    LOST: {_fmt_lost(lost)}")
            if args.heard_reply:
                print(f"    HRD : {result['heard'] or '(unparseable — judged raw)'}")
                print(f"    RPLY: {result['reply']}")
            else:
                print(f"    LLM : {resp}")
            print(f"    JUD : {reason}")

        if sum(counts.values()) == 0:
            print("No instances evaluated.")
            return

        C = scores["answer"] / counts["answer"] if counts["answer"] else 0.0
        R = scores["repair"] / counts["repair"] if counts["repair"] else 0.0
        F = scores["repeat"] / counts["repeat"] if counts["repeat"] else 0.0
        # a kind that wasn't evaluated is absent from EAR, and reported as null
        # rather than 0.0 so nothing downstream reads it as a failed dimension
        means = {"answer": C, "repair": R, "repeat": F}
        EAR = harmonic(*(means[k] for k in kinds))

        fout.write(
            json.dumps(
                {
                    "type": "summary",
                    "model": args.model_path,
                    "adapter": args.adapter_path,
                    "model_family": family,
                    "judge_model": args.judge_model,
                    "heard_reply": args.heard_reply,
                    "restate_prompt": args.restate_prompt,
                    "judge_mode": args.judge_mode,
                    "score_matrix": args.score_matrix,
                    "kinds": kinds,
                    "heard_parse_failures": parse_failures,
                    "judge_parse_failures": judge_failures,
                    "answer_rows": counts["answer"],
                    "repair_rows": counts["repair"],
                    "repeat_rows": counts["repeat"],
                    "C": C if "answer" in kinds else None,
                    "R": R if "repair" in kinds else None,
                    "F": F if "repeat" in kinds else None,
                    "EAR": EAR,
                    # per-kind mode has no types left to confuse, so it reports
                    # a score histogram instead -- but the key stays present
                    # and empty, since results/viz.ipynb indexes it directly
                    "confusion": {
                        f"{k}->{t}": n for (k, t), n in sorted(confusion.items())
                    },
                    "score_hist": dict(sorted(hist.items())),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        fout.flush()

    print("\n======")
    model_desc = args.model_path + (
        f" + {args.adapter_path}" if args.adapter_path else ""
    )
    print(
        f"Final Eval - {model_desc} "
        f"({' / '.join(f'{counts[k]} {k}' for k in kinds)} rows)"
    )
    for k in kinds:
        print(f"{metric_name[k]}  : {means[k]: .3f}")
    print(f"EAR: {EAR: .3f}")
    if args.heard_reply:
        print(
            f"heard-parse failures: {parse_failures}/{sum(counts.values())} "
            "(judged on raw output)"
        )
    if judge_failures:
        print(
            f"judge parse failures: {judge_failures}/{sum(counts.values())} (scored 0)"
        )
    if per_kind:
        print("\nscores per kind:")
        for k in kinds:
            cells = " ".join(f"{s:g}:{hist[f'{k}@{s:g}']:3d}" for s in VALID_SCORES)
            print(f"  {k:8s} {cells}")
    else:
        print("\nconfusion (target kind -> judged type):")
        for k in kinds:
            cells = " ".join(f"{t}:{confusion[(k, t)]:3d}" for t in JUDGED_TYPES)
            print(f"  {k:8s} {cells}")
    print("======")
    print(f"Per-sample results + summary written to {out_path}")


if __name__ == "__main__":
    main()
