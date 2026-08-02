"""
Evaluate a Qwen Omni model (Qwen2.5-Omni or Qwen3-Omni, base or fine-tuned
checkpoint) on the slurp EAR benchmark by feeding the raw audio directly to
the omni model.

  C   = mean task-competence over the answerable audio
  R   = mean conversational-repair over the unanswerable audio
  F   = mean full-repair score over the `repair_full` rows
  EAR = 3 * C * R * F / (C*R + C*F + R*F)

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
    RESPONSE_TYPE_FEWSHOT_SYSTEM,
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

JUDGED_TYPES = ("answer", "repair", "repeat", "bad")
# score = SCORE_MATRICES[--score-matrix][target kind][judged type]
SCORE_MATRICES = {
    "legacy": {
        "answer": {"answer": 1.0, "repair": 0.0, "repeat": 0.0, "bad": 0.0},
        "repair": {"answer": 1.0, "repair": 1.0, "repeat": 0.5, "bad": 0.0},
        "repeat": {"answer": 1.0, "repair": 0.5, "repeat": 1.0, "bad": 0.0},
    },
    # asking one targeted question when the whole command was lost earns
    # nothing: there was no reliably-heard piece to anchor it on. Only cell
    # that differs, so a legacy dataset rescored under "tree" isolates it.
    "tree": {
        "answer": {"answer": 1.0, "repair": 0.0, "repeat": 0.0, "bad": 0.0},
        "repair": {"answer": 1.0, "repair": 1.0, "repeat": 0.5, "bad": 0.0},
        "repeat": {"answer": 1.0, "repair": 0.0, "repeat": 1.0, "bad": 0.0},
    },
}


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
def run_model(model, processor, family, audio_array, sr, max_new_tokens, heard_reply):
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
                    {"type": "text", "text": task_prompt(heard_reply)},
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
            max_new_tokens=max_new_tokens,
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
):
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
                    kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
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
            jtype = str(data.get("type", "bad")).strip().lower()
            if jtype in JUDGED_TYPES:
                return jtype, data.get("reason")
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        return "bad", "Error parsing judge output"

    return _judge


# ---
# Judge rubrics
# ---

def _fmt_lost(lost):
    if not lost:
        return "(none)"
    return "; ".join(f'"{s}"' for s in lost)


def harmonic3(c, r, f):
    denom = c * r + c * f + r * f
    if denom == 0:
        return 0.0
    return 3.0 * c * r * f / denom


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
        help="from vllm --served-model-name; from openai, gpt-4o"
        )
    ap.add_argument(
        "--judge-base-url",
        default="http://g3085:8000/v1",
        help="'http://g3085:8000/v1' for vllm, 'openai' to use openai"
    )
    ap.add_argument(
        "--judge-max-tokens",
        type=int,
        default=4096,
    )
    ap.add_argument("--num-rows", type=int, default=150)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument(
        "--heard-reply",
        action="store_true",
        help="Prompt with TASK_PROMPT_HR and judge only the parsed 'Reply:' "
        "half. Required for adapters trained with sft_qwen.py --heard-reply.",
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
        "--fewshot-judge",
        action="store_true",
        help="Judge with RESPONSE_TYPE_FEWSHOT_SYSTEM instead of the rule "
        "rubric. Independent of --heard-reply so the judge swap can be "
        "ablated on its own.",
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
    tag = "hr" if args.heard_reply else "v2"
    out_path = args.out or f"results/bab_results_{model_name}_{tag}.jsonl"
    judge_system = (
        RESPONSE_TYPE_FEWSHOT_SYSTEM if args.fewshot_judge else RESPONSE_TYPE_SYSTEM
    )
    score_matrix = SCORE_MATRICES[args.score_matrix]
    print(
        f"prompt: {'heard-reply' if args.heard_reply else 'plain'} | "
        f"judge: {'few-shot' if args.fewshot_judge else 'rules'} | "
        f"scores: {args.score_matrix}"
    )

    ds = load_dataset(args.dataset, split=args.split)

    ds = ds.cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE))
    if args.num_rows != -1:
        ds = ds.select(range(min(args.num_rows, len(ds))))

    print(f"Eval {len(ds)} rows from {args.dataset}:{args.split}")

    model, processor = load_model(args.model_path, family, args.adapter_path)
    judge_fn = make_judge(
        args.judge_model,
        base_url=args.judge_base_url,
        max_tokens=args.judge_max_tokens,
    )

    scores = {"answer": 0.0, "repair": 0.0, "repeat": 0.0}
    counts = {"answer": 0, "repair": 0, "repeat": 0}
    confusion = Counter() # (target kind, judge type) -> n
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
                model, processor, family, arr, sr, args.max_new_tokens, args.heard_reply
            )

        # The judge sees the reply alone -- never the Heard line, which would
        # leak evidence into the type decision. A model that ignores the
        # two-line contract is scored on its raw output rather than penalized
        # twice; the summary counts how often that happened.
        heard, reply = ("", resp)
        if args.heard_reply:
            heard, reply = split_heard_reply(resp)

        user = (
            f'Original Spoken Command: "{row["sentence"]}"\n'
            f'Model Response: "{reply}"\n'
        )
        judged_type, reason = judge_fn(judge_system, user)
        return {
            "resp": resp,
            "heard": heard,
            "reply": reply,
            "judged_type": judged_type,
            "reason": reason,
            "score": score_matrix[kind][judged_type],
        }

    parse_failures = 0
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
            confusion[(kind, judged_type)] += 1
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
                f"kind={kind} snr={row['snr_db']} judged={judged_type} "
                f"{metric_name[kind]}={score}"
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
        EAR = harmonic3(C, R, F)

        fout.write(
            json.dumps(
                {
                    "type": "summary",
                    "model": args.model_path,
                    "adapter": args.adapter_path,
                    "model_family": family,
                    "judge_model": args.judge_model,
                    "heard_reply": args.heard_reply,
                    "fewshot_judge": args.fewshot_judge,
                    "score_matrix": args.score_matrix,
                    "heard_parse_failures": parse_failures,
                    "answer_rows": counts["answer"],
                    "repair_rows": counts["repair"],
                    "repeat_rows": counts["repeat"],
                    "C": C,
                    "R": R,
                    "F": F,
                    "EAR": EAR,
                    "confusion": {
                        f"{k}->{t}": n for (k, t), n in sorted(confusion.items())
                    },
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
        f"({counts['answer']} answer / {counts['repair']} repair / "
        f"{counts['repeat']} repeat rows)"
    )
    print(f"C  : {C: .3f}")
    print(f"R  : {R: .3f}")
    print(f"F  : {F: .3f}")
    print(f"EAR: {EAR: .3f}")
    if args.heard_reply:
        print(
            f"heard-parse failures: {parse_failures}/{sum(counts.values())} "
            "(judged on raw output)"
        )
    print("\nconfusion (target kind -> judged type):")
    for k in ("answer", "repair", "repeat"):
        cells = " ".join(f"{t}:{confusion[(k, t)]:3d}" for t in JUDGED_TYPES)
        print(f"  {k:8s} {cells}")
    print("======")
    print(f"Per-sample results + summary written to {out_path}")


if __name__ == "__main__":
    main()