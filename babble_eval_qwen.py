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
import re
import tempfile
import time
from collections import Counter
import soundfile as sf
import torch
from datasets import load_dataset, Audio
from openai import OpenAI

from util import QWEN25_SYSTEM_PROMPT, detect_model_family, load_model

AUDIO_SAMPLING_RATE = 16000

TASK_PROMPT = (
    "You are a smart voice device with full access to the user's apps, "
    "accounts, devices, information, and the internet. Listen to the user's spoken request "
    "and respond naturally and concisely, addressing everything it asks."
)

JUDGED_TYPES = ("answer", "repair", "repeat", "bad")
# Tree-rule score matrix: score = SCORE_MATRIX[target kind][judged type]
SCORE_MATRIX = {
    "answer": {"answer": 1.0, "repair": 0.0, "repeat": 0.0, "bad": 0.0},
    "repair": {"answer": 1.0, "repair": 1.0, "repeat": 0.5, "bad": 0.0},
    "repeat": {"answer": 1.0, "repair": 0.5, "repeat": 1.0, "bad": 0.0},
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


def build_conversation(wav_path: str, family: str):
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
                {"type": "text", "text": TASK_PROMPT},
            ],
        }
    )
    return conversation


@torch.inference_mode()
def run_model(model, processor, family, audio_array, sr, max_new_tokens):
    """input: audio + task prompt => return: model's text reply"""
    from qwen_omni_utils import process_mm_info

    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(wav_path, audio_array, sr)

        conversation = build_conversation(wav_path, family)

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
    if base_url in (None, "", "openai"):
        client = OpenAI()
    else:
        client = OpenAI(base_url=base_url, api_key="EMPTY")

    def _judge(system: str, user: str, retries: int = 3):
        resp = None
        for attempt in range(retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=judge_model,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
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
            text = (getattr(msg, "reasoning", None) or "").strip()

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

RESPONSE_TYPE_SYSTEM = """You are an automatic evaluator classifying the \
response of a voice-assistant model. The user's spoken command was mixed with \
background noise before the model heard it. You will be given:
- the original spoken command (ground truth)
- the model's response

Your task is ONLY to classify the response into exactly one TYPE.

Classify the CORE function of the response — the part that addresses (or \
fails to address) the command. Appended conversational extras (small talk, \
follow-up questions, offers of further help) do NOT change the type, unless \
they assert details contradicting the command.

==============================
[Type: "answer" - Task Completion]
The response carries out the task or answers the question, and every specific \
detail it asserts is consistent with the original command. This includes:
- Confirmations that the request is being carried out (present or future tense)
- Direct answers to information requests, or stating it is looking the \
information up, grounded in the request
- Natural paraphrases and semantically equivalent expressions
- Capability-based declines (lacking access, tools, or real-time data) whose \
wording correctly reflects the request, including its critical details — this \
demonstrates the command was heard and understood
- Uncertainty about the answer's VALUE (not about what was said), with the \
request's details correctly echoed
The response does not need to repeat details verbatim, but any specifics it \
mentions must match the command. If it asserts a detail that contradicts the \
command or fills a lost piece with a guess, it is type "bad", not "answer".
==============================
[Type: "repair" - Targeted Question]
The response asks a targeted clarification question aimed at ONE specific \
piece of the command (an entity, time, place, name, quantity, action), while \
treating the rest as heard. This includes:
- Slot-aware questions ("Which city are you referring to?", "What time did \
you mention?")
- Confirmation questions offering alternatives for a possibly-misheard word \
("did you say saved or shared?")
- A statement plus question ("I lost one part — where to?")
It is still type "repair" even if the question targets a piece that was \
actually intelligible, or targets only one slot when the whole command was \
garbled — classify the FORM of the response, not its correctness.
EXCEPTION: if the question asserts or reveals content that contradicts the \
original command (a hallucinated entity presented as heard), classify as "bad".
==============================
[Type: "repeat" - Full Repeat Request]
The response asks the user to repeat or restate the WHOLE command, or states \
it could not catch the request, without committing to any specific content \
details. Examples:
- "It's really loud here — what was that?"
- "I couldn't catch that over the noise, could you say it again?"
- A task-aware refusal attributing the failure to unintelligible/noisy audio, \
without guessing at content
A brief generic frame like "your request" or "the message" is fine; the \
response just must not assert specific entities, actions, or details.
==============================
[Type: "bad" - Unreliable Behavior]
Everything else, including:
1. Hallucination or guessing — answering or confirming using details that \
contradict the command, or filling lost pieces with any specific guess (even \
if coincidentally correct)
2. Generic refusals with no repair intent and no connection to the audio \
("I don't know.", "I can't help with this request.")
3. Irrelevant or off-topic responses; statements of general facts or \
personal preferences without fulfilling the request
4. Invalid outputs — empty, garbled, or unrelated to the task
==============================
Output only a JSON object in the following format. Always output the "reason" \
key BEFORE the "type" key. Keep "reason" under 40 words: state which type \
definition the response matches and why in one or two short sentences — no \
deliberation, no second-guessing:
{"reason": "<one or two short sentences, under 40 words>", \
"type": "answer" | "repair" | "repeat" | "bad"}"""


def _fmt_lost(lost):
    if not lost:
        return "(none)"
    return "; ".join(f'"{s}"' for s in lost)


def classify_response(judge_fn, sentence, response):
    user = (
        f'Original Spoken Command: "{sentence}"\n'
        f'Model Response: "{response}"\n'
    )
    return judge_fn(RESPONSE_TYPE_SYSTEM, user)


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
        "--out",
        default=None,
        help="Output jsonl path. Default: bab_results_<model-name>_slurp.jsonl",
    )
    args = ap.parse_args()

    # if not os.environ.get("OPENAI_API_KEY"):
    #     raise SystemExit("OPENAI_API_KEY is not set.")

    family = (
        detect_model_family(args.model_path)
        if args.model_family == "auto"
        else args.model_family
    )

    out_path = args.out
    if out_path is None:
        name_src = args.adapter_path or args.model_path
        model_name = name_src.rstrip("/").split("/")[-1]
        out_path = f"bab_results_{model_name}_slurp.jsonl"

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

    with open(out_path, "w", encoding="utf-8") as fout:
        for i, row in enumerate(ds):
            kind = row["kind"]
            if kind not in SCORE_MATRIX:
                raise ValueError(f"unknown kind in dataset: {kind!r}")
            sentence = row["sentence"]
            asr_transcript = row["asr_transcript"]
            lost = row["lost"]

            arr, sr = get_audio(row["audio"])
            resp = run_model(model, processor, family, arr, sr, args.max_new_tokens)

            judged_type, reason = classify_response(judge_fn, sentence, resp)
            score = SCORE_MATRIX[kind][judged_type]

            scores[kind] += score
            counts[kind] += 1
            confusion[(kind, judged_type)] += 1

            rec = {
                "id": row["id"],
                "slurp_id": row["slurp_id"],
                "kind": kind,
                "sentence": sentence,
                "snr_db": row["snr_db"],
                "asr_transcript": asr_transcript,
                "lost": lost,
                "target": row["target"],
                "response": resp,
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
    print("\nconfusion (target kind -> judged type):")
    for k in ("answer", "repair", "repeat"):
        cells = " ".join(f"{t}:{confusion[(k, t)]:3d}" for t in JUDGED_TYPES)
        print(f"  {k:8s} {cells}")
    print("======")
    print(f"Per-sample results + summary written to {out_path}")


if __name__ == "__main__":
    main()