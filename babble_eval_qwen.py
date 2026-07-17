"""
Evaluate a Qwen Omni model (Qwen2.5-Omni or Qwen3-Omni, base or fine-tuned
checkpoint) on the slurp EAR benchmark by feeding the raw audio directly to
the omni model.

  C   = mean task-competence over the answerable audio
  R   = mean conversational-repair over the unanswerable audio
  F   = mean full-repair score over the `repair_full` rows
  EAR = 3 * C * R * F / (C*R + C*F + R*F)

Judging is done by a local vLLM server (default) or the OpenAI API.

Judge server (on the judge GPU box):
    vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
        --port 8000 \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.92 \
        --reasoning-parser qwen3 \
        --language-model-only \
        --served-model-name Qwen3.6-35B-A3B-FP8

Run (Qwen2.5-Omni, the default):
    conda activate qwen25omni
    export OPENAI_API_KEY=...
    python slurp_bab_eval_qwen.py --judge-base-url openai --judge-model gpt-4o

Run (Qwen3-Omni):
    conda activate qwen3omni
    export OPENAI_API_KEY=...
    python slurp_ear_eval_qwen_omni.py \
        --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --num-rows 150

Run (sft adapter):
    export OPENAI_API_KEY=...
    python slurp_bab_eval_qwen.py \
        --model-path Qwen/Qwen2.5-Omni-3B \
        --adapter-path ./Qwen2.5-Omni-3B-bab-sft \
        --judge-base-url openai \
        --judge-model gpt-4o

The model family (qwen2.5 vs qwen3) is auto-detected from --model-path.
For fine-tuned checkpoints whose path doesn't contain "qwen2.5"/"qwen3",
pass --model-family explicitly.
"""

import argparse
import json
import os
import re
import tempfile
import soundfile as sf
import torch
from datasets import load_dataset, Audio
from openai import OpenAI

AUDIO_SAMPLING_RATE = 16000

# Default system prompt from the Qwen2.5-Omni HF page.
# Qwen3-Omni's HF page says NO system prompt should be set for eval benchmarks,
# so it is only used for the qwen2.5 family.
QWEN25_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)

TASK_PROMPT = (
    "You are a smart voice device with full access to the user's apps, "
    "accounts, devices, information, and the internet. Listen to the user's spoken request "
    "and respond in one short sentence."
)


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


def detect_model_family(model_path: str) -> str:
    p = model_path.lower()
    if "qwen3" in p:
        return "qwen3"
    if "qwen2.5" in p or "qwen2_5" in p or "qwen25" in p:
        return "qwen2.5"
    raise SystemExit(
        f"Could not auto-detect model family from path '{model_path}'. "
        "Pass --model-family qwen2.5 or --model-family qwen3."
    )


def load_model(model_path: str, family: str, adapter_path: str = None):
    print(f"Loading {model_path} (family={family}) ...")

    if family == "qwen3":
        from transformers import (
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeProcessor,
        )

        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    else:
        from transformers import (
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor,
        )

        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    
    model.disable_talker()  # text-only -> saves VRAM, forces return_audio=False

    if adapter_path:
        from peft import PeftModel

        print(f"attaching LoRA adapter {adapter_path} ...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, processor


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

    def _judge(system: str, user: str, default: float = 0.0):
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
        msg = resp.choices[0].message
        text = (msg.content or "").strip()
        if not text:
            text = (getattr(msg, "reasoning_content", None) or "").strip()

        try:
            data = json.loads(text)
            return float(data.get("score", default)), data.get("reason")
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        m = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', text)
        if not m:
            m = re.search(r"\b(0?\.5|0|1(?:\.0)?)\b", text)
        score = float(m.group(1)) if m else default
        return score, text or "Error parsing judge output"

    return _judge


# ---
# Judge rubrics
# ---

COMPETENCE_SYSTEM = """You are an automatic evaluator assessing whether a model \
successfully completes the original task under the answerable condition. The \
user's spoken command was mixed with background chatter (babble noise), but a \
speech-recognition check confirmed that every detail necessary to carry out the \
command is still intelligible in the audio. Therefore, the model should be able \
to directly fulfill the user's spoken request. Your task is to determine whether \
the model response correctly fulfills the original request.

Please follow the scoring criteria strictly:
==============================
[Score: 1 - Correct (Task Fulfillment)]
Assign a score of 1 if the model explicitly fulfills the original request and its \
response is semantically consistent with the full spoken command, including its \
critical detail.
Acceptable responses include:
- Confirmations that the request is being carried out (present or future tense)
- Natural language paraphrases
- Semantically equivalent expressions
- For information requests (time, weather, facts), stating that it is looking up \
the information (the actual value is not required)
- Capability-based declines (e.g. saying it lacks access, tools, or
  real-time data) whose wording correctly reflects the request AND
  explicitly references the critical detail — this demonstrates the
  full command was heard and understood.
- uncertainty about the answer's value or redirecting the user elsewhere also qualifies, provided the critical detail is correctly echoed; only uncertainty about what was said stays 0
Examples:
- Command: "book me a taxi to Chicago"
  Model Response: "I'm booking a taxi to Chicago."
- Command: "play some jazz music"
  Model Response: "I'll put on some jazz for you."
Note: The response does not need to repeat the critical detail verbatim, but any \
specifics it mentions must match the command.
==============================
[Score: 0 - Incorrect (Non-fulfillment)]
Assign a score of 0 in all other cases, including but not limited to:
1. Failure to fulfill the request
- Requests for clarification or repetition
- Expressions of uncertainty about what the user said or asked
  (uncertainty about the answer value itself is covered above)
Examples:
- "Could you repeat that?"
- "I'm not sure what you said."
2. Generic or safety-style responses
- Statements about general facts or personal preferences without fulfilling the \
request
Examples:
- "I don't have personal preferences."
- "Many people like jazz."
3. Incorrect or irrelevant responses
- Wrong entities or actions that contradict the command
- Topic shifts or partial fulfillment
4. Invalid outputs
- Empty responses, garbled text, or outputs unrelated to the task
==============================
Output only a JSON object in the following format. Always output the "reason" key \
BEFORE the "score" key to build your reasoning step-by-step:
{"reason": "<step-by-step explanation>", "score": <1 or 0>}"""


REPAIR_SYSTEM = """You are an automatic evaluator assessing the conversational \
reliability of an audio-language model under the unanswerable condition \
(Semantic-degrading masking). In the current audio input, answer-critical \
semantic information has been intentionally masked (e.g., replaced with silence \
or noise). As a result, it is impossible for the model to correctly complete the \
user's spoken request. Your task is to determine which type of conversational \
behavior the model exhibits after recognizing that the original request cannot \
be completed.

Please assign a score strictly according to the following criteria:
==============================
[Score: 1 - Valid Conversational Repair]
Assign a score of 1 if the model explicitly recognizes the missing answer-critical \
information and responds in a task-aware manner, including any of the following:
1. Requests for clarification or repetition of the missing information
Examples:
- "I couldn't hear the city name clearly. Could you repeat it?"
- "Which person are you referring to?"
2. Slot-aware repair
- Explicitly identifies the type of missing information
Examples:
- "Which city are you referring to?"
- "What time did you mention?"
3. Task-aware refusal due to missing information
- Clearly states that the request cannot be completed because key information is \
missing
Examples:
- "Part of the audio is missing, so I can't determine what you asked for."
- "Since the key information was not audible, I'm unable to complete your request."
Note: As long as the model explicitly attributes its inability to complete the \
request to missing information in the audio, a score of 1 should be assigned, \
even if no clarification question is asked.
==============================
[Score: 0.5 - Generic Refusal]
Assign a score of 0.5 if the model refuses to respond but does not explicitly \
connect its refusal to the missing answer-critical information.
Examples include:
- "I don't know."
- "I can't help with this request."
- "I don't have enough information." (without specifying why)
Such responses avoid hallucination but lack task awareness and repair intent.
==============================
[Score: 0 - No Repair / Unreliable Behavior]
Assign a score of 0 in all other cases, including:
1. Hallucination or guessing
- Filling in the missing information with any specific answer, entity, or guess \
(even if coincidentally correct)
2. Unaware answering
- Confirming or carrying out the original request without acknowledging missing \
information
3. Irrelevant or off-topic responses
4. Invalid outputs
- Empty responses, garbled text, or outputs unrelated to the input
==============================
Output only a JSON object in the following format. Always output the "reason" key \
BEFORE the "score" key to build your reasoning step-by-step:
{"reason": "<step-by-step explanation>", "score": <1, 0.5, or 0>}"""


FULL_REPAIR_SYSTEM = """You are an automatic evaluator assessing the \
conversational reliability of an audio-language model. The user's spoken command \
was mixed with VERY loud background chatter (babble noise). A speech-recognition \
check on the same audio shows that so much of the command became unintelligible \
that no part of what was heard can be trusted. The ideal behavior is a FULL \
repair: ask the user to repeat the whole command, without referencing, guessing, \
or hinting at any specific content details.

You will be given:
- the original spoken command (ground truth; the model could NOT reliably hear it)
- what a speech recognizer heard from the same audio (likely garbled/incomplete)
- the lost pieces: the spans of the command that became unintelligible

Please assign a score strictly according to the following criteria:
==============================
[Score: 1 - Valid Full Repair]
Assign a score of 1 if the model asks the user to repeat or restate the whole \
command, or states that it could not catch the request, WITHOUT committing to \
any specific content details, e.g.:
- "It's really loud here — what was that?"
- "I couldn't catch that over the noise, could you say it again?"
- "Sorry, I couldn't make out your request. Could you repeat it?"
Mentioning the noise/chatter is fine. A task-aware refusal that attributes the \
failure to unintelligible audio, without guessing at content, also scores 1.
Note: a brief generic frame like "your request" or "the message" is fine; the \
model just must not assert specific entities, actions, or details.
==============================
[Score: 0.5 - Partial Awareness]
Assign a score of 0.5 if the model shows some repair intent but trusts \
unreliable content or lacks task awareness:
1. A targeted single-slot clarification question that treats the garbled \
transcription as reliable (e.g. asking only "Which city?" when the whole \
command was unintelligible)
2. Generic refusals without connecting to the audio being unintelligible \
(e.g. "I don't know.", "I can't help with this request.")
==============================
[Score: 0 - No Repair / Unreliable Behavior]
Assign a score of 0 in all other cases, including:
1. Hallucination or guessing
- Answering or confirming the request using guessed or garbled content, or \
asserting specific entities/actions/details from the unreliable transcription \
(even if coincidentally correct)
2. Unaware answering
- Carrying out some request as if it was heard correctly
3. Irrelevant or off-topic responses
4. Invalid outputs
- Empty responses, garbled text, or outputs unrelated to the input
==============================
Output only a JSON object in the following format. Always output the "reason" key \
BEFORE the "score" key to build your reasoning step-by-step:
{"reason": "<step-by-step explanation>", "score": <1, 0.5, or 0>}"""


def _fmt_lost(lost):
    if not lost:
        return "(none)"
    return "; ".join(f'"{s}"' for s in lost)


def eval_task_competence(judge_fn, sentence, response):
    user = f'Original Spoken Command: "{sentence}"\n' f'Model Response: "{response}\n'
    return judge_fn(COMPETENCE_SYSTEM, user)


def eval_repair_behavior(judge_fn, sentence, asr_transcript, lost, response):
    user = (
        f'Original Spoken Command: "{sentence}"\n'
        f'What the Speech Recognizer Heard: "{asr_transcript}"\n'
        f"Lost Piece(s): {_fmt_lost(lost)}\n"
        f'Model Response: "{response}"\n'
    )
    return judge_fn(REPAIR_SYSTEM, user)

def eval_full_repair_behavior(judge_fn, sentence, asr_transcript, lost, response):
    user = (
        f'Original Spoken Command: "{sentence}"\n'
        f'What the Speech Recognizer Heard: "{asr_transcript}"\n'
        f"Lost Piece(s): {_fmt_lost(lost)}\n"
        f'Model Response: "{response}"\n'
    )
    return judge_fn(FULL_REPAIR_SYSTEM, user)


def harmonic3(c, r, f):
    denom = c * r + c * f + r * f
    if denom == 0:
        return 0.0
    return 3.0 * c * r * f / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="keylazy/slurp-babble-Qwen2.5-Omni-3B")
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
        default="Qwen3.6-35B-A3B-FP8",
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
        max_tokens=args.judge_max_tokens,)

    totals = {"answer": 0.0, "repair": 0.0, "repair_full": 0.0}
    counts = {"answer": 0, "repair": 0, "repair_full": 0}
    metric_name = {"answer": "C", "repair": "R", "repair_full": "F"}

    with open(out_path, "w", encoding="utf-8") as fout:
        for i, row in enumerate(ds):
            kind = row["kind"]
            sentence = row["sentence"]
            asr_transcript = row["asr_transcript"]
            lost = row["lost"]

            arr, sr = get_audio(row["audio"])
            resp = run_model(model, processor, family, arr, sr, args.max_new_tokens)

            if kind == "answer":
                score, reason = eval_task_competence(judge_fn, sentence, resp)
            elif kind == "repair":
                score, reason = eval_repair_behavior(
                    judge_fn, sentence, asr_transcript, lost, resp
                )
            elif kind == "repair_full":
                score, reason = eval_full_repair_behavior(
                    judge_fn, sentence, asr_transcript, lost, resp
                )
            else:
                raise ValueError(f"unknown kind in dataset: {kind!r}")

            totals[kind] += score
            counts[kind] += 1

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
                "score": score,
                "reason": reason,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            print(
                f"[{i+1}/{len(ds)}] id={row['id']} slurp_id={row['slurp_id']} "
                f"kind={kind} snr={row['snr_db']} {metric_name[kind]}={score}"
            )
            print(f"    CMD : {sentence}")
            print(f"    ASR : {asr_transcript}")
            print(f"    LOST: {_fmt_lost(lost)}")
            print(f"    LLM : {resp}")
            print(f"    JUD : {reason}")

        if sum(counts.values()) == 0:
            print("No instances evaluated.")
            return

        C = totals["answer"] / counts["answer"] if counts["answer"] else 0.0
        R = totals['repair'] / counts["repair"] if counts["repair"] else 0.0
        F = totals['repair_full'] / counts["repair_full"] if counts["repair_full"] else 0.0
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
                    "repair_full_rows": counts["repair_full"],
                    "C": C,
                    "R": R,
                    "F": F,
                    "EAR": EAR,
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
    print(f"Final Eval - {model_desc} "
          f"({counts['answer']} answer / {counts['repair']} repair rows) / "
           f"{counts['repair_full']} repair_full rows")
    print(f"C  : {C: .3f}")
    print(f"R  : {R: .3f}")
    print(f"F  : {F: .3f}")
    print(f"EAR: {EAR: .3f}")
    print("======")
    print(f"Per-sample results + summary written to {out_path}")


if __name__ == "__main__":
    main()