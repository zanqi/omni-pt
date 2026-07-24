"""
Build babble-noise slurp dataset for conversational-repair training.

Scheme: For each noisy probe, the omni base model produce an ASR
transcript and a task response. The LLM  sees (original sentence,
transcript, response) and labels the probe:
- "answer":     every detail needed to perform the task survived the noise;
                filler only loss
- "repair":     exactly ONE key piece was lost or misheard;
                the rest can be trusted
- "repeat":    more than one key piece lost in both passes
"""

import argparse
import itertools
import json
import logging
import os
import random
import re
import string
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, Dataset, DatasetDict, load_dataset
from openai import OpenAI
from qwen_omni_utils import process_mm_info
from tqdm import tqdm
from util import QWEN25_SYSTEM_PROMPT, detect_model_family, load_model
from prompts import TASK_PROMPT

skip = Counter()


def _drop(record):
    return "audio output may not work" not in record.getMessage()


root = logging.getLogger()
root.addFilter(_drop)


def log(*args):
    """print-compatible logging that doesn't break the tqdm bar."""
    tqdm.write(" ".join(str(a) for a in args))


AUDIO_SAMPLING_RATE = 16000

# defend against single long audio causing oom
MAX_AUDIO_SECONDS = 30

N_TRAIN_TRIPLETS = 1000
N_TEST_TRIPLETS = 50

# Classification + Target generation are served by the local vLLM judge
# box. Its slurm job records the node it landed on in VLLM_HOST_FILE.
TARGET_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
VLLM_HOST_FILE = "/gscratch/sciencehub/zanqil/vllm_judge/vllm_judge_host.txt"
REPO_TMPL = "keylazy/slurp-babble-{}-v1"
MASK_DS_ID = "keylazy/slurp-ear-sft"
OUT_DIR = "babble_audio"
SEED = 42
ROW_ID = itertools.count(1)

BABBLE_POOL_SIZE = 300
BABBLE_SPEAKERS = 3
BABBLE_CLIP_MAX_SEC = 10  # trim pool clips to save memory

PROBE_BATCH_SIZE = 16
MAX_PROBES = 3
SLOT_SNR = {
    "answer": (8.0, 20.0),
    "repair": (0.0, 12.0),
    "repeat": (0.0, 4.0),
}
SLOT_WEIGHTS = {"answer": 1, "repair": 2, "repeat": 2}

CLASSIFY_TEMPERATURE = 0.0
CLASSIFY_MAX_TOKENS = 1024
TARGET_MAX_TOKENS = 1024
CLASSIFY_WORKERS = 8  # parallel classifier calls to vLLM

ASR_MAX_NEW_TOKENS = 64
RESP_MAX_NEW_TOKENS = 256  # task response from base omni model

KINDS = ("answer", "repair", "repeat")


random.seed(SEED)
np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

with open(VLLM_HOST_FILE) as _f:
    _vllm_host = _f.read().strip()
client = OpenAI(base_url=f"http://{_vllm_host}:8000/v1", api_key="EMPTY")
print(f"target model: {TARGET_MODEL} @ http://{_vllm_host}:8000/v1")

# set by init_base_model() from --omni-path before build_triplets runs
base_model = None
base_processor = None
base_family = None
IM_END_ID = None


# ---
# base model: ASR + task response
# ---

ASR_SYSTEM_PROMPT = "You are a speech recognition model."
ASR_PROMPT = "Transcribe the English audio into text without any punctuation marks."


def _conv(audio, system_prompt, user_prompt):
    conv = []
    if system_prompt is not None:
        conv.append(
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        )
    conv.append(
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": user_prompt},
            ],
        }
    )
    return conv


@torch.inference_mode()
def base_generate_batch(convs, max_new_tokens):
    texts = base_processor.apply_chat_template(
        convs, add_generation_prompt=True, tokenize=False
    )
    mm_audios, images, videos = process_mm_info(convs, use_audio_in_video=False)
    inputs = base_processor(
        text=texts,
        audio=mm_audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
    ).to(base_model.device, dtype=base_model.dtype)
    out = base_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=IM_END_ID,
        pad_token_id=IM_END_ID,
    )
    gen = out[:, inputs["input_ids"].shape[1] :]
    return [
        t.lower().strip()
        for t in base_processor.batch_decode(gen, skip_special_tokens=True)
    ]


# ---
# LLM calls (vLLM judge server)
# ---


def gpt_json(system, user, temperature, max_tokens):
    raw = None
    try:
        resp = client.chat.completions.create(
            model=TARGET_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        msg = resp.choices[0].message
        raw = (msg.content or "").strip()
        if not raw:
            # reasoning models may leave the answer in reasoning_content
            raw = (getattr(msg, "reasoning", None) or "").strip()
        return json.loads(raw.replace("```json", "").replace("```", "").strip())
    except Exception as e:
        log(f"target-model error: {e}\nraw: {raw}")
        return None


# ---
# LLM probe classification
# ---

# Prompt is split into a static SYSTEM prefix (identical on every call) and a
# tiny USER suffix carrying only the per-probe data. The invariant rubric
# therefore forms a long shared prefix that vLLM's automatic prefix cache can
# reuse across all classify calls; only the short trailing case is recomputed.
CLASSIFY_SYSTEM = """You are labeling noisy-audio for training a smart \
voice assistant. You will be given three texts:
- COMMAND: the user's real spoken command.
- TRANSCRIPT: a speech recognizer's transcription of the SAME command after \
it was mixed with loud background chatter.
- REPLY: a voice assistant's reply to that same noisy audio.
COMMAND is the ground-truth command. TRANSCRIPT, and REPLY are the two \
independent passes over the same noisy audio.

"Key info" means any piece the assistant must know to correctly perform the \
task: entities, names, places, times, dates, quantities, titles, the \
requested action or topic. A piece is key info ONLY if the task cannot be \
correctly performed without it. Carrier/filler words ("please", "could you", \
"hey", "tell me", "what") are NOT key info. Wake words and the assistant's \
name or vocative ("hey olly", "ok google", "assistant") are NOT key info \
either — they are not needed to perform the task, so a misheard wake word \
("olly" heard as "ollie") never counts as a loss or triggers a repair. \
Neither is a word whose \
meaning is already implied by the rest of the command (e.g. "set" in "are \
there any alarms set" — the command means the same thing without it).

First decide, for EACH key piece of the real command, whether it SURVIVED \
the noise. Survival is about whether the piece was HEARD, judged \
semantically, not about exact wording:
- A piece SURVIVED if the transcript contains it correctly (minor wording or \
spelling differences are fine), OR the assistant's reply demonstrates it \
heard and understood that piece. Judge the reply by whether it shows the \
piece got through, NOT by whether it repeats the piece word-for-word: a \
natural paraphrase counts, and so does a capability-decline or a hand-off \
to the user that correctly refers to the piece (e.g. "i cant check the game \
score for you", "you can see your alarms in the clock app") — both \
demonstrate the piece was heard even though the task is refused or offloaded. \
What matters is that the reply uses the piece correctly, whether or not the \
reply agrees to, is able to, or actually does perform the task.
- The COMMAND, TRANSCRIPT, and REPLY are all shown lowercase with punctuation \
stripped, so compare on the words alone — case and punctuation are never \
evidence of loss or of the action/intent changing.
- A piece was LOST only if BOTH passes missed it: it is absent, garbled, or \
replaced by a wrong word in the transcript, AND the reply neither contains \
it nor otherwise demonstrates it was heard.
- The assistant asserting a wrong detail does NOT mark a piece lost when the \
transcript has that piece correctly — the transcript alone is sufficient \
proof of survival.
- Losing only filler words never counts as a loss.
- Singular/plural, spelling, and other minor wording differences in the \
transcript count as survived, even if the reply heard something else.
- SPECIAL CARE with substituted words. A DIFFERENT word in the transcript is \
a mishearing. HOWEVER, if the REPLY contains the correct original word or \
clearly demonstrates it understood the intent anyway, the piece still SURVIVED. \
A correct reply always overrides a bad transcript. Do NOT rationalize \
meaning-changing swaps (e.g., "controls" for "choose") as a spelling variant.
- The action/intent changing counts as that action piece being lost — but \
only when the WORDS actually change (e.g. "set an alarm" heard as "cancel \
an alarm"), never merely because the transcript is rendered with standard \
capitalization/punctuation.

Evaluate the transcript and the reply SEPARATELY before making a final verdict. \
Keep evaluations short and under 60 words total.

Then classify as exactly one of:
- "answer": every key piece survived.
- "repair": exactly ONE key piece was lost and the rest of the command's \
key information survived. Before choosing "repair", apply this test: if the \
user answered a question recovering only the lost piece, would their reply \
tell the assistant anything it actually needs? If the command is already \
complete and unambiguous without the piece, classify "answer" instead. If \
the lost piece was replaced by a similar-sounding wrong word (in the \
transcript or the reply), report that wrong word in "misheard_as".
- "repeat": more than one key piece was lost, OR the lost piece(s) make up \
half or more of the command's key information — including when the command \
has only ONE key piece and it was lost — OR the audio was so garbled that \
neither pass caught the key pieces. A targeted question is impossible when \
there is not enough reliably-heard command left to anchor it on.

Return ONLY valid JSON in exactly this shape (evaluations first):
{"transcript_evaluation": "<one short verdict per key piece checking ONLY the transcript, e.g. '7 am: lost; alarm: survived'>", \
"reply_evaluation": "<one short verdict per key piece checking ONLY the reply, e.g. '7 am: missing; alarm: survived because reply says alarms'>", \
"missing": ["<lost key piece 1>", ...], \
"misheard_as": "<wrong word heard instead, or empty string>", \
"kind": "answer" | "repair" | "repeat"}

Rules for "missing": quote the lost pieces using the words of the REAL \
command. For "answer" it must be an empty list. For "repair" it must contain \
exactly one key piece. For "repeat" it must contain more than one key piece. \
Rules for "misheard_as": when "kind" is "repair", and the lost key piece is \
being misheard as wrong word(s), not deletion. Quote the misheard word(s). \
Otherwise, it will be empty."""


CLASSIFY_USER = (
    'COMMAND:\n"{sentence}"\n\n'
    'TRANSCRIPT:\n"{transcript}"\n\n'
    'REPLY:\n"{response}"'
)


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize_text(s):
    """Lowercase + strip punctuation, so the judge can't misread ASR-standard \
    capitalization/punctuation as a wording or intent change."""
    return " ".join(s.translate(_PUNCT_TABLE).lower().split())


def _words(s):
    return re.findall(r"[a-z0-9']+", s.lower())


# ---
# Target generation
# ---

TARGET_SYSTEM = """You are writing training targets for a smart voice \
assistant that has full access to the user's apps, accounts, information, and the internet.

You will be given, in the next message, the user's spoken COMMAND and two \
speech-recognition transcripts of it under increasing background chatter: a \
REPAIR-TRANSCRIPT that lost exactly one piece (with the LOST-PIECE named, and \
a MISHEARD-AS value if that piece was swapped for a wrong word), and a \
REPEAT-TRANSCRIPT that lost too much. Produce three targets for that COMMAND.

Return ONLY valid JSON in exactly this shape:
{"answer": "<a short natural response, covering every part of the request>", \
"repair": "<one short question>", \
"repeat": "<one short request>"}

Rules for "answer": despite background chatter, the full command was heard correctly.
    - If the command asks for more than one distinct thing (e.g. two \
questions joined by "and", asked back-to-back, or a request plus a \
follow-up question), your response must address EVERY part — never answer \
only the first part and drop the rest.
    - If the request asks for information (time, weather, facts) \
and you know the answer, answer DIRECTLY with the correct fact(s), using as \
few natural sentences as it takes to cover every part asked (often one, \
sometimes two). Otherwise, say you are looking it up, but ground the \
response in what was heard: refer to each part of the request so it is \
clear the assistant followed everything.
    - If the request is a task request, confirm the assistant is carrying \
out the request in one or two natural sentences. Use present or future \
tense ("I'm setting...", "I'll remind you...")
    - never claim the action is already done.
    - Stay natural and concise — don't pad with extra sentences beyond what \
covering the full request requires.

Rules for "repair": the device heard the command over loud background \
chatter, and its speech recognition produced the REPAIR-TRANSCRIPT. The one \
piece it lost from the real command is the LOST-PIECE; everything else can be \
treated as heard. If a MISHEARD-AS value is given, the recognizer heard that \
similar-sounding wrong word in place of the lost piece. \
Write ONE short natural question (under 20 words) that recovers ONLY that \
piece. Test: if the user replied with just the missing words, the command \
would be complete.
  - NEVER ask about parts that were heard correctly — asking about them again \
would sound like the assistant wasn't listening.
  - Ground the question in the parts heard correctly (words matching \
the original), so it is clear the assistant followed everything except this one piece.
  - Do not reveal the missing words. ONE exception: if a word was swapped \
for a similar-sounding wrong word (the MISHEARD-AS value), you may ask a \
confirmation question that offers the true word AND the misheard word as \
alternatives ("did you say saved or shared?") — never the true word alone.
  - Sound like natural speech, not a form. Vary structure freely: \
"Which...?", "How long before...?", "Who should...?", "What time...?", \
"Where...?", or a statement+question like "I lost one part — where to?". \
Do NOT default to starting with "Sorry".

Rules for "repeat": at even louder chatter, speech recognition produced the \
REPEAT-TRANSCRIPT. Too many pieces were lost for a targeted question. Write \
ONE short natural request (under 15 words) asking the user to repeat the \
whole command.
  - Do NOT reference, guess, or hint at ANY content details from either the \
real command or the garbled transcription — the assistant cannot trust any of it.
  - Mentioning the noise/chatter is fine and helps explain why.
  - Sound like natural speech and vary phrasing ("It's really loud here — \
what was that?", "I couldn't catch that over the noise, could you say it \
again?"). Do NOT default to starting with "Sorry".
"""


# variable suffix, kept last so TARGET_SYSTEM stays cacheable
TARGET_USER = (
    'COMMAND:\n"{sentence}"\n\n'
    'REPAIR-TRANSCRIPT:\n"{repair_transcript}"\n'
    'LOST-PIECE: "{lost_span}"{swap_note}\n\n'
    'REPEAT-TRANSCRIPT:\n"{full_transcript}"'
)


def generate_targets(sentence, repair_probe, repeat_probe, retries=3):
    swap_note = ""
    if repair_probe["swapped"]:
        swap_note = f'\nMISHEARD-AS: "{repair_probe["swapped"][0]}"'
    user = TARGET_USER.format(
        sentence=sentence,
        repair_transcript=repair_probe["transcript"],
        lost_span=repair_probe["lost"][0],
        swap_note=swap_note,
        full_transcript=repeat_probe["transcript"],
    )

    answer = None
    for attempt in range(retries):
        obj = gpt_json(
            TARGET_SYSTEM, user, temperature=0.7, max_tokens=TARGET_MAX_TOKENS
        )
        if obj is None:
            time.sleep(2**attempt)
            continue
        answer = str(obj.get("answer", "")).strip() or answer
        repair = str(obj.get("repair", "")).strip()
        repeat = str(obj.get("repeat", "")).strip()

        if answer and repair and repeat:
            return answer, repair, repeat
    return None, None, None  # skip this slurp audio


# ---
# SNR probing -> classify its kind
# ---


def make_probe_batch(clean, pool, missing):
    def make_babble(pool, length):
        picks = random.sample(pool, BABBLE_SPEAKERS)
        mixed = np.zeros(length, dtype=np.float32)
        for b in picks:
            if len(b) < length:
                b = np.pad(b, (0, length - len(b)), "wrap")
            else:
                start = random.randint(0, len(b) - length)
                b = b[start : start + length]
            mixed += b
        return mixed / BABBLE_SPEAKERS

    audios, snrs = [], []
    while len(audios) < PROBE_BATCH_SIZE:
        # sample snr
        weights = [SLOT_WEIGHTS[k] for k in missing]
        slot = random.choices(missing, weights=weights, k=1)[0]
        # round to 1 decimal digit
        snr = round(random.uniform(*SLOT_SNR[slot]), 1)

        # synthesize noisy audio
        # SNR = 10*log10(clean_power / babble_power)
        #   -> target_babble_power = clean_power / 10^(SNR/10)
        #   -> scale babble = sqrt(target_power / current_power)
        babble = make_babble(pool, len(clean))
        clean_power = float(np.mean(clean**2))
        current_babble_power = float(np.mean(babble**2))
        target_babble_power = clean_power / (10 ** (snr / 10))
        scale = np.sqrt(target_babble_power / current_babble_power)
        noisy = clean + scale * babble
        peak = float(np.max(np.abs(noisy)))
        if peak > 1.0:
            # avoid clipping on save; rescaling do not change SNR
            noisy = noisy / peak
        noisy = noisy.astype(np.float32)

        audios.append(noisy)
        snrs.append(snr)
    return audios, snrs


def probe_triplet(clean, pool, sentence):
    def classify(sentence, transcript, response, retries=2):
        user = CLASSIFY_USER.format(
            sentence=_normalize_text(sentence),
            transcript=_normalize_text(transcript),
            response=_normalize_text(response),
        )
        for attempt in range(retries + 1):
            obj = gpt_json(
                CLASSIFY_SYSTEM,
                user,
                temperature=CLASSIFY_TEMPERATURE,
                max_tokens=CLASSIFY_MAX_TOKENS,
            )

            # parse classifier response
            if obj is None:
                return None
            kind = str(obj.get("kind", "")).strip().lower()
            if kind not in KINDS:
                return None
            missing = [str(s).strip() for s in obj.get("missing", []) if str(s).strip()]
            misheard = str(obj.get("misheard_as", "")).strip()
            
            # Extract both evaluations and combine them for the final dataset
            transcript_eval = str(obj.get("transcript_evaluation", "")).strip()
            reply_eval = str(obj.get("reply_evaluation", "")).strip()
            reason = f"Transcript: {transcript_eval} | Reply: {reply_eval}"

            if (kind == "answer" and missing) or (
                kind == "repair" and len(missing) != 1
            ):
                time.sleep(2**attempt)
                continue

            return {
                "kind": kind,
                "missing": missing,
                "misheard_as": misheard,
                "reason": reason,
            }

        return None

    results: dict[str, dict | None] = {k: None for k in KINDS}
    for _ in range(MAX_PROBES):
        missing_slots = [k for k, v in results.items() if v is None]
        if not missing_slots:
            break
        audios, snrs = make_probe_batch(clean, pool, missing_slots)

        # get batch omni asr respond
        sysp = ASR_SYSTEM_PROMPT if base_family == "qwen2.5" else None
        convs = [_conv(a, sysp, ASR_PROMPT) for a in audios]
        transcripts = base_generate_batch(convs, ASR_MAX_NEW_TOKENS)

        # get batch omni assistant respond
        sysp = QWEN25_SYSTEM_PROMPT if base_family == "qwen2.5" else None
        convs = [_conv(a, sysp, TASK_PROMPT) for a in audios]
        responses = base_generate_batch(convs, RESP_MAX_NEW_TOKENS)

        with ThreadPoolExecutor(max_workers=CLASSIFY_WORKERS) as ex:
            labels = list(
                ex.map(
                    lambda it: classify(*it),
                    [(sentence, t, r) for t, r in zip(transcripts, responses)],
                )
            )

        for snr, noisy, transcript, response, label in zip(
            snrs, audios, transcripts, responses, labels
        ):
            if label is None:
                continue
            kind = label["kind"]
            if results[kind] is None:
                results[kind] = {
                    "snr_db": snr,
                    "audio": noisy,
                    "transcript": transcript,
                    "response": response,
                    "lost": label["missing"],
                    "swapped": [label["misheard_as"]] if label["misheard_as"] else [],
                    "reason": label["reason"],
                }

    if any(v is None for v in results.values()):
        return None
    return results


# ---
# triplet-building loop
# ---


def build_triplets(source_split, n_triplets, tag, seen_slurp_ids):
    # collect babble pool
    stream = load_dataset("qmeeus/slurp", split=source_split, streaming=True)
    stream = stream.cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE))
    max_len = BABBLE_CLIP_MAX_SEC * AUDIO_SAMPLING_RATE
    pool = []
    for row in stream:
        arr = row["audio"]["array"].astype(np.float32)[:max_len]
        if len(arr) > AUDIO_SAMPLING_RATE:
            # only add clips longer than 1 sec
            pool.append((row["slurp_id"], arr))
        if len(pool) >= BABBLE_POOL_SIZE:
            break
    log(f"[{source_split}] babble pool: {len(pool)} clips")

    stream = load_dataset("qmeeus/slurp", split=source_split, streaming=True)
    stream = stream.cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE))

    rows, scanned, done = [], 0, 0
    skip.clear()
    pbar = tqdm(total=n_triplets, desc=f"[{tag}]", unit="triplet", dynamic_ncols=True)
    for row in stream:
        if done >= n_triplets:
            break
        scanned += 1
        pbar.set_postfix({**skip, "scanned": scanned}, refresh=False)

        slurp_id = row["slurp_id"]
        sentence = row["sentence"]
        if slurp_id in seen_slurp_ids or len(sentence.split()) < 4:
            skip["seen/short"] += 1
            continue

        clean = row["audio"]["array"].astype(np.float32)
        clean = clean[: MAX_AUDIO_SECONDS * AUDIO_SAMPLING_RATE]

        triplet = probe_triplet(
            clean, [arr for sid, arr in pool if sid != slurp_id], sentence
        )
        if triplet is None:
            skip["probe"] += 1
            continue

        answer_target, repair_target, repeat_target = generate_targets(
            sentence, triplet["repair"], triplet["repeat"]
        )
        if not answer_target:
            skip["targets"] += 1
            continue
        targets = {
            "answer": answer_target,
            "repair": repair_target,
            "repeat": repeat_target,
        }

        seen_slurp_ids.add(slurp_id)
        for kind in KINDS:
            probe = triplet[kind]
            assert probe is not None
            path = os.path.join(OUT_DIR, f"{tag}_{slurp_id}_{kind}.wav")
            sf.write(path, probe["audio"], AUDIO_SAMPLING_RATE)
            rows.append(
                {
                    "id": next(ROW_ID),
                    "kind": kind,
                    "target": targets[kind],
                    "audio": path,
                    "snr_db": probe["snr_db"],
                    "asr_transcript": probe["transcript"],
                    "omni_response": probe["response"],
                    "lost": probe["lost"],
                    "swapped": probe["swapped"],
                    "classifier_reason": probe["reason"],
                    "slurp_id": slurp_id,
                    "sentence": sentence,
                    "source": "babble",
                }
            )

        done += 1
        pbar.update(1)

    pbar.close()
    log(f"[{tag}] built {len(rows)} rows from {done} utterances ({scanned} scanned)")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--omni-path", default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument(
        "--ds-id",
        required=True,
        help="HF Hub repo id to push the generated dataset to. Defaults to "
        "REPO_TMPL formatted with the omni model name.",
    )
    args = ap.parse_args()

    # init base omni model
    base_family = detect_model_family(args.omni_path)
    base_model, base_processor = load_model(
        args.omni_path, base_family, thinker_only=True
    )
    IM_END_ID = base_processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    print("base models loaded")

    # find used slurp ids to avoid
    seen_ids = set()
    for split in ("train", "test"):
        ds = load_dataset(MASK_DS_ID, split=split, streaming=True)
        ds = ds.select_columns(["slurp_id"])
        for r in ds:
            seen_ids.add(r["slurp_id"])

    test_rows = build_triplets("test", N_TEST_TRIPLETS, "test", seen_ids)
    train_rows = build_triplets("train", N_TRAIN_TRIPLETS, "train", seen_ids)

    def list2ds(rows):
        return Dataset.from_list(rows).cast_column(
            "audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE)
        )

    # push to hub
    repo_id = args.ds_id
    DatasetDict({"train": list2ds(train_rows), "test": list2ds(test_rows)}).push_to_hub(
        repo_id
    )
    log(
        f"Pushed {len(train_rows)} train / {len(test_rows)} test rows " f"to {repo_id}."
    )
