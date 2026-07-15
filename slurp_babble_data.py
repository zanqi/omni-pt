"""
Build babble-noise slurp dataset for conversational-repair training.

Babble is mixed from BABBLE_SPEAKERS utterances at a randomly
sampled SNR -- "clean" or uniform in [0, 20] dB.

The base model (Qwen2.5-Omni-3B) transcribes it, and an alignment
judge compares the transcript vs ground-truth sentence:
  answer: nothing lost and nothing hallucinated
  repair: an essential detail was lost (deleted or substituted)

Probing continues (fresh SNR + fresh babble each draw, up to
MAX_PROBES) until the utterance produces 1 audio of each kind.
Skip if the utterance can't produce them.

Contamination guards:
  - slurp_ids already used by keyword-masking dataset are excluded
    to avoid double weighting an utterance on same response
  - babble come from the same split but never include the utterance
"""

import difflib
import itertools
import json
import logging
import os
import random
import time
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, Dataset, DatasetDict, load_dataset
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from openai import OpenAI
from qwen_omni_utils import process_mm_info
from tqdm import tqdm
from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
    WhisperTokenizer,
)
from collections import Counter

skip = Counter()


def _drop(r):
    return "audio output may not work" not in r.getMessage()


root = logging.getLogger()
root.addFilter(_drop)


def log(*args):
    """print-compatible logging that doesn't break the tqdm bar."""
    tqdm.write(" ".join(str(a) for a in args))


AUDIO_SAMPLING_RATE = 16000
MAX_AUDIO_SECONDS = 30  # TODO: needed?

N_TRAIN_TRIPLETS = 1000
N_TEST_TRIPLETS = 50

BASE_MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
OPENAI_MODEL = "gpt-4o"
REPO_ID = "keylazy/slurp-babble-Qwen2.5-Omni-3B"
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
    "repair_full": (0.0, 4.0),
}
SLOT_WEIGHTS = {"answer": 1, "repair": 2, "repair_full": 2}
FULL_REPEAT_MIN_LOST = 2
LOST_FRACTION_FULL = 0.5


nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english")) - {
    "am",
    "pm",
    "no",
    "not",
    "nor",
    "on",
    "off",
    "up",
    "down",
    "what",
    "when",
    "where",
    "who",
    "which",
    "how",
    "why",
    "all",
    "now",
    "again",
}
STOPWORDS.update(["please"])


def _is_content(tok):
    return tok not in STOPWORDS


def _content_words(tokens):
    return [t for t in tokens if _is_content(t)]


def kind_of(lost_spans, untrusted, sentence):
    if not lost_spans:
        return None if untrusted else "answer"
    if len(lost_spans) >= FULL_REPEAT_MIN_LOST:
        return "repair_full"
    n_total = len(_content_words(sentence_tokens(sentence)))
    n_lost = sum(len(_content_words(s.split())) for s in lost_spans)
    if n_total and n_lost / n_total >= LOST_FRACTION_FULL:
        return "repair_full"
    return "repair" if not untrusted else "repair_full"


random.seed(SEED)
np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

client = OpenAI()

base_processor = Qwen2_5OmniProcessor.from_pretrained(BASE_MODEL_ID)
base_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
).eval()
IM_END_ID = base_processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
print("base models loaded")


_wtok = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
_stem = PorterStemmer().stem


def sentence_tokens(sentence):
    return _wtok.normalize(sentence).split()


def _stems(tokens):
    return {_stem(t) for t in tokens}


def lost_content_words(sentence, transcript):
    heard = _stems(sentence_tokens(transcript))
    return {
        w for w in sentence_tokens(sentence) if _stem(w) not in heard and _is_content(w)
    }


def leaks(sentence, transcript, repair_text):
    """
    True if the repair text contains the lost words.
    """
    rep = _stems(sentence_tokens(repair_text))
    return bool(_stems(lost_content_words(sentence, transcript)) & rep)


# ---
# Babble synthesis
# ---


def collect_babble_pool(source_split, size):
    stream = load_dataset("qmeeus/slurp", split=source_split, streaming=True)
    stream = stream.cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE))
    max_len = BABBLE_CLIP_MAX_SEC * AUDIO_SAMPLING_RATE
    pool = []
    for row in stream:
        arr = row["audio"]["array"].astype(np.float32)[:max_len]
        if len(arr) > AUDIO_SAMPLING_RATE:
            # only add clips longer than 1 sec
            pool.append((row["slurp_id"], arr))
        if len(pool) >= size:
            break
    log(f"[{source_split}] babble pool: {len(pool)} clips")
    return pool


def make_babble(pool, length, exclude_slurp_id):
    candidates = [arr for sid, arr in pool if sid != exclude_slurp_id]
    picks = random.sample(candidates, BABBLE_SPEAKERS)
    mixed = np.zeros(length, dtype=np.float32)
    for b in picks:
        if len(b) < length:
            b = np.pad(b, (0, length - len(b)), "wrap")
        else:
            start = random.randint(0, len(b) - length)
            b = b[start : start + length]
        mixed += b
    return mixed / BABBLE_SPEAKERS


def synthesize_noisy_audio(clean, pool, snr_db, exclude_slurp_id):
    """
    Mix babble into `clean` at `snr_db`.
    SNR = 10*log10(clean_power / babble_power)
      -> target_babble_power = clean_power / 10^(SNR/10)
      -> amplitude scale = sqrt(target_power / current_power)
    """
    babble = make_babble(pool, len(clean), exclude_slurp_id)
    clean_power = float(np.mean(clean**2))
    babble_power = float(np.mean(babble**2))
    if babble_power < 1e-10 or clean_power < 1e-10:
        return None
    target_babble_power = clean_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_babble_power / babble_power)
    noisy = clean + scale * babble
    peak = float(np.max(np.abs(noisy)))
    if peak > 1.0:
        # avoid clipping on save; rescaling do not change SNR
        noisy = noisy / peak
    return noisy.astype(np.float32)


# ---
# base model ASR
# ---

ASR_SYSTEM_PROMPT = "You are a speech recognition model."
ASR_PROMPT = "Transcribe the English audio into text without any punctuation marks."


def _asr_conv(audio):
    return [
        {"role": "system", "content": [{"type": "text", "text": ASR_SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": ASR_PROMPT},
            ],
        },
    ]


@torch.inference_mode()
def base_transcribe_batch(audios, max_new_tokens=64):
    convs = [_asr_conv(a) for a in audios]
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
    ).to(base_model.device)
    out = base_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=IM_END_ID,
        pad_token_id=IM_END_ID,
    )
    gen = out[:, inputs["input_ids"].shape[1] :]
    return [
        t.strip() for t in base_processor.batch_decode(gen, skip_special_tokens=True)
    ]


# ---
# GPT target generation
# ---


def gpt_json(prompt, temperature, max_tokens):
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content
        return json.loads(raw.replace("```json", "").replace("```", "").strip())
    except Exception as e:
        log(f"gpt error: {e}")
        return None


def diff(sentence, transcript):
    """
    Return lost_spans: list[str], new_toks: set[str]
    """
    s_toks = sentence_tokens(sentence)
    t_toks = sentence_tokens(transcript)
    matcher = difflib.SequenceMatcher(
        a=[_stem(t) for t in s_toks],
        b=[_stem(t) for t in t_toks],
        autojunk=False,
    )
    raw_spans, new_toks = [], set()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("delete", "replace"):
            raw_spans.append([i1, i2])
        if tag in ("replace", "insert"):
            new_toks.update(w for w in t_toks[j1:j2] if _is_content(w))
    # merge adjacent spans seperated only by matched stopwords
    # (filler bridging: contA, filler, contB -> "contA filler contB")
    merged = []
    for i1, i2 in raw_spans:
        if merged and all(not _is_content(t) for t in s_toks[merged[-1][1] : i1]):
            merged[-1][1] = i2
        else:
            merged.append([i1, i2])
    lost_spans = [
        " ".join(s_toks[i1:i2])
        for i1, i2 in merged
        if any(_is_content(t) for t in s_toks[i1: i2])
    ]
    return lost_spans, new_toks


TARGET_PROMPT = """You are writing training targets for a smart voice \
assistant that has full access to the user's apps, accounts, information, and the internet.

The user's spoken command was:
"{sentence}"

Return ONLY valid JSON in exactly this shape:
{{"answer": "<one short sentence>", "repair": "<one short question>", \
"repair_full": "<one short request>"}}

Rules for "answer": despite background chatter, the full command was heard correctly.
    - If the request asks for information (time, weather, facts) \
and you know the answer, use ONE short natural sentence that DIRECTLY answers the question with the correct fact. Otherwise, say you are looking it up, but ground the response in what was heard: refer to the request so it is clear the assistant followed everything.
    - If the request is a task request, in ONE short natural sentence, \
confirm the assistant is carrying out the request. Use present or future \
tense ("I'm setting...", "I'll remind you...")
    - never claim the action is already done.

Rules for "repair": the device heard the command over loud background chatter, \
and its speech recognition produced:
"{repair_transcript}"
The one piece it lost from the real command is "{lost_span}". Everything else \
can be treated as heard. \
Write ONE short natural question (under 20 words) that recovers ONLY that \
piece. Test: if the user replied with just the missing words, the command \
would be complete.
  - NEVER ask about parts that were heard correctly — asking about them again \
would sound like the assistant wasn't listening.
  - Ground the question in the parts heard correctly (words matching \
the original), so it is clear the assistant followed everything except this one piece.
  - Do not reveal the missing words. ONE exception: if a word was swapped \
for a similar-sounding wrong word, you may ask a confirmation question that \
offers the true word AND the misheard word as alternatives ("did you say \
saved or shared?") — never the true word alone.
  - Sound like natural speech, not a form. Vary structure freely: \
"Which...?", "How long before...?", "Who should...?", "What time...?", \
"Where...?", or a statement+question like "I lost one part — where to?". \
Do NOT default to starting with "Sorry".

Rules for "repair_full": at even louder chatter, speech recognition produced:
"{full_transcript}"
Too many pieces were lost for a targeted question. Write ONE short natural \
request (under 15 words) asking the user to repeat the whole command.
  - Do NOT reference, guess, or hint at ANY content details from either the \
real command or the garbled transcription — the assistant cannot trust any of it.
  - Mentioning the noise/chatter is fine and helps explain why.
  - Sound like natural speech and vary phrasing ("It's really loud here — \
what was that?", "I couldn't catch that over the noise, could you say it \
again?"). Do NOT default to starting with "Sorry".
"""


def generate_targets(sentence, repair_probe, full_probe, retries=3):
    """Return (answer, repair, repair_full) or (None, None, None)"""
    prompt = TARGET_PROMPT.format(
        sentence=sentence,
        repair_transcript=repair_probe["transcript"],
        lost_span=repair_probe["lost"][0],
        full_transcript=full_probe["transcript"],
    )

    answer = None
    for attempt in range(retries):
        obj = gpt_json(prompt, temperature=0.7, max_tokens=300)
        if obj is None:
            time.sleep(2**attempt)
            continue
        answer = str(obj.get("answer", "")).strip() or answer
        repair = str(obj.get("repair", "")).strip()
        repair_full = str(obj.get("repair_full", "")).strip()

        if answer and repair and repair_full:
            return answer, repair, repair_full
    return None, None, None  # skip this slurp audio


# ---
# SNR probing -> judge kind
# ---


def sample_snr(target_slots):
    weights = [SLOT_WEIGHTS[k] for k in target_slots]
    slot = random.choices(target_slots, weights=weights, k=1)[0]
    # round to 1 decimal digit
    return round(random.uniform(*SLOT_SNR[slot]), 1)


def make_probe_batch(clean, pool, slurp_id, missing, include_clean):
    """
    Build 1 batch of probe audios.
    Returns (audios, snrs) or (None, None) if failed
    """
    audios, snrs = [], []
    if include_clean:
        audios.append(clean)
        snrs.append(None)
    while len(audios) < PROBE_BATCH_SIZE:
        snr = sample_snr(missing)
        noisy = synthesize_noisy_audio(clean, pool, snr, slurp_id)
        if noisy is None:
            return None, None
        audios.append(noisy)
        snrs.append(snr)
    return audios, snrs


def probe_triplet(clean, pool, slurp_id, sentence):
    """
    Return {"answer": ..., "repair": ..., "repair_full": ...}
    or None if the budget ran out.

    Each round synthesize PROBE_BATCH_SIZE probes.
    The first round always include a clean audio as a sanity check.
    """
    results = {"answer": None, "repair": None, "repair_full": None}
    for batch_idx in range(MAX_PROBES):
        missing = [k for k, v in results.items() if v is None]
        if not missing:
            break
        audios, snrs = make_probe_batch(
            clean, pool, slurp_id, missing, include_clean=(batch_idx == 0)
        )
        if audios is None:
            # clean or pool is quiet
            return None
        transcripts = base_transcribe_batch(audios)
        for snr, noisy, transcript in reversed(list(zip(snrs, audios, transcripts))):
            lost, untrusted = diff(sentence, transcript)
            if snr is None and (lost or untrusted):
                # model can't transcribe clean audio correctly
                # -> skip bad recording
                return None
            kind = kind_of(lost, untrusted, sentence)
            if kind is None:
                continue
            if results[kind] is None:
                results[kind] = {
                    "snr_db": snr,
                    "audio": noisy,
                    "transcript": transcript,
                    "lost": lost,
                    "untrusted": sorted(untrusted)
                }

    if any(v is None for v in results.values()):
        return None
    return results


# ---
# Main triplet-building loop
# ---


def collect_existing_ids(repo_id):
    ids = set()
    for split in ("train", "test"):
        ds = load_dataset(repo_id, split=split, streaming=True)
        ds = ds.select_columns(["slurp_id"])
        for r in ds:
            ids.add(r["slurp_id"])

    return ids


def build_triplets(source_split, n_triplets, tag, seen_slurp_ids):
    pool = collect_babble_pool(source_split, BABBLE_POOL_SIZE)

    stream = load_dataset("qmeeus/slurp", split=source_split, streaming=True)
    stream = stream.cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE))

    rows, scanned, done = [], 0, 0
    skip.clear()
    pbar = tqdm(total=n_triplets, desc=f"[{tag}]", unit="triplet", dynamic_ncols=True)
    for row in stream:
        if done >= n_triplets:
            break
        scanned += 1
        pbar.set_postfix(scanned=scanned, **skip, refresh=False)

        slurp_id = row["slurp_id"]
        sentence = row["sentence"]
        if slurp_id in seen_slurp_ids or len(sentence.split()) < 4:
            skip["seen/short"] += 1
            continue

        clean = row["audio"]["array"].astype(np.float32)
        clean = clean[: MAX_AUDIO_SECONDS * AUDIO_SAMPLING_RATE]

        triplet = probe_triplet(clean, pool, slurp_id, sentence)
        if triplet is None:
            skip["probe"] += 1
            continue

        answer_target, repair_target, repair_full_target = generate_targets(
            sentence, triplet["repair"], triplet["repair_full"]
        )
        if not answer_target:
            skip["targets"] += 1
            continue
        targets = {
            "answer": answer_target,
            "repair": repair_target,
            "repair_full": repair_full_target,
        }

        seen_slurp_ids.add(slurp_id)
        for kind in ("answer", "repair", "repair_full"):
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
                    "lost": probe["lost"],
                    "untrusted": probe["untrusted"],
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
    seen = collect_existing_ids(MASK_DS_ID)
    test_rows = build_triplets("test", N_TEST_TRIPLETS, "test", seen)
    train_rows = build_triplets("train", N_TRAIN_TRIPLETS, "train", seen)

    def to_hf(rows):
        return Dataset.from_list(rows).cast_column(
            "audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE)
        )

    DatasetDict({"train": to_hf(train_rows), "test": to_hf(test_rows)}).push_to_hub(
        REPO_ID
    )

    log(
        f"Pushed {len(train_rows)} train / {len(test_rows)} test rows " f"to {REPO_ID}."
    )
