"""
Build SFT slurp dataset for EAR conversational-repair training.

Word timestamps come from Qwen3-ForcedAligner-0.6B, force-aligning the
ground-truth SLURP sentence (no ASR, no Whisper).

Because the SLURP annotation tags entities verbatim inside the sentence,
phrase matching is done at the token-index level on the sentence text.
Answerable/unanswerable feasibility is therefore decided BEFORE running
the aligner or GPT, and sentences that can never work (no maskable
stopword outside the critical phrase) are skipped across all recordings.

Requires transformers from source until Qwen3-ASR lands in a release:
    pip install git+https://github.com/huggingface/transformers
"""

import json
import os
import random
import re
import time
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoProcessor
from openai import OpenAI
import itertools

def log(*args):
    """print-compatible logging that doesn't break the tqdm bar."""
    tqdm.write(" ".join(str(a) for a in args))

AUDIO_SAMPLING_RATE = 16000
N_TRAIN_PAIRS = 1000
N_TEST_PAIRS = 50
ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
ALIGNER_LANGUAGE = "English"
MIN_MASK_SEC = 0.10  # aligner grid is ~0.08s; guard against zero-width words
OPENAI_MODEL = "gpt-4o"
REPO_ID = "keylazy/slurp-ear-sft"
OUT_DIR = "masked_audio"
SEED = 42
ROW_ID = itertools.count(1)

STOPWORDS = [
    "the",
    "a",
    "an",
    "is",
    "for",
    "to",
    "please",
    "can",
    "you",
    "me",
    "my",
    "of",
    "in",
]

SLOT_DESCRIPTIONS = {
    "transport_agency": "ride service or company",
    "event_name": "event or type of event",
    "media_type": "news source or social media platform",
    "date": "date",
    "time": "time",
    "timeofday": "time of day",
    "person": "person's name",
    "relation": "relationship (e.g. mom, boss)",
    "place_name": "place name",
    "business_name": "business name",
    "news_topic": "news topic",
    "music_genre": "music genre",
    "artist_name": "artist name",
    "song_name": "song name",
    "playlist_name": "playlist name",
    "podcast_name": "podcast name",
    "podcast_descriptor": "podcast description",
    "radio_name": "radio station name",
    "audiobook_name": "audiobook name",
    "device_type": "device name",
    "house_place": "room in the house",
    "food_type": "food item",
    "drink_type": "drink",
    "meal_type": "meal (e.g. breakfast, dinner)",
    "weather_descriptor": "weather detail",
    "currency_name": "currency",
    "transport_type": "mode of transport",
    "transport_name": "transport line or service name",
    "list_name": "list name",
    "email_address": "email address",
    "email_folder": "email folder",
    "app_name": "app name",
    "game_name": "game name",
    "movie_name": "movie name",
    "movie_type": "movie genre",
    "coffee_type": "type of coffee",
    "cooking_type": "cooking method",
    "ingredient": "ingredient",
    "joke_type": "type of joke",
    "color_type": "color",
    "change_amount": "amount to change",
    "definition_word": "word to define",
    "general_frequency": "how often",
    "time_zone": "time zone",
    "alarm_type": "alarm name",
    "order_type": "order type (e.g. takeaway)",
    "business_type": "type of business",
    "personal_info": "contact detail (e.g. phone number, address)",
    "player_setting": "playback setting",
}


def slot_description(slot_type):
    return SLOT_DESCRIPTIONS.get(slot_type, slot_type.replace("_", " "))


random.seed(SEED)
np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

client = OpenAI()

aligner_processor = AutoProcessor.from_pretrained(ALIGNER_MODEL_ID)
aligner_model = AutoModelForTokenClassification.from_pretrained(
    ALIGNER_MODEL_ID, dtype=torch.bfloat16, device_map="auto"
).eval()
print("models loaded")


# ---------------------------------------------------------------------------
# Token-level matching on the ground-truth sentence.
# The SLURP annotation tags entities verbatim in the sentence, so a phrase is
# always a contiguous token subsequence; no fuzzy/char-level matching needed.
# ---------------------------------------------------------------------------


def norm_token(tok):
    return re.sub(r"[^a-z0-9']", "", tok.lower())


def sentence_tokens(sentence):
    return [norm_token(t) for t in sentence.split()]


def find_phrase_indices(tokens, phrase):
    """Return [start, end) token index range of `phrase` in `tokens`, or None."""
    phrase_toks = [norm_token(t) for t in phrase.split()]
    n = len(phrase_toks)
    if n == 0:
        return None
    for i in range(len(tokens) - n + 1):
        if tokens[i : i + n] == phrase_toks:
            return (i, i + n)
    return None


def pick_stopword_index(tokens, exclude):
    """Random index of a stopword token outside the [start, end) exclude range."""
    candidates = [
        i
        for i, tok in enumerate(tokens)
        if tok in STOPWORDS and not (exclude[0] <= i < exclude[1])
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def plan_masks(sentence, critical_phrase):
    """
    Text-only feasibility check. Returns (crit_idx_range, stopword_idx)
    or (None, None) if this sentence can never produce a pair.
    """
    tokens = sentence_tokens(sentence)
    crit_range = find_phrase_indices(tokens, critical_phrase)
    if not crit_range:
        return None, None
    stop_idx = pick_stopword_index(tokens, exclude=crit_range)
    if stop_idx is None:
        return None, None
    return crit_range, stop_idx


# ---------------------------------------------------------------------------
# Forced alignment
# ---------------------------------------------------------------------------


def align_words(audio, transcript):
    """
    Force-align `transcript` against `audio` (float32 mono @ 16 kHz) using
    Qwen3-ForcedAligner. Returns [{"word", "start", "end"}, ...] or None.
    """
    try:
        aligner_inputs, word_lists = aligner_processor.prepare_forced_aligner_inputs(
            audio=audio,
            transcript=transcript,
            language=ALIGNER_LANGUAGE,
        )
        aligner_inputs = aligner_inputs.to(aligner_model.device, aligner_model.dtype)

        with torch.inference_mode():
            outputs = aligner_model(**aligner_inputs)

        timestamps = aligner_processor.decode_forced_alignment(
            logits=outputs.logits,
            input_ids=aligner_inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=aligner_model.config.timestamp_token_id,
        )[0]
    except Exception as e:
        log(f"forced alignment error: {e}")
        return None

    return [
        {"word": item["text"], "start": item["start_time"], "end": item["end_time"]}
        for item in timestamps
    ]


def indices_to_span(words, idx_range):
    """Convert a [start, end) word-index range into a (t_start, t_end) span."""
    return (words[idx_range[0]]["start"], words[idx_range[1] - 1]["end"])


def extract_critical_targets(annotation, sentence):
    """return (slot_type, critical_phrase) or (None, None)"""

    entities = re.findall(r"\[(.*?) : (.*?)\]", annotation)
    if not entities:
        return None, None

    slot_type, critical_phrase = (s.strip().lower() for s in random.choice(entities))

    return slot_type, critical_phrase


def inject_noise(audio, sr, span, pad=0.05):
    start = int((span[0] - pad) * sr)
    end = int((span[1] + pad) * sr)

    # Aligner timestamps sit on a coarse (~0.08s) grid; short words can come
    # back with zero width. Enforce a minimum mask length, centered on span.
    min_len = int(MIN_MASK_SEC * sr)
    if end - start < min_len:
        center = (start + end) // 2
        start = center - min_len // 2
        end = start + min_len

    start = max(0, start)
    end = min(len(audio), end)
    if start >= end:
        return None
    noisy = audio.copy()
    amp = max(np.mean(np.abs(noisy[start:end])), 0.01)
    noisy[start:end] = np.random.normal(0, amp, end - start)
    return noisy


def mask_variants(tag, slurp_id, audio_dict, sentence, crit_range, stop_idx):
    """return (ans_path, una_path) or (None, None)"""

    audio = audio_dict["array"].astype(np.float32)
    sr = audio_dict["sampling_rate"]
    if sr != AUDIO_SAMPLING_RATE:
        return None, None

    words = align_words(audio, sentence)
    if not words:
        log(f"\n[{tag}:{slurp_id}] forced alignment failed, skipping")
        log("sentence", sentence)
        return None, None

    if len(words) != len(sentence.split()):
        # Aligner tokenized differently than whitespace split; indices would
        # not line up. Rare for English; skip to stay safe.
        log(f"\n[{tag}:{slurp_id}] token count mismatch, skipping")
        log("sentence", sentence)
        log("words", [w["word"] for w in words])
        return None, None

    crit_span = indices_to_span(words, crit_range)
    non_crit_span = indices_to_span(words, (stop_idx, stop_idx + 1))

    unans = inject_noise(audio, sr, crit_span, pad=0.0)
    ans = inject_noise(audio, sr, non_crit_span, pad=0.0)

    if unans is None or ans is None:
        log(f"[{tag}:{slurp_id}] empty noise span, skipping")
        return None, None

    ans_path = os.path.join(OUT_DIR, f"{tag}_{slurp_id}_answerable.wav")
    unans_path = os.path.join(OUT_DIR, f"{tag}_{slurp_id}_unanswerable.wav")
    sf.write(ans_path, ans, sr)
    sf.write(unans_path, unans, sr)
    return ans_path, unans_path


def masked_sentence(sentence, crit_phrase):
    pat = re.compile(re.escape(crit_phrase), re.IGNORECASE)
    return pat.sub("[GARBLED]", sentence, count=1)


TARGET_PROMPT = """You are writing training targets for a smart voice \
assistant that has full access to the user's apps, accounts, information, and the internet.

The user's spoken command was:
"{sentence}"

Return ONLY valid JSON in exactly this shape:
{{"answer": "<one short sentence>", "repair": "<one short question>"}}

Rules for "answer": The full user command is heard by the voice assistant.
    - If the request asks for information (time, weather, facts) \
and you know the asnwer, use ONE short natural sentence that DIRECTLY answers the question with the correct fact. Otherwise, say you are looking it up, but ground the response in what was heard: refer the to request so it is clear the assistant followed everything.
    - If the request is a task request, in ONE short natural sentence, \
confirming the assistant is carrying out the request. Use present or future \
tense ("I'm setting...", "I'll remind you...")
    - never claim the action is already done. 

Rules for "repair": The device actually heard:
"{masked}"
where [GARBLED] marks the ONE segment that was lost. Everything else was \
heard clearly. In ONE short natural question (under 20 words) that recovers ONLY \
the [GARBLED] segment ({slot_desc}). Test: if the user replied with just the \
missing words, the command would be complete.
  - NEVER ask about anything outside [GARBLED] — those parts were heard, and \
asking about them again would sound like the assistant wasn't listening.
  - Ground the question in what WAS heard: reference the surrounding context \
so it is clear the assistant followed everything except this one piece.
  - NEVER guess, mention, or hint at the lost words themselves.
  - Sound like natural speech, not a form. Vary structure freely: \
"Which...?", "How long before...?", "Who should...?", "What time...?", \
"Where...?", or a statement+question like "I lost one part — where to?". \
Do NOT default to starting with "Sorry".
"""

GENERIC_WORDS = {
    "city",
    "town",
    "country",
    "place",
    "room",
    "list",
    "name",
    "day",
    "date",
    "time",
    "week",
    "list",
    "song",
    "game",
    "meeting",
    "event",
    "news",
    "weather",
    "email",
    "reminder",
}


def leaks(masked_phrase, text):
    """
    return True if the repair text contains the phrase
    """

    text_words = set(re.findall(r"[a-z']+", text.lower()))
    masked_words = {
        w
        for w in re.findall(r"[a-z']+", masked_phrase.lower())
        if w not in STOPWORDS and w not in GENERIC_WORDS and len(w) > 2
    }
    return bool(masked_words & text_words)


def generate_targets(sentence, slot_type, masked_phrase, retries=3):
    """
    Return (answer, repair)
    """

    slot_desc = slot_description(slot_type)
    prompt = TARGET_PROMPT.format(
        sentence=sentence,
        slot_desc=slot_desc,
        masked=masked_sentence(sentence, masked_phrase),
    )

    answer = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.7,
                max_tokens=200,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content
            obj = json.loads(raw.replace("```json", "").replace("```", "").strip())
            answer = obj["answer"].strip() or answer
            repair = obj["repair"].strip()
            if answer and repair and not leaks(masked_phrase, repair):
                return answer, repair
        except Exception as e:
            log(f"target generation error (attempt {attempt + 1}): {e}")
            time.sleep(2**attempt)

    if answer:
        log("Fallback! slot_desc:", slot_desc)
        return answer, (
            f"Sorry, I didn't catch the {slot_desc} — "
            f"could you say that part again?"
        )
    return None, None


def build_pairs(source_split, n_pairs, tag, seen_slurp_ids):
    stream = load_dataset("qmeeus/slurp", split=source_split, streaming=True)
    rows, scanned, done = [], 0, 0
    pbar = tqdm(total=n_pairs, desc=f"[{tag}]", unit="pair", dynamic_ncols=True)
    for row in stream:
        if done >= n_pairs:
            break
        scanned += 1
        pbar.set_postfix(scanned=scanned, refresh=False)

        slurp_id = row["slurp_id"]
        sentence = row["sentence"]
        if slurp_id in seen_slurp_ids or len(sentence.split()) < 4:
            continue

        slot_type, crit_phrase = extract_critical_targets(row["annotation"], sentence)
        if not crit_phrase or (len(crit_phrase.split()) / len(sentence.split())) > 0.5:
            continue

        # Text-only feasibility check: runs before alignment and GPT.
        # If it fails, it fails for every recording of this sentence, so
        # mark the slurp_id as seen to avoid re-trying other recordings.
        crit_range, stop_idx = plan_masks(sentence, crit_phrase)
        if crit_range is None:
            seen_slurp_ids.add(slurp_id)
            continue

        ans_path, unans_path = mask_variants(
            tag, slurp_id, row["audio"], sentence, crit_range, stop_idx
        )
        if not ans_path:
            continue

        answer_target, repair_target = generate_targets(
            sentence, slot_type, crit_phrase
        )
        if not answer_target:
            continue

        seen_slurp_ids.add(slurp_id)
        common = {
            "slurp_id": slurp_id,
            "slot_type": slot_type,
            "crit_phrase": crit_phrase,
            "sentence": sentence,
        }
        rows.append(
            {
                "id": next(ROW_ID),
                "kind": "answer",
                "target": answer_target,
                "audio": ans_path,
                **common,
            }
        )
        rows.append(
            {
                "id": next(ROW_ID),
                "kind": "repair",
                "target": repair_target,
                "audio": unans_path,
                **common,
            }
        )

        done += 1
        pbar.update(1)

    pbar.close()
    log(
        f"[{tag}] built {len(rows)} rows from {done} utterances " f"({scanned} scanned)"
    )
    return rows


if __name__ == "__main__":
    seen = set()
    test_rows = build_pairs("test", N_TEST_PAIRS, "test", seen)
    train_rows = build_pairs("train", N_TRAIN_PAIRS, "train", seen)

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