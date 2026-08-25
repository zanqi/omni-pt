"""
Build the mask-track dataset (steps/mask.html).

Every row carries exactly one masked span, and the mask is the only thing that
removes information:

  kind="repair"  the top-priority SLURP slot is masked -> ask about it
  kind="answer"  a non-critical span (stopword, or a rank-5 slot) is masked -> just act

Both rows of an utterance share a background, drawn once per utterance:
3-speaker babble at 0-20 dB SNR, or (CLEAN_BG_PROB of the time) the raw clean
recording. There is no probe loop and no classifier -- the mask is placed by us
over a span we chose, so the kind is true by construction. What replaces the
probe is a whisper-tiny gate: the background alone must transcribe with nothing
lost before any mask goes on, and the finished audio must transcribe with the
masked phrase gone and everything else still there. That is what makes
"exactly one piece is missing" a fact about the audio rather than a hope.

Four masks, ordered by how much signal they leave behind: silence (none),
white (noise), splice (the speaker's own audio shredded into sub-phoneme
chunks), burst (the real word, buried under babble at -10 dB local SNR).

Runs in the qwen3omni env: that is where transformers is new enough to carry
Qwen3ASRProcessor.prepare_forced_aligner_inputs. No omni model is loaded -- the
0.6B aligner and whisper-tiny are the only GPU residents. Targets are written by
babble_data.write_target(), served by the same vLLM box as the other tracks.

    python mask_data.py --ds-id keylazy/slurp-mask-v1 --n-train 1500 --n-test 80
"""

import argparse
import itertools
import json
import os
import random
import re
import shutil
import threading
from difflib import SequenceMatcher
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoProcessor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

import babble_data
from babble_data import (
    AUDIO_SAMPLING_RATE,
    BABBLE_SPEAKERS,
    MAX_AUDIO_SECONDS,
    NON_PIECE_WORDS,
    UTTERANCE_WORKERS,
    collect_babble_pool,
    imap_ordered,
    log,
    skip,
    slurp_ds_stream,
    write_target,
)

ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
ALIGNER_LANGUAGE = "English"
WHISPER_MODEL_ID = "openai/whisper-small.en"  # set in __main__ from --asr-model
WHISPER_MAX_NEW_TOKENS = 64
# a hypothesis token this close to a sentence word counts as that word
WORD_MATCH_RATIO = 0.8

# test rows must not repeat a sentence either track already trained on, so the
# sent/ear adapters can be scored on this split. Train ids are not excluded:
# this track supersedes those datasets rather than running beside them.
EXCLUDE_DS_IDS = (
    "keylazy/slurp-ear-sft",
    "keylazy/slurp-babble-Qwen2.5-Omni-3B-v3",
)

AUDIO_ROOT = "babble_audio"
AUDIO_DIR = None  # set in __main__ from --ds-id
SEED = 42
ROW_ID = itertools.count(1)

N_TRAIN = 1000
N_TEST = 50

# noise-free audio is the common case in deployment, and a masked span must not
# only ever appear in noise
CLEAN_BG_PROB = 0.25
SNR_BAND = (0.0, 20.0)
# 0 dB babble is not reliably survivable, so the low end of the band leans on
# the gate: an utterance that loses a word gets a redrawn mixture, then a
# louder one. What ships in snr_db is therefore a gate-conditioned draw.
BED_SNR_STEP = 4.0
BED_MAX_RETRIES = 5

MASK_PAD = 0.05  # set in __main__ from --mask-pad
MIN_MASK_SEC = 0.10  # aligner grid is ~0.08s; guard against zero-width words
VARIANTS = ("silence", "white", "splice", "burst")
SILENCE_RAMP_SEC = 0.02
BURST_RAMP_SEC = 0.05
BURST_SNR_DB = -10.0
# fix the chunk length, not the count: spans run 0.15-0.6s, so a constant N
# would give a 3-syllable name chunks four times longer than a short one. 40ms
# is under a phoneme and over a pitch period -- voiced-sounding, wordless.
SPLICE_CHUNK_SEC = 0.04
SPLICE_XFADE_SEC = 0.005
SPLICE_MIN_CHUNKS = 4

SILENCE_HARD = False  # set in __main__ from --silence-hard

STOPWORDS = {
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
}

# What the command cannot be acted on without, most-essential first. Rank 5 is
# everything unlisted: category-level words the command usually survives.
SLOT_RANKS = {
    1: (
        "person",
        "relation",
        "business_name",
        "place_name",
        "app_name",
        "device_type",
        "email_address",
        "personal_info",
        "artist_name",
    ),
    2: ("date", "time", "timeofday", "general_frequency", "time_zone"),
    3: (
        "change_amount",
        "currency_name",
        "order_type",
        "player_setting",
        "color_type",
    ),
    4: (
        "song_name",
        "playlist_name",
        "podcast_name",
        "radio_name",
        "audiobook_name",
        "movie_name",
        "game_name",
        "news_topic",
        "event_name",
        "list_name",
        "definition_word",
        "food_type",
        "drink_type",
        "ingredient",
        "coffee_type",
    ),
}
SLOT_RANK = {slot: rank for rank, slots in SLOT_RANKS.items() for slot in slots}
LOWEST_RANK = 5

random.seed(SEED)
np.random.seed(SEED)

GPU_LOCK = threading.Lock()
# loaded in __main__
aligner_model = None
aligner_processor = None
whisper_model = None
whisper_processor = None


# ---
# text helpers -- one normalization shared by the slot matching and both gates
# ---


# whisper writes "three days" as "3 days" and "ten am" as "10 am", and the
# sentence side always spells them out. Without this the bed gate reads a
# correct transcript as having lost a content word.
NUM_WORDS = {
    w: str(i)
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}
NUM_WORDS.update(
    {"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60",
     "seventy": "70", "eighty": "80", "ninety": "90", "hundred": "100"}
)


def norm_tokens(s):
    return [
        NUM_WORDS.get(t, t)
        for t in (re.sub(r"[^a-z0-9']", "", w.lower()) for w in s.split())
        if t
    ]


def count_phrase(tokens, phrase_toks):
    """Non-overlapping occurrences of `phrase_toks` inside `tokens`."""
    n = len(phrase_toks)
    if not n:
        return 0
    hits, i = 0, 0
    while i <= len(tokens) - n:
        if tokens[i : i + n] == phrase_toks:
            hits += 1
            i += n
        else:
            i += 1
    return hits


def word_hits(hyp_toks, word):
    """How many hypothesis tokens are this word, allowing for ASR spelling.

    whisper-tiny writes "divya" as "deepya" and "olly" as "ollie". Scoring
    those as missing would reject a bed that lost nothing and quietly bias the
    dataset toward sentences a 39M-parameter ASR happens to spell right, so a
    close-enough token counts as heard.
    """
    return sum(
        1
        for t in hyp_toks
        if t == word or SequenceMatcher(None, t, word).ratio() >= WORD_MATCH_RATIO
    )


def gate(hyp, sentence, must_hear=(), must_lose=None):
    """Did this audio keep everything but `must_lose`? "" if so, else why not.

    Both gates ask the same question of a whisper-tiny hypothesis, from
    opposite sides: before masking `must_lose` is None and every content word
    has to survive; after masking the masked phrase has to be gone *and*
    everything outside it still there. Function words are deliberately not
    required -- whisper-tiny drops "the"/"a" on clean audio too, and failing on
    that would reject good beds. A phrase in `must_hear` is checked whatever it
    is made of, which is how a stopword span still gets verified.

    The three failures are worth telling apart: "heard" means the mask did not
    work, "collateral" means it took a neighbour with it, "inaudible" means the
    span was never there to mask.
    """
    hyp_toks = norm_tokens(hyp)
    sent_toks = norm_tokens(sentence)
    lose_toks = norm_tokens(must_lose) if must_lose else []
    sent_count, lose_count = Counter(sent_toks), Counter(lose_toks)

    if lose_toks:
        if count_phrase(hyp_toks, lose_toks) >= count_phrase(sent_toks, lose_toks):
            return "heard"
        # the phrase check alone passes when only part of a multi-word name
        # survives ("john smith" -> "john"), which is exactly the two-losses
        # row this gate exists to reject
        for word in lose_count:
            if word not in NON_PIECE_WORDS and word_hits(hyp_toks, word) >= sent_count[word]:
                return "heard"

    for word, n in sent_count.items():
        if word in NON_PIECE_WORDS:
            continue
        if word_hits(hyp_toks, word) < n - lose_count[word]:
            return "collateral"

    for phrase in must_hear:
        toks = norm_tokens(phrase)
        if count_phrase(hyp_toks, toks) < count_phrase(sent_toks, toks) and any(
            word_hits(hyp_toks, w) < sent_count[w] for w in toks
        ):
            return "inaudible"
    return ""


def ramp(n):
    """Raised-cosine 0 -> 1 over n samples; reverse it for a fade-out."""
    if n <= 0:
        return np.ones(0, dtype=np.float32)
    return (0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, n)))).astype(np.float32)


# ---
# GPU passes
# ---


def whisper_asr(arrays):
    """Greedy whisper-tiny over a batch of float32 @ 16 kHz. Returns texts."""
    inputs = whisper_processor(
        arrays, sampling_rate=AUDIO_SAMPLING_RATE, return_tensors="pt"
    )
    features = inputs.input_features.to(whisper_model.device, whisper_model.dtype)
    with GPU_LOCK, torch.inference_mode():
        ids = whisper_model.generate(
            features, max_new_tokens=WHISPER_MAX_NEW_TOKENS, num_beams=1
        )
    return [t.strip() for t in whisper_processor.batch_decode(ids, skip_special_tokens=True)]


def align_words(audio, sentence):
    """Force-align `sentence` onto clean audio. [{"word","start","end"}] or None.

    Runs on the clean signal, before any mixing: the aligner does this
    measurably better without babble on top, which matters more now that the
    bed can sit at 0 dB. The timestamps transfer to the mix unchanged, because
    every mask is applied in place and nothing shifts.
    """
    try:
        inputs, word_lists = aligner_processor.prepare_forced_aligner_inputs(
            audio=audio, transcript=sentence, language=ALIGNER_LANGUAGE
        )
        inputs = inputs.to(aligner_model.device, aligner_model.dtype)
        with GPU_LOCK, torch.inference_mode():
            outputs = aligner_model(**inputs)
        stamps = aligner_processor.decode_forced_alignment(
            logits=outputs.logits,
            input_ids=inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=aligner_model.config.timestamp_token_id,
        )[0]
    except Exception as e:
        log(f"forced alignment error: {e}")
        return None

    words = [
        {"word": w["text"], "start": w["start_time"], "end": w["end_time"]}
        for w in stamps
    ]
    if len(words) != len(sentence.split()):
        # aligner tokenized differently than the whitespace split, so token
        # indices would not line up. Rare for English; skip to stay safe.
        return None
    return words


# ---
# per-utterance pipeline: pick spans -> background -> masks
# ---


def plan_spans(sentence, annotation):
    """Text-only: which span each kind masks. None if the sentence can't work.

    Feasibility is decided before any GPU runs, as in slurp_sft_data_qwen.py --
    the SLURP annotation tags entities verbatim in the sentence, so a phrase is
    always a contiguous token subsequence and matching needs no audio.
    """
    # positional, unlike norm_tokens(): a token that normalizes to nothing has
    # to keep its slot, because these indices address the aligner's word list
    tokens = [re.sub(r"[^a-z0-9']", "", w.lower()) for w in sentence.split()]

    def indices(phrase):
        phrase_toks = norm_tokens(phrase)
        n = len(phrase_toks)
        if not n:
            return None
        for i in range(len(tokens) - n + 1):
            if tokens[i : i + n] == phrase_toks:
                return (i, i + n)
        return None

    slots = []
    for slot_type, phrase in re.findall(r"\[(.*?) : (.*?)\]", annotation):
        slot_type, phrase = slot_type.strip().lower(), phrase.strip()
        span = indices(phrase)
        if span:
            slots.append({"range": span, "phrase": phrase, "slot": slot_type})

    # "send a mail to my friend divya" tags both relation=friend and
    # person=divya, and the command survives losing "friend" but not "divya".
    # relation is a rank-1 target on its own ("call my mom") and a rank-5
    # modifier next to a name -- the one cross-slot rule in the table.
    has_person = any(s["slot"] == "person" for s in slots)
    for s in slots:
        rank = SLOT_RANK.get(s["slot"], LOWEST_RANK)
        if s["slot"] == "relation" and has_person:
            rank = LOWEST_RANK
        s["rank"] = rank

    # best slot: lowest rank, then longest span, then leftmost
    crit = min(
        slots,
        key=lambda s: (s["rank"], -(s["range"][1] - s["range"][0]), s["range"][0]),
        default=None,
    )
    if crit is None or crit["rank"] >= LOWEST_RANK:
        return None

    # answer span: a stopword outside every slot, longest first -- a short mask
    # is nearly an unmasked row and teaches little. Rank-5 slots are the
    # fallback for sentences with no free stopword.
    taken = [s["range"] for s in slots]
    free = [
        i
        for i, tok in enumerate(tokens)
        if tok in STOPWORDS and not any(a <= i < b for a, b in taken)
    ]
    ans = [
        {"range": (i, i + 1), "phrase": tokens[i], "slot": "", "rank": 0}
        for i in sorted(free, key=lambda i: (-len(tokens[i]), i))
    ]
    ans += [s for s in slots if s["rank"] >= LOWEST_RANK and s is not crit]
    if not ans:
        return None
    return {"crit": crit, "ans": ans}


def build_bed(clean, pool, plan, sentence, rng):
    """Draw a background and verify with whisper-tiny that nothing is lost yet.

    Returns (bg, reason). On success `bg` carries the background split into the
    two parts every mask needs (bed == speech + bg_only), the SNR that
    survived, the hypothesis, and the answer spans that were audible in it.
    """
    length = len(clean)
    clean_power = float(np.mean(clean**2))

    def babble_mixture():
        mix = np.zeros(length, dtype=np.float32)
        for clip in rng.sample(pool, BABBLE_SPEAKERS):
            if len(clip) < length:
                clip = np.pad(clip, (0, length - len(clip)), "wrap")
            else:
                start = rng.randint(0, len(clip) - length)
                clip = clip[start : start + length]
            mix += clip
        return mix / BABBLE_SPEAKERS

    clean_bg = rng.random() < CLEAN_BG_PROB
    base_snr = round(rng.uniform(*SNR_BAND), 1)

    for attempt in range(1 if clean_bg else BED_MAX_RETRIES):
        babble = babble_mixture()
        if clean_bg:
            snr, speech, bg_only = None, clean, np.zeros(length, dtype=np.float32)
        else:
            # a failure at 4 dB is often an unlucky mixture rather than a
            # too-low SNR, so the first retry redraws the clips before the
            # band moves
            snr = round(min(base_snr + BED_SNR_STEP * max(0, attempt - 1), SNR_BAND[1]), 1)
            # SNR = 10*log10(clean_power / babble_power)
            scale = np.sqrt(clean_power / (10 ** (snr / 10)) / float(np.mean(babble**2)))
            speech, bg_only = clean, scale * babble
        bed = speech + bg_only
        peak = float(np.max(np.abs(bed)))
        if peak > 1.0:
            # avoid clipping on save; rescaling does not change SNR, and the
            # two parts have to keep summing to what ships
            bed, speech, bg_only = bed / peak, speech / peak, bg_only / peak

        hyp = whisper_asr([bed.astype(np.float32)])[0]
        reason = gate(hyp, sentence, must_hear=[plan["crit"]["phrase"]])
        if reason:
            log(
                f"[bed:{reason}] snr={snr} crit={plan['crit']['phrase']!r}\n"
                f"    said: {sentence}\n"
                f"    bed:  {hyp}"
            )
            continue

        # the crit span came through; now take the first answer span whisper
        # can actually hear. Falling through the candidates costs no new audio,
        # so it runs against the hypothesis already in hand rather than raising
        # the SNR.
        ans = next(
            (c for c in plan["ans"] if not gate(hyp, sentence, must_hear=[c["phrase"]])),
            None,
        )
        if ans is None:
            reason = "no-ans-span"
            continue
        return {
            "bed": bed.astype(np.float32),
            "speech": speech.astype(np.float32),
            "bg_only": bg_only.astype(np.float32),
            "babble": babble.astype(np.float32),
            "snr": snr,
            "hyp": hyp,
            "ans": ans,
        }, None

    return None, f"bed-clean-{reason}" if clean_bg else f"bed-{reason}"


def apply_mask(parts, span, variant, rng):
    """Replace `span` (t_start, t_end) of the background in place.

    All four masks keep the total duration, so nothing after the span shifts
    and the alignment stays valid for the rest of the utterance.
    """
    sr = AUDIO_SAMPLING_RATE
    bed = parts["bed"]
    start = int((span[0] - MASK_PAD) * sr)
    end = int((span[1] + MASK_PAD) * sr)
    # aligner timestamps sit on a coarse (~0.08s) grid; short words can come
    # back with zero width. Enforce a minimum, centered on the span.
    if end - start < int(MIN_MASK_SEC * sr):
        center = (start + end) // 2
        start = center - int(MIN_MASK_SEC * sr) // 2
        end = start + int(MIN_MASK_SEC * sr)
    start, end = max(0, start), min(len(bed), end)
    if start >= end:
        return None, {}

    out = bed.copy()
    n = end - start
    meta = {
        "mask_snr_db": None,
        "splice_n": None,
        "mask_start": round(start / sr, 3),
        "mask_end": round(end / sr, 3),
    }

    if variant == "silence":
        # the speaker stops, the room does not: the span keeps the bed alone at
        # its existing level. --silence-hard zeroes the mixed signal outright.
        target = (
            np.zeros(n, dtype=np.float32) if SILENCE_HARD else parts["bg_only"][start:end]
        )
        # weight of the ORIGINAL audio: 1 at the two edges so the transition
        # has no click, 0 through the middle so the word is actually gone
        edge = min(int(SILENCE_RAMP_SEC * sr), n // 2)
        blend = np.zeros(n, dtype=np.float32)
        blend[:edge] = 1 - ramp(edge)
        blend[n - edge :] = 1 - ramp(edge)[::-1]
        out[start:end] = blend * bed[start:end] + (1 - blend) * target

    elif variant == "white":
        # EAR track's mask, but the amplitude is read off the mixed signal so
        # the burst matches the bed's level rather than the clean speech's
        amp = max(float(np.mean(np.abs(bed[start:end]))), 0.01)
        out[start:end] = np.random.default_rng(rng.getrandbits(64)).normal(
            0, amp, n
        ).astype(np.float32)

    elif variant == "splice":
        # Built on the main speaker alone, then the babble goes back on top:
        # shred the clean signal, drop it in place of the key span, and add
        # bg_only afterwards. Cutting the mixture instead would chop the babble
        # into the same 40 ms pieces, and a bed that stutters exactly where the
        # word went is a cue the model could key on without hearing anything.
        src = parts["speech"]
        chunk = max(int(SPLICE_CHUNK_SEC * sr), 1)
        count = max(SPLICE_MIN_CHUNKS, round(n / chunk))
        chunk = n // count
        if chunk < 2:
            return None, {}
        xfade = min(int(SPLICE_XFADE_SEC * sr), chunk // 4)
        # offsets come from this utterance's own speech, outside the span, so
        # the fragments carry the same voice and room as the rest of the row
        pool = [i for i in range(0, len(src) - chunk) if i + chunk <= start or i >= end]
        if not pool:
            return None, {}
        picks, prev = [], None
        for _ in range(count):
            # two consecutive chunks lifted from adjacent offsets would rebuild
            # a real word, which is the one thing this mask must not do
            off = rng.choice(pool)
            for _ in range(4):
                if prev is None or abs(off - (prev + chunk)) >= chunk:
                    break
                off = rng.choice(pool)
            picks.append(off)
            prev = off
        shred = np.concatenate([src[o : o + chunk] for o in picks])
        if xfade:
            fade = ramp(xfade)
            for i in range(1, count):
                edge = i * chunk
                shred[edge : edge + xfade] = (
                    shred[edge : edge + xfade] * fade
                    + src[picks[i - 1] + chunk - xfade : picks[i - 1] + chunk] * (1 - fade)
                )
        shred = np.pad(shred, (0, n - len(shred)), "wrap")[:n]
        rms = float(np.sqrt(np.mean(src[start:end] ** 2)))
        shred_rms = max(float(np.sqrt(np.mean(shred**2))), 1e-6)
        shred = shred * (rms / shred_rms)
        edge = min(xfade * 4, n // 2)
        blend = np.ones(n, dtype=np.float32)
        blend[:edge] = ramp(edge)
        blend[n - edge :] = ramp(edge)[::-1]
        spliced = src.copy()
        spliced[start:end] = blend * shred + (1 - blend) * src[start:end]
        # the babble bed is untouched and continuous across the span
        out = spliced + parts["bg_only"]
        meta["splice_n"] = count

    elif variant == "burst":
        # the real word is still there, buried: raise the babble inside the
        # span until the local SNR hits BURST_SNR_DB. On a clean background
        # there is no bed to raise, so the burst is mixed in locally -- which
        # is the same event, just one that starts at the span.
        speech = parts["speech"][start:end]
        source = parts["bg_only"] if float(np.mean(parts["bg_only"] ** 2)) > 0 else parts["babble"]
        span_power = max(float(np.mean(source[start:end] ** 2)), 1e-12)
        target_power = max(float(np.mean(speech**2)), 1e-12) / (10 ** (BURST_SNR_DB / 10))
        gain = np.sqrt(target_power / span_power)
        edge = min(int(BURST_RAMP_SEC * sr), n // 2)
        env = np.full(n, gain, dtype=np.float32)
        env[:edge] = 1 + (gain - 1) * ramp(edge)
        env[n - edge :] = 1 + (gain - 1) * ramp(edge)[::-1]
        out[start:end] = speech + env * source[start:end]
        meta["mask_snr_db"] = BURST_SNR_DB

    peak = float(np.max(np.abs(out)))
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32), meta


# ---
# build loop
# ---


def build_rows(split, n_utts, seen_slurp_ids, babble_pool):
    """One pass over a SLURP split: one answer row and one repair row per utterance."""
    rows, scanned, done = [], 0, 0
    skip.clear()
    pbar = tqdm(total=n_utts, desc=f"[{split}]", unit="utt", dynamic_ncols=True)

    def candidates():
        nonlocal scanned
        for row in slurp_ds_stream(split):
            scanned += 1
            pbar.set_postfix({**skip, "scanned": scanned}, refresh=False)
            if row["slurp_id"] in seen_slurp_ids or len(row["sentence"].split()) < 4:
                skip["seen/short"] += 1
                continue
            # claim the id HERE, not after a successful build: slurp streams
            # several recordings of the same prompt back to back (up to ~10),
            # and imap_ordered keeps workers builds in flight, so marking it in
            # the consumer let duplicates of one sentence race through together
            seen_slurp_ids.add(row["slurp_id"])
            yield row

    def build(row):
        # this is the threadpool worker
        slurp_id, sentence = row["slurp_id"], row["sentence"]
        rng = random.Random(f"{SEED}:{slurp_id}")

        plan = plan_spans(sentence, row["annotation"])
        if plan is None:
            return {"skip": "no-slot"}
        crit = plan["crit"]
        if len(crit["phrase"].split()) / len(sentence.split()) > 0.5:
            # masking half the command is a repeat, not a repair
            return {"skip": "crit-too-long"}

        clean = row["audio"]["array"].astype(np.float32)
        clean = clean[: MAX_AUDIO_SECONDS * AUDIO_SAMPLING_RATE]

        words = align_words(clean, sentence)
        if words is None:
            return {"skip": "align"}

        bg, reason = build_bed(
            # never mix an utterance with itself
            clean,
            [arr for sid, arr in babble_pool if sid != slurp_id],
            plan,
            sentence,
            rng,
        )
        if bg is None:
            return {"skip": reason}

        def to_span(idx_range):
            return (words[idx_range[0]]["start"], words[idx_range[1] - 1]["end"])

        pending = []
        for kind, chosen in (("repair", crit), ("answer", bg["ans"])):
            span = to_span(chosen["range"])
            # test pairs both spans against all four masks; train draws one per
            # row, so a sentence never repeats as audio
            for variant in VARIANTS if split == "test" else (rng.choice(VARIANTS),):
                audio, meta = apply_mask(bg, span, variant, rng)
                if audio is None:
                    skip[f"mask-{variant}"] += 1
                    continue
                path = os.path.join(
                    AUDIO_DIR, f"{split}_{slurp_id}_{kind}_{variant}.wav"
                )
                sf.write(path, audio, AUDIO_SAMPLING_RATE)
                pending.append(
                    {
                        "kind": kind,
                        "span": span,
                        "piece": chosen,
                        "variant": variant,
                        "path": path,
                        "meta": meta,
                    }
                )
        if not pending:
            return {"skip": "mask"}

        kinds = {p["kind"] for p in pending}
        if split == "train":
            # the answer target reads the sentence alone, so one call covers
            # every answer row this utterance produced
            with ThreadPoolExecutor(max_workers=len(kinds)) as ex:
                targets = dict(
                    zip(
                        kinds,
                        ex.map(
                            lambda k: write_target(
                                sentence,
                                k,
                                {"lost": [crit["phrase"]] if k == "repair" else []},
                            ),
                            kinds,
                        ),
                    )
                )
            if not all(targets.values()):
                for p in pending:
                    os.remove(p["path"])
                return {"skip": "targets"}
        else:
            targets = {k: "" for k in kinds}

        return {
            "rows": [
                {
                    "kind": p["kind"],
                    "target": targets[p["kind"]],
                    "audio": p["path"],
                    "mask": p["variant"],
                    "snr_db": bg["snr"],
                    "mask_snr_db": p["meta"]["mask_snr_db"],
                    "splice_n": p["meta"]["splice_n"],
                    "slot_type": p["piece"]["slot"],
                    "slot_rank": p["piece"]["rank"],
                    "lost": [p["piece"]["phrase"]],
                    "mask_start": p["meta"]["mask_start"],
                    "mask_end": p["meta"]["mask_end"],
                    "bed_asr": bg["hyp"],
                    "slurp_id": slurp_id,
                    "sentence": sentence,
                }
                for p in pending
            ]
        }

    for _, built in imap_ordered(candidates(), build, UTTERANCE_WORKERS):
        if "skip" in built:
            skip[built["skip"]] += 1
            continue
        for r in built["rows"]:
            rows.append({"id": next(ROW_ID), **r})
        done += 1
        pbar.update(1)
        if done >= n_utts:
            break

    pbar.close()
    log(f"[{split}] built {len(rows)} rows from {done} utterances ({scanned} scanned)")
    log(f"[{split}] skips: {dict(skip)}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds-id", required=True)
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--n-test", type=int, default=N_TEST)
    ap.add_argument("--clean-bg-prob", type=float, default=CLEAN_BG_PROB)
    ap.add_argument(
        "--silence-hard",
        action="store_true",
        help="mask='silence' zeroes the mixed signal outright (a hard dropout, "
        "babble and all) instead of letting the background run on.",
    )
    ap.add_argument(
        "--exclude-ds",
        nargs="*",
        default=list(EXCLUDE_DS_IDS),
        help="datasets whose train slurp_ids the test split must avoid.",
    )
    ap.add_argument(
        "--asr-model",
        default=WHISPER_MODEL_ID,
        help="the gate's ASR. A weaker one rejects good audio for its own "
        "errors, which is a silent bias toward sentences it can spell.",
    )
    ap.add_argument("--mask-pad", type=float, default=MASK_PAD)
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    MASK_PAD = args.mask_pad
    CLEAN_BG_PROB = args.clean_bg_prob
    SILENCE_HARD = args.silence_hard

    AUDIO_DIR = os.path.join(AUDIO_ROOT, args.ds_id.split("/")[-1])
    shutil.rmtree(AUDIO_DIR, ignore_errors=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    log(f"audio dir: {AUDIO_DIR}")

    aligner_processor = AutoProcessor.from_pretrained(ALIGNER_MODEL_ID)
    aligner_model = (
        AutoModelForTokenClassification.from_pretrained(
            ALIGNER_MODEL_ID, dtype=torch.bfloat16, device_map="auto"
        )
        .eval()
    )
    whisper_processor = WhisperProcessor.from_pretrained(args.asr_model)
    whisper_model = (
        WhisperForConditionalGeneration.from_pretrained(
            args.asr_model, dtype=torch.float16, device_map="auto"
        )
        .eval()
    )
    log(f"aligner + {args.asr_model} loaded")

    # the vLLM box gets re-served with different models, so take the name from
    # the server rather than trusting babble_data's constant -- a mismatch is a
    # 404 on every target call, and the rows just quietly skip
    served = babble_data.client.models.list().data[0].id
    if served != babble_data.TARGET_MODEL:
        log(f"target model: {babble_data.TARGET_MODEL} -> {served} (from the server)")
        babble_data.TARGET_MODEL = served

    # test first, so it can claim the sentences it wants: it must avoid every
    # sentence the ear/babble adapters trained on (otherwise scoring them here
    # is a leak), and train then avoids what test took
    # streaming a column still pulls the parquet row groups the audio sits in,
    # so this scan costs minutes per dataset. The ids never change once a
    # dataset is pushed -- cache them beside the audio and pay it once.
    cache_path = os.path.join(AUDIO_ROOT, "exclude_ids.json")
    try:
        with open(cache_path) as f:
            cached = json.load(f)
    except (OSError, ValueError):
        cached = {}

    excluded = set()
    for ds_id in args.exclude_ds:
        if ds_id not in cached:
            try:
                stream = load_dataset(ds_id, split="train", streaming=True)
                cached[ds_id] = sorted(
                    {r["slurp_id"] for r in stream.select_columns(["slurp_id"])}
                )
            except Exception as e:
                log(f"could not read {ds_id} for exclusion: {e}")
                continue
            with open(cache_path, "w") as f:
                json.dump(cached, f)
        excluded.update(cached[ds_id])
        log(f"  {ds_id}: {len(cached[ds_id])} train ids")
    log(f"excluded {len(excluded)} slurp_ids from the test split")

    test_pool = collect_babble_pool("test")
    test_rows = build_rows("test", args.n_test, set(excluded), test_pool)

    train_pool = collect_babble_pool("train")
    train_rows = build_rows(
        "train", args.n_train, {r["slurp_id"] for r in test_rows}, train_pool
    )

    # sits with the wavs it references, so a dataset's audio and its row
    # metadata stay together under one gitignored folder
    dump = os.path.join(AUDIO_DIR, "rows.json")
    with open(dump, "w") as f:
        json.dump({"train": train_rows, "test": test_rows}, f, indent=1)
    log(f"wrote {dump} before pushing")

    for name, rows in (("train", train_rows), ("test", test_rows)):
        log(
            f"{name} {len(rows)} rows "
            f"{Counter(r['kind'] for r in rows)} {Counter(r['mask'] for r in rows)} "
            f"clean-bg {sum(r['snr_db'] is None for r in rows)}"
        )

    def list2ds(rows):
        return Dataset.from_list(rows).cast_column(
            "audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE)
        )

    dsd = DatasetDict({"train": list2ds(train_rows), "test": list2ds(test_rows)})
    if args.no_push:
        log(f"--no-push: built {args.ds_id} locally, see {dump}")
        os._exit(0)

    dsd.push_to_hub(args.ds_id)
    log(f"Pushed {len(train_rows)} train / {len(test_rows)} test rows to {args.ds_id}.")
