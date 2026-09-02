import argparse
import itertools
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, Dataset, DatasetDict, IterableDataset, load_dataset
from openai import OpenAI
from qwen_omni_utils import process_mm_info
from tqdm import tqdm
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from prompts import (
    ANSWER_TARGET_SYSTEM,
    ASR_LOSS_SYSTEM,
    ASR_PROMPT_QWEN2_5,
    ASR_SYSTEM_PROMPT_QWEN2_5,
    BEAM_LOSS_SYSTEM,
    CLASSIFY_SYSTEM,
    HEARD_PREFILL,
    KEY_PIECES_SYSTEM,
    LABEL_TARGET_SYSTEM,
    QWEN25_SYSTEM_PROMPT,
    REPAIR_TARGET_TREE_SYSTEM,
    REPEAT_TARGET_SYSTEM,
    RESP_LOSS_SYSTEM,
    SENT_ASR_LOSS_SYSTEM,
    SENT_RESP_LOSS_SYSTEM,
    TARGET_SYSTEM,
    TASK_PROMPT,
    TASK_PROMPT_TREE,
    split_heard_reply,
    task_prompt,
)
from util import (
    NUM_BAB_SPEAKERS,
    add_noise,
    detect_model_family,
    load_config,
    load_model,
    omni_generate,
    quiet_chat_template,
)

skip = Counter()


def log(*args):
    """print-compatible logging that doesn't break the tqdm bar."""
    tqdm.write(" ".join(str(a) for a in args))


AUDIO_SAMPLING_RATE = 16000

# defend against single long audio causing oom
MAX_AUDIO_SECONDS = 30
MIN_AUDIO_SECONDS = 1

N_TRAIN_TRIPLETS = 1000
N_TEST_TRIPLETS = 50
N_TRAIN_EXTRA_ANS = 1000

# Classification + Target generation are served by the local vLLM judge
# box. Its slurm job records the node it landed on in VLLM_HOST_FILE.
VLLM_HOST_FILE = "/gscratch/sciencehub/zanqil/vllm_judge/vllm_judge_host.txt"
MASK_DS_ID = "keylazy/slurp-ear-sft"
AUDIO_ROOT = "babble_audio"
AUDIO_DIR = None  # set in __main__ basedon ds name
PROBE_DIR = None  # scratch wavs the probes listen to, under AUDIO_DIR
SEED = 42
ROW_ID = itertools.count(1)

BABBLE_POOL_SIZE = 300
BABBLE_CLIP_MAX_SEC = 10  # trim pool clips to save memory

PROBE_BATCH_SIZE = 16
# --beam-label runs ASR_NUM_BEAMS sequences per probe against a 30s audio
# context, so the batch has to shrink to keep the same beams-in-flight budget.
# The dropped task-response pass buys the wall clock back.
MAX_PROBES = 3
# A round draws probes for the slots it still needs, not for the three it
# started with: the first round asks for all of KINDS and hits the cap above,
# but a second round chasing one leftover kind paid the same 16-audio GPU pass
# to fill one slot. 8 draws off a single weighted band is plenty, and the
# multi-slot rounds are unchanged because the cap still binds there.
PROBE_PER_SLOT = 8
ANSWER_PROBE_BATCH_SIZE = 4
UTTERANCE_WORKERS = 6
GPU_LOCK = threading.Lock()

SLOT_SNR = {
    "answer": (8.0, 20.0),
    "repair": (0.0, 12.0),
    "repeat": (0.0, 4.0),
}
# --heard-reply and --tree-label use disjoint bands, so SNR range and kind
# stay close to a function of each other.
SLOT_SNR_DISJOINT = {
    "answer": (12.0, 20.0),
    "repair": (5.0, 12.0),
    "repeat": (0.0, 5.0),
}
# keep training on clean audio, which is the common case in deployment
CLEAN_ANSWER_PROB = 0.25
SLOT_WEIGHTS = {"answer": 1, "repair": 2, "repeat": 2}

CLASSIFY_TEMPERATURE = 0.0
CLASSIFY_MAX_TOKENS = 1024
TARGET_MAX_TOKENS = 1024
TARGET_RETRIES = 8
CLASSIFY_WORKERS = 8  # parallel classifier calls to vLLM

ASR_MAX_NEW_TOKENS = 64
ASR_N_BEST = 4
ASR_NUM_BEAMS = 4
RESP_MAX_NEW_TOKENS = 256  # task response from base omni model
# --heard-reply keeps the full 256: there the reply IS the SFT target. On the
# tracks below, the response is only a witness -- the labeler reads it to
# decide which key pieces went missing and nothing else keeps it -- so the
# tail is decoded for nobody. Batched generate runs until the LONGEST sequence
# stops, so the cap, not the mean, sets the wall clock: base responses measured
# over results/bab_results_*.jsonl are 42 words at the mean and 111 at p99, so
# 128 tokens leaves ~97% of them whole and halves the worst-case decode.
PROBE_RESP_MAX_NEW_TOKENS = 128

KINDS = ("answer", "repair", "repeat")


random.seed(SEED)
np.random.seed(SEED)

# both connected on the first LLM call, see gpt_json
client = None
TARGET_MODEL = None
_vllm_lock = threading.Lock()

# set in __main__ from --omni-path before build_triplets runs
base_model = None
base_processor = None
base_family = None
# set in __main__ from the track flags; switches the probe pass, the labeler,
# the SNR bands, and how the SFT target is composed
TRACK = "two-pass"
IM_END_ID = None


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
def base_generate_batch(convs, max_new_tokens, prefill=None, n_best=1):
    """One batched greedy/beam pass. Takes GPU_LOCK itself -- callers don't.

    The chat template, the audio loads and the mel features are pure CPU work
    that used to sit inside the caller's `with GPU_LOCK`, so a worker preparing
    its batch blocked the worker that actually had the GPU. Only the transfer
    and generate hold the lock now, and the tensors stay on CPU until it is
    held, so a worker queued behind the lock keeps its features off the device.
    """
    with quiet_chat_template():
        texts = base_processor.apply_chat_template(
            convs, add_generation_prompt=True, tokenize=False
        )
    if prefill is not None:
        texts = [t + prefill for t in texts]
    mm_audios, images, videos, *_ = process_mm_info(convs, use_audio_in_video=False)

    # inputs computed on CPU do not need to lock the GPU
    # Only the transfering of inputs from cpu to gpu
    # needs lock
    inputs = base_processor(
        text=texts,
        audio=mm_audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
    )

    gen_kwargs = {"do_sample": False}
    if n_best > 1:
        gen_kwargs = {
            "do_sample": False,
            "num_beams": ASR_NUM_BEAMS,
            "num_return_sequences": n_best,
        }
    with GPU_LOCK:
        inputs = inputs.to(base_model.device, dtype=base_model.dtype)
        # omni_generate, not base_model.generate: with --asr-adapter this is the
        # FULL omni model (the adapter's keys are `thinker.`-prefixed), whose
        # generate silently DROPS a bare max_new_tokens and decodes to its own
        # 1024 default -- on 30s probe audio that is a thousand tokens of
        # "dc dc dc ..." per hypothesis, four per probe. It also owns
        # return_audio, which only exists on that path. The helper spells the
        # kwargs for whichever shape is loaded and asserts the cap held.
        # It returns the generated ids only, and gets them off the device
        # before the lock is released so the decode below -- and the tensors it
        # would otherwise pin -- are outside the critical section.
        gen = omni_generate(
            base_model,
            inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=IM_END_ID,
            pad_token_id=IM_END_ID,
            **gen_kwargs,
        ).cpu()
    decoded = [
        t.lower().strip()
        for t in base_processor.batch_decode(gen, skip_special_tokens=True)
    ]
    if prefill is not None:
        decoded = [f"{prefill}{d}" for d in decoded]
    if n_best > 1:
        # group by input
        return [decoded[i : i + n_best] for i in range(0, len(decoded), n_best)]
    return decoded


# ---
# LLM calls (vLLM judge server)
# ---


def gpt_json(system, user, temperature, max_tokens):
    global client, TARGET_MODEL

    with _vllm_lock:
        if client is None:
            with open(VLLM_HOST_FILE) as f:
                host = f.read().strip()
            c = OpenAI(base_url=f"http://{host}:8000/v1", api_key="EMPTY")
            TARGET_MODEL = c.models.list().data[0].id
            log(f"target model: {TARGET_MODEL} @ http://{host}:8000/v1")
            client = c  # assigned last: it is what the next thread checks

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
CLASSIFY_USER = (
    'COMMAND:\n"{sentence}"\n\n'
    'TRANSCRIPT:\n"{transcript}"\n\n'
    'REPLY:\n"{response}"'
)


_TEXT_NORMALIZER = BasicTextNormalizer()


def _normalize_text(s):
    return _TEXT_NORMALIZER(s).strip()


# ---
# --heard-reply: label + target in one few-shot call
# ---

# With one witness instead of two, labeling is a single text comparison
# (diff HEARD against the command, count what's gone) rather than a
# reconciliation of two passes that can disagree -- which is what makes
# examples viable where CLASSIFY_SYSTEM needed ~90 lines of survival rules.
# Every rule those lines enforced is now carried by an example: the wake-word
# and implied-word exclusions (ex. 2), no-reveal on deletions (ex. 4), the
# misheard-as confirmation form (ex. 3, 5), content-free repeat requests
# (ex. 6, 7). Only the key-piece definition and the per-kind reply style are
# still stated, because no single example teaches them.
LABEL_TARGET_USER = "COMMAND: {sentence}\nHEARD:   {heard}"


def label_target(sentence, heard):
    """(command, Heard line) -> {kind, missing, misheard_as, reply, reason}.

    The kind/lost-count consistency check that used to retry here was removed:
    the model's own kind judgment tracked the prompt's intent better than a
    mechanical count match did.
    """
    if not heard:
        return None

    user = LABEL_TARGET_USER.format(
        sentence=_normalize_text(sentence), heard=_normalize_text(heard)
    )
    obj = gpt_json(
        LABEL_TARGET_SYSTEM,
        user,
        temperature=CLASSIFY_TEMPERATURE,
        max_tokens=CLASSIFY_MAX_TOKENS,
    )
    if obj is None:
        return None

    kind = str(obj.get("kind", "")).strip().lower()
    if kind not in KINDS:
        return None

    missing = [str(s).strip() for s in obj.get("lost", []) if str(s).strip()]
    return {
        "kind": kind,
        "missing": missing,
        "misheard_as": str(obj.get("misheard_as", "")).strip(),
        "reply": str(obj.get("reply", "")).strip(),
        "reason": (
            f"lost from Heard: {'; '.join(missing)}"
            if missing
            else "all key pieces survived in Heard"
        ),
    }


# ---
# Target generation
# ---

# variable suffix, kept last so TARGET_SYSTEM stays cacheable
TARGET_USER = (
    'COMMAND:\n"{sentence}"\n\n'
    'REPAIR-TRANSCRIPT:\n"{repair_transcript}"\n'
    'LOST-PIECE: "{lost_span}"{swap_note}\n\n'
    'REPEAT-TRANSCRIPT:\n"{full_transcript}"'
)


ANSWER_TARGET_USER = 'COMMAND:\n"{sentence}"'


ASR_LOSS_USER = "COMMAND: {sentence}\nHEARD:   {transcript}"


RESP_LOSS_USER = "COMMAND: {sentence}\nREPLY:   {response}"

# SLURP's assistant is "olly" (~150 sentences open with it). Both labeler
# prompts say the wake word is never a key piece, and both still occasionally
# list it -- so a lost piece made only of these is dropped before the table
# sees it. Otherwise a garbled wake word becomes a repair anchor and the
# target asks the user to re-confirm their assistant's name.
WAKE_WORDS = {
    "hey",
    "ok",
    "okay",
    "olly",
    "ollie",
    "alexa",
    "siri",
    "google",
    "assistant",
    "computer",
}


def drop_wake_only(pieces: list[str]):
    return [p for p in pieces if set(_normalize_text(p).split()) - WAKE_WORDS]


LOST_MAX_PCT = 0.6


def lost_too_much(lost, sentence):
    cmd = set(_normalize_text(sentence).split()) - PIECE_STOPWORDS
    anc = set(_normalize_text(lost).split()) - PIECE_STOPWORDS
    return bool(cmd) and len(anc & cmd) / len(cmd) >= LOST_MAX_PCT


# dropped before intersecting two free-text quotes of the same lost piece
PIECE_STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "are",
    "at",
    "be",
    "by",
    "do",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "please",
    "s",
    "that",
    "the",
    "this",
    "to",
    "what",
    "you",
    "your",
}


def label_loss(system, witness_user):
    """Ask LLM for lost"""
    obj = gpt_json(
        system,
        witness_user,
        temperature=CLASSIFY_TEMPERATURE,
        max_tokens=CLASSIFY_MAX_TOKENS,
    )
    if obj is None:
        return None
    return drop_wake_only(
        [str(s).strip() for s in obj.get("lost", []) if str(s).strip()]
    )


def lost_pieces(system: str, pieces: list[str], witness_line: str):
    """Ask LLM for lost within pieces.
    Returns a set for intersection downstream
    """
    pieces_txt = "\n".join(f"{i}. {p}" for i, p in enumerate(pieces, 1))
    obj = gpt_json(
        system,
        f"KEY PIECES:\n{pieces_txt}\n{witness_line}",
        temperature=CLASSIFY_TEMPERATURE,
        max_tokens=CLASSIFY_MAX_TOKENS,
    )
    if obj is None:
        return None
    ids = set()
    for n in obj.get("lost", []):
        try:
            n = int(n)
        except (TypeError, ValueError):
            continue
        if 1 <= n <= len(pieces):
            ids.add(n)
    return ids


def key_pieces(sentence: str):
    """Return None if LLM returns a bad JSON, or every piece filtered.
    Otherwise, returns a list of key pieces -> list[str]

    """
    obj = gpt_json(
        KEY_PIECES_SYSTEM,
        f"COMMAND: {_normalize_text(sentence)}",
        temperature=CLASSIFY_TEMPERATURE,
        max_tokens=CLASSIFY_MAX_TOKENS,
    )
    if obj is None:
        return None
    pieces = drop_wake_only(
        [str(s).strip() for s in obj.get("pieces", []) if str(s).strip()]
    )
    return pieces or None


def decide_kind_ids(pieces, sides: list[set[int]]):
    """Used only for the 2-witness classifer.
    sides: [asr_lost_ids, resp_lost_ids]
    """
    agreed = sorted(set.intersection(*sides)) if sides else []
    kind = "answer" if not agreed else "repair" if len(agreed) == 1 else "repeat"
    lost = [pieces[i - 1] for i in agreed]
    if kind == "repair" and len(pieces) == 1:
        kind = "repeat"

    return {
        "kind": kind,
        "lost": lost,
        "asr_bucket": f"a{len(sides[0])}",
        "resp_bucket": f"r{len(sides[-1])}",
        "reason": (
            f"{len(pieces)} pieces | "
            + " x ".join(str(sorted(s)) for s in sides)
            + f" -> agreed {agreed}"
        ),
        "pieces": pieces,
    }


def decide_kind(sentence, asr_lost, resp_lost):
    """Two independent loss lists -> {kind, lost_piece, missing, buckets}.

    The passes never see each other, so a piece only counts as really lost
    when BOTH of them report it -- the intersection is the label:

        0 pieces  -> "answer"   (the passes disagree about every loss)
        1 piece   -> "repair"   (one agreed piece to ask about)
        >=2       -> "repeat"   (too many holes to build one question on)

    Plus one override: when each pass on its own says most of the command went
    missing, the audio is a "repeat" whatever the two lists happen to share --
    an intersection of one there is a coincidence between two wrecks, not a
    single clean hole.

    Two passes that each lost something but agree on none of it are "answer" by
    the first rule, deliberately: disagreement is read as labeler noise rather
    than as evidence. It is not free -- on a 21-triplet build 3 of 21 answer
    rows came out that way, all three real losses (e.g. asr lost "score" while
    the reply lost "the game"), so they carry a confident target on audio that
    dropped a piece. Kept anyway: the alternative rules cost either probe
    throughput or repair rows.
    """

    def tokens(pieces):
        toks = set()
        for p in pieces:
            toks |= set(_normalize_text(p).split())
        return toks - PIECE_STOPWORDS

    cmd = tokens([sentence])
    # `lost` entries are free-text quotes of the same command, so "the 7 am
    # alarm" and "7 am" have to intersect: one shared content token is enough
    # for two entries to be the same piece.
    r_toks = tokens(resp_lost)
    agreed = [p for p in asr_lost if tokens([p]) & r_toks]
    # ASR wording for the surviving piece: both rubrics quote the real
    # command, and the transcript pass is the more literal of the two
    # witnesses, so its quote tends to sit closer to the spoken words.
    lost_piece = agreed[0] if len(agreed) == 1 else ""
    kind = "answer" if not agreed else "repair" if len(agreed) == 1 else "repeat"

    wrecked = bool(cmd) and all(
        len(tokens(side) & cmd) / len(cmd) >= LOST_MAX_PCT
        for side in (asr_lost, resp_lost)
    )
    if wrecked or (kind == "repair" and lost_too_much(lost_piece, sentence)):
        kind, lost_piece = "repeat", ""

    return {
        "kind": kind,
        "lost_piece": lost_piece,
        # no witness of a similar-sounding substitute on this track, so a
        # repair question here always asks openly
        "misheard_as": "",
        # "missing" rather than "lost", to match what the other labelers return
        "missing": (
            [lost_piece]
            if kind == "repair"
            else list(dict.fromkeys(asr_lost + resp_lost)) if kind == "repeat" else []
        ),
        "asr_bucket": f"a{len(asr_lost)}",
        "resp_bucket": f"r{len(resp_lost)}",
        "reason": (
            f"a{len(asr_lost)}xr{len(resp_lost)} -> {len(agreed)} agreed"
            f"{' (both wrecked)' if wrecked else ''}"
            f" | asr lost: {'; '.join(asr_lost) or 'none'}"
            f" | reply lost: {'; '.join(resp_lost) or 'none'}"
        ),
    }


def label_tree(sentence, transcript, response):
    """Label both passes independently, then intersect their loss lists.

    Either labeler failing -- an empty witness, or a call that never returned
    valid JSON -- leaves nothing to intersect, so the probe is dropped and
    probe_by_kinds redraws.
    """
    if not transcript or not response:
        return None
    cmd = _normalize_text(sentence)
    asr_lost = label_loss(
        ASR_LOSS_SYSTEM,
        ASR_LOSS_USER.format(sentence=cmd, transcript=_normalize_text(transcript)),
    )
    resp_lost = label_loss(
        RESP_LOSS_SYSTEM,
        RESP_LOSS_USER.format(sentence=cmd, response=_normalize_text(response)),
    )
    if asr_lost is None or resp_lost is None:
        return None
    return decide_kind(sentence, asr_lost, resp_lost)


def label_sent(pieces, transcript, resp):
    if not transcript or not resp:
        return None
    asr_ids = lost_pieces(
        SENT_ASR_LOSS_SYSTEM, pieces, f"HEARD: {_normalize_text(transcript)}"
    )
    resp_ids = lost_pieces(
        SENT_RESP_LOSS_SYSTEM, pieces, f"REPLY: {_normalize_text(resp)}"
    )
    if asr_ids is None or resp_ids is None:
        # bad json
        return None

    return decide_kind_ids(pieces, [asr_ids, resp_ids])


# ---
# --beam-label: one labeler over the N-best ASR hypotheses
# ---

# `hypotheses` is the caller-built block, one "HYP <i>: <text>" line per
# hypothesis -- the count is left to the caller so ASR_N_BEST can change
# without touching the prompt.
BEAM_LOSS_USER = "COMMAND: {sentence}\n{hypotheses}"


# A consensus entry made only of these carries no task information, whatever
# the rubric's prose says: the first smoke build returned "i", "please", "can",
# "from" and "s" as lost pieces. Wider than PIECE_STOPWORDS, which is tuned for
# a different job (intersecting two free-text quotes) and is shared with the
# other tracks, so it stays as it is.
NON_PIECE_WORDS = (
    PIECE_STOPWORDS
    | WAKE_WORDS
    | {
        "am",
        "as",
        "been",
        "being",
        "can",
        "could",
        "did",
        "does",
        "had",
        "has",
        "have",
        "he",
        "her",
        "him",
        "may",
        "might",
        "must",
        "shall",
        "she",
        "should",
        "them",
        "they",
        "us",
        "was",
        "we",
        "were",
        "will",
        "would",
    }
)


def label_sent_beam(pieces, hyps):
    hyps = [h for h in hyps if h and h.strip()]
    if not hyps:
        return None

    with ThreadPoolExecutor(max_workers=len(hyps)) as ex:
        sides = list(
            ex.map(
                lambda h: lost_pieces(
                    SENT_ASR_LOSS_SYSTEM, pieces, f"HEARD: {_normalize_text(h)}"
                ),
                hyps,
            )
        )
    if any(s is None for s in sides):
        return None
    return decide_kind_ids(pieces, sides)


def label_beam(sentence, hyps):
    """(command, [hyp1..hypK]) -> label dict | None.

    Kind is recomputed here rather than taken from the model, because the
    filters below can shorten the consensus list after the model has counted
    it -- a lone wake-word or function-word entry would otherwise make an
    "answer" probe a "repair" whose question asks the user to re-confirm the
    assistant's own name, or pair with one real loss to file a repair as a
    repeat. The model's own "kind" is therefore ignored.
    """
    hyps = [h for h in hyps if h and h.strip()]
    if not hyps:
        return None

    obj = gpt_json(
        BEAM_LOSS_SYSTEM,
        BEAM_LOSS_USER.format(
            sentence=_normalize_text(sentence),
            hypotheses="\n".join(
                f"HYP {i}: {_normalize_text(h)}" for i, h in enumerate(hyps, 1)
            ),
        ),
        temperature=CLASSIFY_TEMPERATURE,
        max_tokens=CLASSIFY_MAX_TOKENS,
    )
    if obj is None:
        return None

    lost = [
        p
        for p in (str(s).strip() for s in obj.get("lost", []))
        if p and set(_normalize_text(p).split()) - NON_PIECE_WORDS
    ]
    unintelligible = bool(obj.get("unintelligible", False))

    if unintelligible:
        kind = "repeat"
    else:
        kind = {0: "answer", 1: "repair"}.get(len(lost), "repeat")
    lost_piece = lost[0] if kind == "repair" else ""
    misheard = str(obj.get("misheard_as", "")).strip() if kind == "repair" else ""
    if kind == "repair" and lost_too_much(lost_piece, sentence):
        # one bundled entry spanning most of the command: a repeat request
        # wearing a repair's clothes
        kind, lost_piece, misheard = "repeat", "", ""
    if kind == "repair" and not lost_piece:
        # unusable: a repair with no piece to build the question around
        return None

    per_hyp = obj.get("per_hypothesis", [])
    return {
        # "missing" rather than "lost", to match the other labelers
        "missing": (
            [lost_piece] if kind == "repair" else (lost if kind == "repeat" else [])
        ),
        "kind": kind,
        "lost_piece": lost_piece,
        "misheard_as": misheard,
        "unintelligible": unintelligible,
        # kept on the row so a mislabeled probe can be diagnosed without
        # re-running the GPU pass
        "per_hypothesis": per_hyp,
        "reason": (
            f"{len(hyps)} hyps | consensus lost: {'; '.join(lost) or 'none'}"
            f"{' (unintelligible)' if unintelligible else ''}"
            + (f" | misheard as: {misheard}" if misheard else "")
        ),
    }


# ---
# one target per probe -- every track but --heard-reply
# ---


# The only repair pair left. Every surviving labeler intersects per-witness
# loss lists (two on --tree-label and --sent-2, K hypotheses on --beam-label
# and --sent-4), so by construction every part of the command outside
# LOST-PIECE came through on all of them -- saying that in the prompt is both
# more accurate and shorter than handing over a noisy transcript and asking the
# writer to work out which words it can trust. No MISHEARD-AS either: none of
# them reports a substitute word any more, so the question always asks openly.
REPAIR_TARGET_TREE_USER = 'COMMAND:\n"{sentence}"\nLOST-PIECE: "{lost_piece}"'


REPEAT_TARGET_USER = 'COMMAND:\n"{sentence}"'


def write_target(sentence, kind, probe):
    """One SFT target for one already-labeled probe (every track but heard-reply).

    Deliberately a second call, after the labeler: the label wants
    temperature 0 so a rerun reproduces the dataset while the reply wants 0.7
    so a few thousand targets don't all open the same way, and a target retry
    here must not re-roll the row's kind.
    """
    # on repair the label's `lost` is the single agreed piece, and it is the
    # only thing the question may not speak
    lost_piece = probe["lost"][0] if kind == "repair" else ""

    if kind == "answer":
        system = ANSWER_TARGET_SYSTEM
        user = ANSWER_TARGET_USER.format(sentence=sentence)
    elif kind == "repeat":
        system = REPEAT_TARGET_SYSTEM
        user = REPEAT_TARGET_USER.format(sentence=sentence)
    else:
        # every labeler that reaches here now reports losses as key-piece ids
        # against one inventory, with no witness of a similar-sounding
        # substitute -- so the question always asks openly
        system = REPAIR_TARGET_TREE_SYSTEM
        user = REPAIR_TARGET_TREE_USER.format(sentence=sentence, lost_piece=lost_piece)

    leakable = (
        set(_normalize_text(lost_piece).split()) - NON_PIECE_WORDS
        if kind == "repair"
        else set()
    )

    for attempt in range(TARGET_RETRIES):
        obj = gpt_json(system, user, temperature=0.7, max_tokens=TARGET_MAX_TOKENS)
        if obj is None:
            time.sleep(2**attempt)
            continue
        target = str(obj.get(kind, "")).strip()
        if not target:
            continue
        if leakable & set(_normalize_text(target).split()):
            skip["target-leak"] += 1
            continue
        return target
    return ""


# ---
# SNR probing -> classify its kind
# ---


def probe_by_kinds(clean, pool, sentence, kinds_need, batch_size, rng):
    def make_probe_batch(kinds_need):
        """return a list of wav paths and a list of snr vals"""
        bands = SLOT_SNR if TRACK == "two-pass" else SLOT_SNR_DISJOINT
        paths, snrs = [], []
        n_draw = min(batch_size, PROBE_PER_SLOT * len(kinds_need))
        while len(paths) < n_draw:
            weights = [SLOT_WEIGHTS[k] for k in kinds_need]
            slot = rng.choices(kinds_need, weights=weights, k=1)[0]

            if (
                TRACK != "two-pass"
                and slot == "answer"
                and rng.random() < CLEAN_ANSWER_PROB
            ):
                # keep noise-free audio in the answer band; snr_db=None reads
                # as "clean" everywhere downstream
                noisy, snr = clean, None
            else:
                noisy, snr = add_noise(clean, pool, bands[slot], rng)

            # the probe reads this file rather than the float32 array, so the
            # model that gets labeled hears the exact PCM_16 samples the row
            # will ship -- feeding it the pre-quantization array made the base
            # model answer a slightly different audio at eval time
            fd, path = tempfile.mkstemp(suffix=".wav", dir=PROBE_DIR)
            os.close(fd)
            sf.write(path, noisy, AUDIO_SAMPLING_RATE)
            paths.append(path)
            snrs.append(snr)
        return paths, snrs

    def classify(transcript, response, retries=2):
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

    results: dict[str, dict | None] = {k: None for k in kinds_need}

    pieces = None
    if TRACK in ("sent-2", "sent-4"):
        pieces = key_pieces(sentence)
        if pieces is None:
            skip["pieces"] += 1
            return results

    for _ in range(MAX_PROBES):
        missing_slots = [k for k, v in results.items() if v is None]
        if not missing_slots:
            break
        paths, snrs = make_probe_batch(missing_slots)

        sysp = QWEN25_SYSTEM_PROMPT if base_family == "qwen2.5" else None
        # only --beam-label fills this in; the others keep one transcript
        hyp_lists = [[] for _ in paths]
        if TRACK in ("beam", "sent-4"):
            asr_sysp = ASR_SYSTEM_PROMPT_QWEN2_5 if base_family == "qwen2.5" else None
            convs = [_conv(p, asr_sysp, ASR_PROMPT_QWEN2_5) for p in paths]
            hyp_lists = base_generate_batch(
                convs, ASR_MAX_NEW_TOKENS, n_best=ASR_N_BEST
            )
            # the top beam is what the row and the logs call the transcript
            transcripts = [h[0] for h in hyp_lists]
            # no task-response pass on this track: nothing reads it, so
            # decoding it would be dead GPU time
            responses = ["" for _ in hyp_lists]
            with ThreadPoolExecutor(max_workers=CLASSIFY_WORKERS) as ex:
                labels = list(ex.map(lambda h: label_sent_beam(pieces, h), hyp_lists))
        elif TRACK == "heard-reply":
            convs = [_conv(p, sysp, task_prompt(True)) for p in paths]
            outs = base_generate_batch(
                convs, RESP_MAX_NEW_TOKENS, prefill=HEARD_PREFILL
            )
            pairs = [split_heard_reply(o) for o in outs]
            transcripts = [h for h, _ in pairs]
            responses = [r for _, r in pairs]
            n_format_fail = sum(1 for h in transcripts if not h)
            if n_format_fail and skip["format"] < 5:
                # print one raw sample per round while format failures are
                # still rare in this run, to see what the model actually said
                for o in outs:
                    if not split_heard_reply(o)[0]:
                        log(f"[format-fail raw output]: {o!r}")
                        break
            skip["format"] += n_format_fail
            with ThreadPoolExecutor(max_workers=CLASSIFY_WORKERS) as ex:
                labels = list(ex.map(lambda h: label_target(sentence, h), transcripts))
        else:  # 2-witness (sent-2) track goes here
            # get batch omni asr respond
            asr_sysp = ASR_SYSTEM_PROMPT_QWEN2_5 if base_family == "qwen2.5" else None
            convs = [_conv(p, asr_sysp, ASR_PROMPT_QWEN2_5) for p in paths]
            transcripts = base_generate_batch(convs, ASR_MAX_NEW_TOKENS)

            # get batch omni assistant respond.
            # The lock is released between 2 base_generate_batch calls.
            # It allows others more chance to use the GPU
            task = TASK_PROMPT_TREE if TRACK in ("tree", "sent-2") else TASK_PROMPT
            convs = [_conv(p, sysp, task) for p in paths]
            responses = base_generate_batch(convs, PROBE_RESP_MAX_NEW_TOKENS)

            if TRACK == "two-pass":
                label_one = classify
            elif TRACK == "sent-2":
                label_one = lambda t, r: label_sent(pieces, t, r)
            else:
                label_one = lambda t, r: label_tree(sentence, t, r)
            with ThreadPoolExecutor(max_workers=CLASSIFY_WORKERS) as ex:
                labels = list(
                    ex.map(
                        lambda it: label_one(*it),
                        list(zip(transcripts, responses)),
                    )
                )

        filled = False
        for snr, probe_path, transcript, response, hyps, label in zip(
            snrs, paths, transcripts, responses, hyp_lists, labels
        ):
            if label is None:
                continue

            kind = label["kind"]
            # tree/beam/two-pass call it "missing" and quote it out of the
            # command; the sent tracks call it "lost" and it is already exact
            # key-piece text
            lost = label.get("lost", label.get("missing", []))
            if kind in results and results[kind] is None:
                filled = True
                results[kind] = {
                    "snr_db": snr,
                    "audio": probe_path,
                    "transcript": transcript,
                    "response": response,
                    "lost": lost,
                    "reason": label["reason"],
                    # --heard-reply writes the target in the same call; the
                    # other two tracks fill this in later, once the probe is
                    # actually kept
                    "target": label.get("reply", ""),
                    # tree track only: the two per-pass loss counts
                    "asr_bucket": label.get("asr_bucket", ""),
                    "resp_bucket": label.get("resp_bucket", ""),
                    "pieces": label.get("pieces", []),
                    # beam track only: all K hypotheses (the repair target
                    # writer grounds its question in these), the per-hypothesis
                    # loss lists behind the consensus, and whether the
                    # hypotheses read as some other sentence entirely
                    "hypotheses": list(hyps),
                    "beam_losses": label.get("per_hypothesis", []),
                }

        # clean up non-kept wav files
        kept = {r["audio"] for r in results.values() if r}
        for p in paths:
            if p not in kept:
                os.remove(p)

        if not filled:
            # a whole round of labeled probes claimed no slot: the SNR bands
            # are already redrawn per probe, so a second identical draw is not
            # a new experiment. Most starved utterances used to burn all
            # MAX_PROBES rounds to reach the same skip.
            break

    return results


# ---
# triplet-building loop
# ---


def imap_ordered(items, work, workers):
    """Yield (item, work(item)) in input order, `workers` builds at a time.

    The queue is kept deeper than the pool is wide. With a window of exactly
    `workers`, nothing is submitted until the head is yielded, so a slow head
    left the other threads sitting on finished futures -- and utterance times
    vary a lot here (an early skip after one probe round vs. three rounds plus
    three target calls). The pool still runs `workers` at once; the extra
    queued items only keep a thread from idling.
    """

    it = iter(items)
    pending = deque()
    window = workers * 3
    with ThreadPoolExecutor(max_workers=workers) as ex:
        try:
            while True:
                while len(pending) < window:
                    item = next(it, None)
                    if item is None:
                        break
                    pending.append((item, ex.submit(work, item)))
                if not pending:
                    return
                item, fut = pending.popleft()
                yield item, fut.result()
        finally:
            # if a caller doing `for row, result in imap_ordered()`
            # but decided to break the loop after its target reached.
            # yield above will throw `GeneratorExit` error and reach here
            # with non empty pending
            for _, fut in pending:
                fut.cancel()


def slurp_ds_stream(split) -> IterableDataset:
    stream = load_dataset("qmeeus/slurp", split=split, streaming=True)
    assert isinstance(stream, IterableDataset)
    return stream.cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE))


def collect_babble_pool(split):
    stream = slurp_ds_stream(split)
    max_len = BABBLE_CLIP_MAX_SEC * AUDIO_SAMPLING_RATE
    min_len = MIN_AUDIO_SECONDS * AUDIO_SAMPLING_RATE
    pool = []
    for row in stream:
        arr = row["audio"]["array"].astype(np.float32)[:max_len]
        if len(arr) > min_len:
            pool.append((row["slurp_id"], arr))
        if len(pool) >= BABBLE_POOL_SIZE:
            break
    log(f"[{split}] babble pool: {len(pool)} clips")
    return pool


def make_row(kind, target, path, probe, slurp_id, sentence):
    if TRACK == "heard-reply" and target:
        target = f"Heard: {probe['transcript']}\nReply: {target}"
    return {
        "id": next(ROW_ID),
        "kind": kind,
        "target": target,
        "audio": path,
        "snr_db": probe["snr_db"],
        "asr_transcript": probe["transcript"],
        "omni_response": probe["response"],
        "lost": probe["lost"],
        "classifier_reason": probe["reason"],
        "asr_bucket": probe["asr_bucket"],
        "resp_bucket": probe["resp_bucket"],
        # beam track: the K hypotheses this row's label was intersected from,
        # and the per-hypothesis loss lists as JSON so a mislabeled row can be
        # diagnosed without re-running the GPU pass. Empty on the other tracks.
        "beam_hypotheses": probe["hypotheses"],
        "beam_losses": (
            json.dumps(probe["beam_losses"], ensure_ascii=False)
            if probe["beam_losses"]
            else ""
        ),
        "key_pieces": probe["pieces"],
        "slurp_id": slurp_id,
        "sentence": sentence,
        "source": "babble",
    }


def build_triplets(split, n_triplets, seen_slurp_ids, babble_pool):
    rows, scanned, done = [], 0, 0
    skip.clear()
    pbar = tqdm(total=n_triplets, desc=f"[{split}]", unit="triplet", dynamic_ncols=True)

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
            # and imap_ordered keeps UTTERANCE_WORKERS builds in flight, so
            # marking it in the consumer let duplicates of one sentence race
            # through the check together.
            seen_slurp_ids.add(row["slurp_id"])
            yield row

    def build(row):
        # this is the threadpool worker
        slurp_id = row["slurp_id"]
        sentence = row["sentence"]
        clean = row["audio"]["array"].astype(np.float32)
        clean = clean[: MAX_AUDIO_SECONDS * AUDIO_SAMPLING_RATE]

        triplet = probe_by_kinds(
            clean,
            # never mix an utterance with itself
            [arr for sid, arr in babble_pool if sid != slurp_id],
            sentence,
            KINDS,
            PROBE_BATCH_SIZE,
            random.Random(f"{SEED}:{slurp_id}"),
        )
        starved = [k for k, v in triplet.items() if v is None]
        if starved:
            # name the slot that never filled: the kind is the labeler's call,
            # not the SNR band's, so which band starves is not predictable
            return {"skip": f"probe:{','.join(starved)}"}

        if split == "test":
            # test split skips target writing
            return {"triplet": triplet, "targets": {k: "" for k in KINDS}}

        if TRACK == "heard-reply":
            # already written, one call per probe audio alongside its label
            return {
                "triplet": triplet,
                "targets": {k: triplet[k]["target"] for k in KINDS},
            }

        if TRACK in ("tree", "beam", "sent-2", "sent-4"):
            with ThreadPoolExecutor(max_workers=len(KINDS)) as ex:
                targets = dict(
                    zip(
                        KINDS,
                        ex.map(lambda k: write_target(sentence, k, triplet[k]), KINDS),
                    )
                )

            if not all(targets.values()):
                return {"skip": "targets"}
            return {"triplet": triplet, "targets": targets}

        # ---
        # one LLM call writes all 3 targets for the utterance
        # ---
        # STALE: probe dicts no longer carry "swapped" -- see the note above the
        # track flags in __main__
        repair_probe, repeat_probe = triplet["repair"], triplet["repeat"]
        swap_note = ""
        if repair_probe["swapped"]:
            swap_note = f'\nMISHEARD-AS: "{repair_probe["swapped"][0]}"'
        target_user = TARGET_USER.format(
            sentence=sentence,
            repair_transcript=repair_probe["transcript"],
            lost_span=repair_probe["lost"][0],
            swap_note=swap_note,
            full_transcript=repeat_probe["transcript"],
        )
        targets = None
        answer, repair, repeat = "", "", ""
        for attempt in range(TARGET_RETRIES):
            obj = gpt_json(
                TARGET_SYSTEM,
                target_user,
                temperature=0.7,
                max_tokens=TARGET_MAX_TOKENS,
            )
            if obj is None:
                time.sleep(2**attempt)
                continue
            # a retry that drops a type of target still reuses an earlier
            # good one
            answer = str(obj.get("answer", "")).strip() or answer
            repair = str(obj.get("repair", "")).strip() or repair
            repeat = str(obj.get("repeat", "")).strip() or repeat
            if answer and repair and repeat:
                targets = {"answer": answer, "repair": repair, "repeat": repeat}
                break

        if targets is None:
            return {"skip": "targets"}

        return {"triplet": triplet, "targets": targets}

    for row, built in imap_ordered(candidates(), build, UTTERANCE_WORKERS):
        if "skip" in built:
            skip[built["skip"]] += 1
            continue

        slurp_id = row["slurp_id"]
        sentence = row["sentence"]
        for kind in KINDS:
            probe = built["triplet"][kind]
            path = os.path.join(AUDIO_DIR, f"{split}_{slurp_id}_{kind}.wav")
            # move from temp folder to audio dir
            # ready for hf upload
            os.replace(probe["audio"], path)
            rows.append(
                make_row(kind, built["targets"][kind], path, probe, slurp_id, sentence)
            )

        done += 1
        pbar.update(1)
        if done >= n_triplets:
            break

    pbar.close()
    log(f"[{split}] built {len(rows)} rows from {done} utterances ({scanned} scanned)")
    return rows


def build_answer_rows(split, n_rows, seen_slurp_ids, babble_pool):
    rows, scanned = [], 0
    skip.clear()
    pbar = tqdm(total=n_rows, desc=f"[{split}ans]", unit="row", dynamic_ncols=True)

    def candidates():
        nonlocal scanned
        for row in slurp_ds_stream(split):
            scanned += 1
            pbar.set_postfix({**skip, "scanned": scanned}, refresh=False)
            if row["slurp_id"] in seen_slurp_ids or len(row["sentence"].split()) < 4:
                skip["seen/short"] += 1
                continue
            seen_slurp_ids.add(row["slurp_id"])  # see build_triplets.candidates
            yield row

    def build(row):
        slurp_id = row["slurp_id"]
        sentence = row["sentence"]
        clean = row["audio"]["array"].astype(np.float32)
        clean = clean[: MAX_AUDIO_SECONDS * AUDIO_SAMPLING_RATE]

        probe = probe_by_kinds(
            clean,
            [arr for sid, arr in babble_pool if sid != slurp_id],
            sentence,
            ["answer"],
            ANSWER_PROBE_BATCH_SIZE,
            random.Random(f"{SEED}:{slurp_id}"),
        )["answer"]
        if probe is None:
            return {"skip": "probe"}

        if TRACK == "heard-reply":
            return {"probe": probe, "target": probe["target"]}

        if TRACK in ("tree", "beam", "sent-2", "sent-4"):
            target = write_target(sentence, "answer", probe)
            return {"probe": probe, "target": target} if target else {"skip": "targets"}

        target = ""
        for attempt in range(TARGET_RETRIES):
            obj = gpt_json(
                ANSWER_TARGET_SYSTEM,
                ANSWER_TARGET_USER.format(sentence=sentence),
                temperature=0.7,
                max_tokens=TARGET_MAX_TOKENS,
            )
            if obj is None:
                time.sleep(2**attempt)
                continue
            target = str(obj.get("answer", "")).strip()
            if target:
                break
        if not target:
            return {"skip": "targets"}

        return {"probe": probe, "target": target}

    for row, built in imap_ordered(candidates(), build, UTTERANCE_WORKERS):
        if "skip" in built:
            skip[built["skip"]] += 1
            continue

        slurp_id = row["slurp_id"]
        sentence = row["sentence"]

        path = os.path.join(AUDIO_DIR, f"{split}_{slurp_id}_answer.wav")
        os.replace(built["probe"]["audio"], path)
        rows.append(
            make_row(
                "answer", built["target"], path, built["probe"], slurp_id, sentence
            )
        )

        pbar.update(1)
        if len(rows) >= n_rows:
            break

    pbar.close()
    log(f"[{split}ans] built {len(rows)} answer rows ({scanned} scanned)")
    return rows


@dataclass
class Config:
    omni_path: str = "Qwen/Qwen2.5-Omni-3B"
    ds_id: str = "keylazy/slurp-babble-Qwen2.5-Omni-3B"
    n_train: int = N_TRAIN_TRIPLETS
    n_test: int = N_TEST_TRIPLETS
    n_extra_ans: int = N_TRAIN_EXTRA_ANS
    no_push: bool = False
    # the track, as one value instead of five mutually exclusive store_trues.
    # The flags below still write it, so babble_data.sh is untouched.
    label: str = "two-pass"
    # the ft-asr track: the ASR LoRA attached to the probe model
    asr_adapter: str = None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str)
    ap.add_argument("--omni-path", type=str)
    ap.add_argument("--ds-id", type=str)
    ap.add_argument("--n-train", type=int)
    ap.add_argument("--n-test", type=int)
    ap.add_argument("--n-extra-ans", type=int)
    ap.add_argument(
        "--no-push",
        action="store_true",
        default=None,
    )
    ap.add_argument(
        "--asr-adapter",
        default=None,
        help="LoRA attached to the probe model (the ft-asr track). Legal only "
        "on --beam-label / --sent-4, whose probe pass is ASR and nothing "
        "else: every other track writes the task response with this same "
        "model, and an ASR-only adapter answers a spoken command by "
        "transcribing it back.",
    )

    # STALE TRACKS. The sent tracks dropped the probe dict's "swapped" and
    # "lost_piece" entries -- their labels report key-piece ids against one
    # inventory, so the single agreed piece IS probe["lost"][0] and no witness
    # of a similar-sounding substitute exists. Everything downstream now reads
    # only "lost", which leaves two tracks needing work before they run again:
    #   two-pass (the no-flag default): its utterance-level target call still
    #     reads repair_probe["swapped"] and will KeyError there.
    #   --beam-label: still decodes K hypotheses, but is labeled by
    #     label_sent_beam now, so label_beam / BEAM_LOSS_SYSTEM are unused and
    #     the beam_losses column comes out empty.
    # --heard-reply and --tree-label are unaffected.
    #
    # All five write the one `label` field, so the YAML can say `label: sent-4`
    # and the drivers keep passing --sent-4. store_const, not store_true: the
    # override loop needs None when the flag is absent.
    track = ap.add_mutually_exclusive_group()
    track.add_argument(
        "--heard-reply",
        dest="label",
        action="store_const",
        const="heard-reply",
        default=None,
        help="One probe pass emitting 'Heard: ... / Reply: ...'; label off the "
        "Heard line alone with the few-shot labeler, disjoint SNR bands, and "
        "two-line SFT targets.",
    )

    track.add_argument(
        "--tree-label",
        dest="label",
        action="store_const",
        const="tree",
        default=None,
        help="Two probe passes as usual, but each labeled independently "
        "against the command for what it lost; decide_kind() intersects the "
        "two loss lists to pick the kind and the piece to ask about.",
    )

    track.add_argument(
        "--beam-label",
        dest="label",
        action="store_const",
        const="beam",
        default=None,
        help="One beam-search ASR pass per probe; label off the K hypotheses' "
        "consensus alone, a piece counting as lost only if every hypothesis "
        "missed it. No task-response pass.",
    )

    track.add_argument(
        "--sent-2", dest="label", action="store_const", const="sent-2", default=None
    )

    track.add_argument(
        "--sent-4", dest="label", action="store_const", const="sent-4", default=None
    )

    args = ap.parse_args()

    cfg = load_config(args.config, Config) if args.config else Config()
    for key, value in vars(args).items():
        if value is not None and key != "config":
            setattr(cfg, key, value)

    TRACK = cfg.label
    log(f"config: {cfg}")
    log(f"track: {TRACK}")

    if cfg.asr_adapter and TRACK not in ("beam", "sent-4"):
        raise SystemExit(
            f"--asr-adapter is not usable on track {TRACK}: its probe also "
            "generates the task response from this model, which an ASR-only "
            "adapter cannot do."
        )
    if TRACK in ("beam", "sent-4"):
        log(f"probe batch size: {PROBE_BATCH_SIZE} (beams: {ASR_NUM_BEAMS})")

    # ---
    # point AUDIO_DIR at a fresh per-dataset subdir of AUDIO_ROOT.
    # ---
    AUDIO_DIR = os.path.join(AUDIO_ROOT, cfg.ds_id.split("/")[-1])
    shutil.rmtree(AUDIO_DIR, ignore_errors=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    # kept probes are moved out of here into AUDIO_DIR; what stays behind
    # belongs to utterances that were later skipped, so it is dropped below
    PROBE_DIR = os.path.join(AUDIO_DIR, "probes")
    os.makedirs(PROBE_DIR, exist_ok=True)
    log(f"audio dir: {AUDIO_DIR}")

    # init base omni model
    base_family = detect_model_family(cfg.omni_path)
    base_model, base_processor = load_model(
        cfg.omni_path,
        base_family,
        adapter_path=cfg.asr_adapter,
        thinker_only=not cfg.asr_adapter,
    )
    IM_END_ID = base_processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    print("base models loaded")

    # avoid slurp ids in word-masking dataset, so the same sentence
    # isn't double-weighted across the 2 tracks
    seen_ids = set()
    for split in ("train", "test"):
        mask_ds = load_dataset(MASK_DS_ID, split=split, streaming=True)
        for r in mask_ds.select_columns(["slurp_id"]):
            seen_ids.add(r["slurp_id"])

    test_babble_pool = collect_babble_pool("test")
    test_rows = build_triplets("test", cfg.n_test, seen_ids, test_babble_pool)
    train_babble_pool = collect_babble_pool("train")
    train_rows = build_triplets("train", cfg.n_train, seen_ids, train_babble_pool)

    if cfg.n_extra_ans:
        train_rows += build_answer_rows(
            "train",
            cfg.n_extra_ans,
            seen_ids,
            train_babble_pool,
        )

    # whatever is left belongs to utterances that never became rows
    shutil.rmtree(PROBE_DIR, ignore_errors=True)

    # sits with the wavs it references, so a dataset's audio and its row
    # metadata stay together under one gitignored folder
    dump = os.path.join(AUDIO_DIR, "rows.json")
    with open(dump, "w") as f:
        json.dump(
            {
                # the ASR adapter and the git sha are the only things that
                # distinguish this build from an earlier one on the same
                # track, and neither fits in a dataset id
                "config": asdict(cfg),
                "git": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "train": train_rows,
                "test": test_rows,
            },
            f,
            indent=1,
        )
    log(f"wrote {dump} before pushing")

    log(f"train {len(train_rows)} rows {Counter(r['kind'] for r in train_rows)}")
    log(f"test  {len(test_rows)} rows {Counter(r['kind'] for r in test_rows)}")

    def list2ds(rows):
        return Dataset.from_list(rows).cast_column(
            "audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE)
        )

    # built either way, so --no-push still exercises the Audio cast and the
    # schema inference that a push would hit
    dsd = DatasetDict({"train": list2ds(train_rows), "test": list2ds(test_rows)})
    if cfg.no_push:
        log(f"--no-push: built {cfg.ds_id} locally, see {dump}")
        raise SystemExit(0)

    dsd.push_to_hub(cfg.ds_id)
    log(
        f"Pushed {len(train_rows)} train / {len(test_rows)} test rows "
        f"to {cfg.ds_id}."
    )
