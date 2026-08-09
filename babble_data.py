"""
Build babble-noise slurp dataset for conversational-repair training.

Four tracks, selected by --heard-reply / --tree-label / --beam-label:

Default (two-pass): for each noisy probe the omni base model produces an ASR
transcript and, separately, a task response. The LLM sees (original sentence,
transcript, response) and labels the probe:
- "answer":     every detail needed to perform the task survived the noise;
                filler only loss
- "repair":     exactly ONE key piece was lost or misheard;
                the rest can be trusted
- "repeat":     more than one key piece lost in both passes

--tree-label: the same two probe passes, but neither labeler sees the other's
pass, and neither judges the kind. Each returns only a list of the command's
key pieces its own witness lost -- one diffing the transcript, one reading what
the reply demonstrates it heard (under TASK_PROMPT_TREE, which asks the model
to restate what it caught, so the reply is readable as evidence). decide_kind()
intersects the two lists in code: no agreed piece means "answer", exactly one
means "repair" and is the piece the question asks about, two or more means
"repeat" -- as does both lists on their own spanning most of the command.

--beam-label: ONE probe pass, an ASR decode with beam search returning the
ASR_N_BEST best hypotheses for the same noisy audio. There is no task-response
pass -- the hypotheses' disagreement replaces the reply as the second witness.
label_beam diffs the command against each hypothesis separately and intersects
the per-hypothesis loss lists: a key piece counts as lost only if EVERY
hypothesis missed it, so one beam hearing it correctly proves it was audible.
What the hypotheses put in a lost piece's place decides whether the repair
question can offer an alternative back ("misheard_as") or has to ask openly.

--heard-reply (one pass): the base model answers TASK_PROMPT_HR, which asks
for `Heard: <what got through>` then `Reply: <the response>`, so a single
decode yields both. The LLM sees (original sentence, Heard) only -- it lists
the key pieces that did not survive, judges the kind (answer/repair/repeat)
itself, and writes that kind's target reply in the same call. The SFT target
is the two-line string `Heard: <the probe's own Heard>\\nReply: <the written
target>`, so the model is trained to write down what it heard before
conditioning its response on it.
"""

import argparse
import itertools
import json
import logging
import os
import random
import re
import shutil
import tempfile
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, Dataset, DatasetDict, load_dataset
from openai import OpenAI
from qwen_omni_utils import process_mm_info
from tqdm import tqdm
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from prompts import (
    ASR_LOSS_SYSTEM,
    HEARD_PREFILL,
    RESP_LOSS_SYSTEM,
    TASK_PROMPT,
    TASK_PROMPT_TREE,
    split_heard_reply,
    task_prompt,
)
from util import QWEN25_SYSTEM_PROMPT, detect_model_family, load_model

skip = Counter()


def log(*args):
    """print-compatible logging that doesn't break the tqdm bar."""
    tqdm.write(" ".join(str(a) for a in args))


AUDIO_SAMPLING_RATE = 16000

# defend against single long audio causing oom
MAX_AUDIO_SECONDS = 30

N_TRAIN_TRIPLETS = 1000
N_TEST_TRIPLETS = 50
N_TRAIN_EXTRA_ANS = 1000

# Classification + Target generation are served by the local vLLM judge
# box. Its slurm job records the node it landed on in VLLM_HOST_FILE.
TARGET_MODEL = "Qwen/Qwen3.5-122B-A10B-FP8"  # "Qwen/Qwen3.6-35B-A3B-FP8"
VLLM_HOST_FILE = "/gscratch/sciencehub/zanqil/vllm_judge/vllm_judge_host.txt"
MASK_DS_ID = "keylazy/slurp-ear-sft"
AUDIO_ROOT = "babble_audio"
AUDIO_DIR = None  # set in __main__ basedon ds name
PROBE_DIR = None  # scratch wavs the probes listen to, under AUDIO_DIR
SEED = 42
ROW_ID = itertools.count(1)

BABBLE_POOL_SIZE = 300
BABBLE_SPEAKERS = 3
BABBLE_CLIP_MAX_SEC = 10  # trim pool clips to save memory

PROBE_BATCH_SIZE = 16
# --beam-label runs ASR_NUM_BEAMS sequences per probe against a 30s audio
# context, so the batch has to shrink to keep the same beams-in-flight budget.
# The dropped task-response pass buys the wall clock back.
BEAM_PROBE_BATCH_SIZE = 8
MAX_PROBES = 3
ANSWER_PROBE_BATCH_SIZE = 4
UTTERANCE_WORKERS = 4
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
ASR_NUM_BEAMS = 8
RESP_MAX_NEW_TOKENS = 256  # task response from base omni model

KINDS = ("answer", "repair", "repeat")


random.seed(SEED)
np.random.seed(SEED)

with open(VLLM_HOST_FILE) as _f:
    _vllm_host = _f.read().strip()
client = OpenAI(base_url=f"http://{_vllm_host}:8000/v1", api_key="EMPTY")
print(f"target model: {TARGET_MODEL} @ http://{_vllm_host}:8000/v1")

# set in __main__ from --omni-path before build_triplets runs
base_model = None
base_processor = None
base_family = None
# set in __main__ from the track flags; switches the probe pass, the labeler,
# the SNR bands, and how the SFT target is composed
TRACK = "two-pass"
IM_END_ID = None


# ---
# base model: ASR + task response
# ---

ASR_SYSTEM_PROMPT = "You are a speech recognition model."
ASR_PROMPT = "Transcribe the English audio into text without any punctuation marks."  # from Qwen2.5-Omni github cookbooks


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
    logging.disable(logging.WARNING)
    try:
        texts = base_processor.apply_chat_template(
            convs, add_generation_prompt=True, tokenize=False
        )
    finally:
        logging.disable(logging.NOTSET)
    if prefill is not None:
        texts = [t + prefill for t in texts]
    mm_audios, images, videos = process_mm_info(convs, use_audio_in_video=False)
    inputs = base_processor(
        text=texts,
        audio=mm_audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
    ).to(base_model.device, dtype=base_model.dtype)
    gen_kwargs = dict(do_sample=False)
    if n_best > 1:
        gen_kwargs = dict(
            do_sample=False,
            num_beams=ASR_NUM_BEAMS,
            num_return_sequences=n_best,
        )
    out = base_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        eos_token_id=IM_END_ID,
        pad_token_id=IM_END_ID,
        **gen_kwargs,
    )
    gen = out[:, inputs["input_ids"].shape[1] :]
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
LABEL_TARGET_SYSTEM = """You are labeling noisy-audio data for a smart voice \
assistant.

You get the user's real spoken COMMAND, and HEARD -- what the device caught \
after loud background chatter. Compare them and list the key pieces of the \
command that did not survive: entities, names, places, times, dates, \
quantities, titles, and the requested action. Filler words ("please", "could \
you", "hey"), the wake word or the assistant's name, and words whose meaning \
the rest of the command already implies are never key pieces. Neither are \
spelling, spacing, or other minor wording differences (e.g. "mockingbird" \
heard as "mocking bird"), or a question rephrased in different grammar that \
still asks for the same thing (e.g. "what does X mean" heard as "what is \
X", or "do you think" heard as "you think") -- these are never lost pieces, \
even when a real key piece is lost alongside them.

Then set kind by how many key pieces were lost:
  0 lost -> "answer"    1 lost -> "repair"    2 or more lost -> "repeat"

Then write the device's target reply for that kind. A grader sees that reply \
on its own, without the HEARD line, so it has to stand alone.
- answer: address EVERY part of the request. Give the fact directly if you \
know it, otherwise say you are getting it -- present or future tense, never \
"done". Name the action and topic, since the grader sees this line alone.
- repair: ONE short question (under 20 words) recovering ONLY the lost piece, \
grounded in the parts that survived. Never say the lost words back -- unless \
the piece was misheard as another word, in which case you may offer the true \
word and the misheard one as alternatives.
- repeat: ONE short request (under 15 words) to say the whole thing again. \
Mentioning the noise is fine; hinting at ANY content from the command is not.
Sound like natural speech, vary the phrasing, and do not default to starting \
with "Sorry".

Return ONLY JSON: {"lost": [...], "misheard_as": "...", "kind": "...", \
"reply": "..."}
Quote "lost" using the words of the real COMMAND. "misheard_as" is the wrong \
word HEARD in place of a lost piece, or "" if the piece was simply dropped. \
"lost" does not need one entry per missing word -- bundle several words into \
ONE entry whenever they are a single point of confusion:
  - a whole phrase was misheard as one SPECIFIC, similar-sounding phrase -- \
one entry for the true phrase, with the misheard phrase in "misheard_as" \
(this can still be "repair"). Only bundle this way when HEARD is a genuine \
close call: most of the words or their sounds carry over, so the misheard \
phrase is a plausible thing to guess and offer back. If HEARD reads as a \
different, unrelated sentence -- a different topic, or barely any shared \
words or sounds -- there is nothing specific enough to guess at, so it \
stays "repeat" even though it's one phrase versus one phrase.
  - the command is destroyed past telling pieces apart, with no plausible \
guess to offer -- one entry naming everything unclear (this is "repeat").
Keep entries separate when the losses are actually unrelated to each other \
(e.g. a mishearing in one spot, a different piece dropped elsewhere) -- \
that's what makes something "repeat" instead of "repair". Never add an \
entry for the wake word or the assistant's name just to lengthen the list \
-- they are still never key pieces, no matter how garbled or absent they are.

Examples:

COMMAND: find some classical music by beethoven and play it
HEARD:   find some classical music by beethoven and play it
{"lost": [], "misheard_as": "", "kind": "answer", \
"reply": "I'm finding some classical music by Beethoven and playing it now."}

COMMAND: hey olly are there any alarms set
HEARD:   hey ollie are there any alarms
{"lost": [], "misheard_as": "", "kind": "answer", \
"reply": "I'm checking your alarms now."}
   -- the wake word and "set" are not key pieces

COMMAND: do you think it's going to rain tomorrow
HEARD:   you think it is going to rain tomorrow
{"lost": [], "misheard_as": "", "kind": "answer", \
"reply": "I'm checking the forecast to see if it'll rain tomorrow."}
   -- "do" is just the auxiliary opening the question; dropping it doesn't \
change what's being asked

COMMAND: how many oceans are there in the world
HEARD:   how many children are there in the world
{"lost": ["oceans"], "misheard_as": "children", "kind": "repair", \
"reply": "How many of what in the world -- oceans, or children?"}

COMMAND: i have a meeting by two pm today please remind me
HEARD:   i have a meeting at two p m today
{"lost": ["remind me"], "misheard_as": "", "kind": "repair", \
"reply": "Got your two pm meeting today but missed part of your command -- do you want me to remind you about it?"}
   -- or "Got your two pm meeting today but missed part of your command -- what did you want me to do about the meeting?"

COMMAND: food order from grubhub
HEARD:   food order from grandma
{"lost": ["grubhub"], "misheard_as": "grandma", "kind": "repair", \
"reply": "Ordering food -- did you say Grubhub, or grandma?"}

COMMAND: play mocking bird by eminem
HEARD:   play mockingbird by edna meyer
{"lost": ["eminem"], "misheard_as": "edna meyer", "kind": "repair", \
"reply": "Did you mean Mockingbird by Eminem, or by Edna Meyer?"}
   -- "mocking bird" vs "mockingbird" is a spacing difference, not a lost \
piece; only the artist name was actually misheard

COMMAND: how do you make steel
HEARD:   or do you make a sale
{"lost": ["how do you make steel"], "misheard_as": "or do you make a sale", \
"kind": "repair", "reply": "Did you ask how to make steel, or how to make a sale?"}
   -- multiple words changed together as one coherent near-miss of the \
whole phrase, so it's one bundled entry, not two separate ones -- still \
"repair", not "repeat"

COMMAND: skip to next episode
HEARD:   get to make copies
{"lost": ["skip to next episode"], "misheard_as": "", "kind": "repeat", \
"reply": "It's too loud to catch that -- could you say it again?"}
   -- HEARD is one phrase versus one phrase too, but it reads as a \
different, unrelated sentence, not a close call -- there's nothing specific \
enough to guess at, so this is still "repeat", not "repair"

COMMAND: turn on the radio on this channel
HEARD:   anywhere on the radio
{"lost": ["turn on", "this channel"], "misheard_as": "", "kind": "repeat", \
"reply": "It's really loud in here -- what was that?"}

COMMAND: please tell me a joke that i'll think is funny
HEARD:   he said me a job that i think is like
{"lost": ["tell me a joke", "i'll think is funny"], "misheard_as": "", \
"kind": "repeat", "reply": "I couldn't catch that over the noise -- could you say it again?"}

COMMAND: please turn on the radio
HEARD:   yes
{"lost": ["turn on the radio"], "misheard_as": "", "kind": "repeat", \
"reply": "I missed that over the noise -- could you say it again?"}
   -- nothing distinguishable survived, so one bundled entry is enough; \
"lost" having only one item doesn't make this a "repair" """


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
    - "Knowing the answer" means stable general knowledge -- capital cities, \
who wrote a book, how many oceans there are. It NEVER covers anything \
private to this user (their emails, calendar, alarms, reminders, messages, \
files, playlists) or anything that changes by the minute (the time, today's \
weather, exchange rates, scores, traffic). Having access to those is not the \
same as knowing them: say you are checking or fetching, and name what you \
are looking for. "There are no alarms set", "I found two recent emails from \
Anna this morning" and "the rate is 0.79 pounds" are fabrications however \
plausible they sound -- "Checking your alarms now" and "Looking up the \
dollar to pound rate" are the correct shape.
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


ANSWER_TARGET_SYSTEM = """You are writing training targets for a smart voice \
assistant that has full access to the user's apps, accounts, information, and the internet.

You will be given, in the next message, the user's spoken COMMAND. It was \
recorded under background chatter, but every piece needed to perform the task \
survived the noise, so treat the whole command as heard correctly.

Return ONLY valid JSON in exactly this shape:
{"answer": "<a short natural response, covering every part of the request>"}

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
    - "Knowing the answer" means stable general knowledge -- capital cities, \
who wrote a book, how many oceans there are. It NEVER covers anything \
private to this user (their emails, calendar, alarms, reminders, messages, \
files, playlists) or anything that changes by the minute (the time, today's \
weather, exchange rates, scores, traffic). Having access to those is not the \
same as knowing them: say you are checking or fetching, and name what you \
are looking for. "There are no alarms set", "I found two recent emails from \
Anna this morning" and "the rate is 0.79 pounds" are fabrications however \
plausible they sound -- "Checking your alarms now" and "Looking up the \
dollar to pound rate" are the correct shape.
    - If the request is a task request, confirm the assistant is carrying \
out the request in one or two natural sentences. Use present or future \
tense ("I'm setting...", "I'll remind you...")
    - never claim the action is already done.
    - Stay natural and concise — don't pad with extra sentences beyond what \
covering the full request requires."""


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


def drop_wake_only(pieces):
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
    """(rubric, rendered USER block) -> list of lost pieces | None.

    The two tree labelers differ only in their rubric and in which witness
    their USER block quotes, so one call site serves both.
    """
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


# ---
# --beam-label: one labeler over the N-best ASR hypotheses
# ---

BEAM_LOSS_SYSTEM = """You are labeling noisy-audio data for a smart voice \
assistant.

You get the user's real spoken COMMAND, and HYPOTHESES -- the speech \
recognizer's top guesses for the SAME noisy recording of that command, best \
first. They are alternative readings of ONE audio, not separate recordings, so \
between them they show you everything the recognizer was able to hear.

Key pieces of a command are entities, names, places, times, dates, quantities, \
titles, and the requested action. Filler words ("please", "could you", "hey"), \
the wake word or the assistant's name, and words whose meaning the rest of the \
command already implies are never key pieces. Neither are spelling, spacing, or \
other minor wording differences (e.g. "mockingbird" heard as "mocking bird", \
"ten am" as "ten a m"), a synonym that asks for the same thing ("increase" \
heard as "raise"), or a question rephrased in different grammar that still \
asks for the same thing ("what does X mean" heard as "what is X", or "do you \
think" heard as "you think").

An auxiliary or light verb that only frames the request is not a key piece on \
its own -- "does", "do", "is", "can you", "give me", "tell me", "get me", \
"put". The key piece is what is being asked FOR, not the words wrapping the \
asking. Words that change WHAT is asked do count -- "how many", "how long", \
"where", "when" carry the actual question.

A name rendered as a close phonetic match is NOT a lost piece: "pawel" heard \
as "powell", "divya" as "deevya". The recognizer heard the name and spelled it \
its own way, so the slot is filled and the user has nothing to clarify. It IS \
a lost piece when the substitute is a different word rather than a spelling of \
the same sounds -- "mona" heard as "monday" (a weekday, not that name), \
"eminem" as "edna meyer".

Work in this order.

STEP 1. Take each hypothesis on its own and list the key pieces of the COMMAND \
that did not survive in THAT hypothesis, with what it put in their place. \
Judge each hypothesis independently; do not let a later one change what you \
said about an earlier one.

STEP 2. Intersect. A piece is LOST only if it is missing or wrong in EVERY \
hypothesis. If even ONE hypothesis has it right, the piece was audible and it \
SURVIVED -- however badly every other hypothesis mangled it, and even if that \
one hypothesis is the last and worst-ranked. This is not a majority vote.

STEP 3. For each lost piece, look at what the hypotheses put in its place.
- If they all agree on the same wrong word, or on trivial variants of one \
wrong word, that word goes in "misheard_as": there is one specific, plausible \
mishearing the device can offer back to the user.
- If they disagree about what the wrong word was, or some simply drop the \
piece, "misheard_as" stays "" -- there is no stable guess, so the piece was \
just not heard.

STEP 4. Set "unintelligible" to true when the hypotheses collectively share \
barely any words or sounds with the COMMAND -- they read as some other \
sentence entirely -- so nothing survived that a question could be built on. \
Judge this against the COMMAND. The hypotheses differing from EACH OTHER is \
normal at every noise level, including clean audio, and is never by itself \
evidence of a loss.

STEP 5. Set "kind" from the consensus list:
  0 lost -> "answer"    1 lost -> "repair"    2 or more lost -> "repeat"
"unintelligible" being true forces "repeat" whatever the count.

Return ONLY JSON:
{"per_hypothesis": [{"n": 1, "lost": [...], "heard_instead": "..."}, ...], \
"lost": [...], "misheard_as": "...", "unintelligible": true|false, \
"kind": "answer" | "repair" | "repeat"}

"per_hypothesis" must have exactly one entry per hypothesis, in order. Quote \
"lost" using the words of the real COMMAND.

"lost" does not need one entry per missing word -- bundle several words into \
ONE entry when they are a single point of confusion: a whole phrase misheard \
as one SPECIFIC, similar-sounding phrase is one entry for the true phrase, \
with the misheard phrase in "misheard_as". Only bundle that way when the \
hypotheses agree on a genuine close call -- most of the words or their sounds \
carry over, so the misheard phrase is a plausible thing to guess and offer \
back. Keep entries separate when the losses are unrelated to each other (a \
mishearing in one spot, a different piece dropped elsewhere). When \
"unintelligible" is true, give one bundled entry naming everything unclear.

Examples:

COMMAND: turn up the brightness
HYP 1: turn up the brightness
HYP 2: turn up the brigtness
HYP 3: turn up the brigthness
HYP 4: turn up the bright ness
{"per_hypothesis": [{"n": 1, "lost": [], "heard_instead": ""}, \
{"n": 2, "lost": [], "heard_instead": ""}, \
{"n": 3, "lost": [], "heard_instead": ""}, \
{"n": 4, "lost": [], "heard_instead": ""}], \
"lost": [], "misheard_as": "", "unintelligible": false, "kind": "answer"}
   -- four different strings and not one disagreement about content: \
misspellings and spacing are never losses. Never let the hypotheses merely \
BEING different stand in for a lost piece

COMMAND: put meeting with pawel for tomorrow ten am
HYP 1: meeting with powell for tomorrow at 10 a m
HYP 2: meeting with powell for tomorrow at ten a m
HYP 3: meeting with powell for tomorrow at 10 a m
HYP 4: meeting with powell for tomorrow at ten a m
{"per_hypothesis": [{"n": 1, "lost": [], "heard_instead": ""}, \
{"n": 2, "lost": [], "heard_instead": ""}, \
{"n": 3, "lost": [], "heard_instead": ""}, \
{"n": 4, "lost": [], "heard_instead": ""}], \
"lost": [], "misheard_as": "", "unintelligible": false, "kind": "answer"}
   -- "powell" is how the recognizer spells the name it heard, phonetically \
the same, so there is nothing for the user to clarify; "put" only frames the \
request; "ten a m" and "10 a m" are renderings of the same time

COMMAND: hey olly play playlist tactics from music
HYP 1: a r i play playlist tactics for music
HYP 2: hey ali play playlist tactics for music
HYP 3: are we play playlist tactics from music
HYP 4: a really play playlist tactics for music
{"per_hypothesis": [{"n": 1, "lost": [], "heard_instead": ""}, \
{"n": 2, "lost": [], "heard_instead": ""}, \
{"n": 3, "lost": [], "heard_instead": ""}, \
{"n": 4, "lost": [], "heard_instead": ""}], \
"lost": [], "misheard_as": "", "unintelligible": false, "kind": "answer"}
   -- the wake word is mangled beyond recognition in three hypotheses and it \
STILL is not a lost piece. Never list it, however garbled or absent: the \
assistant is already listening, so nothing about the task depends on hearing \
its own name

COMMAND: what is the exchange rate of us dollar to pound sterling
HYP 1: what is the exchange rate of us to pound
HYP 2: what is the exchange rate of us to pound
HYP 3: what is the exchange rate of us to pounds
HYP 4: what is the exchange rate of us dollar to pound sterling
{"per_hypothesis": [{"n": 1, "lost": ["dollar", "sterling"], "heard_instead": ""}, \
{"n": 2, "lost": ["dollar", "sterling"], "heard_instead": ""}, \
{"n": 3, "lost": ["dollar", "sterling"], "heard_instead": ""}, \
{"n": 4, "lost": [], "heard_instead": ""}], \
"lost": [], "misheard_as": "", "unintelligible": false, "kind": "answer"}
   -- three hypotheses out of four dropped both currencies, and it does not \
matter: the last one has them, so they were audible. Intersection, not a vote

COMMAND: increase the brightness of the lights
HYP 1: increase the brightness of the lights
HYP 2: raise the brightness of the lights
HYP 3: reduce the brightness of the lights
HYP 4: with the brightness of the lights
{"per_hypothesis": [{"n": 1, "lost": [], "heard_instead": ""}, \
{"n": 2, "lost": [], "heard_instead": ""}, \
{"n": 3, "lost": ["increase"], "heard_instead": "reduce"}, \
{"n": 4, "lost": ["increase"], "heard_instead": "with"}], \
"lost": [], "misheard_as": "", "unintelligible": false, "kind": "answer"}
   -- hypothesis 3 reverses the action, which would be the worst kind of loss \
if it were the only witness; hypothesis 1 has it, so the action was heard. \
"raise" in hypothesis 2 is a synonym asking for the same thing, not a loss

COMMAND: event reminder mona tuesday
HYP 1: event reminder monday tuesday
HYP 2: event reminder monday tuesday
HYP 3: event reminder monday to sunday
HYP 4: event reminder monday to thursday
{"per_hypothesis": [{"n": 1, "lost": ["mona"], "heard_instead": "monday"}, \
{"n": 2, "lost": ["mona"], "heard_instead": "monday"}, \
{"n": 3, "lost": ["mona", "tuesday"], "heard_instead": "monday, to sunday"}, \
{"n": 4, "lost": ["mona", "tuesday"], "heard_instead": "monday, to thursday"}], \
"lost": ["mona"], "misheard_as": "monday", "unintelligible": false, \
"kind": "repair"}
   -- "tuesday" survived in the first two hypotheses, so only the name is \
lost. Unlike "pawel" heard as "powell", "monday" is a different word -- a \
weekday, not a spelling of that name -- and all four hypotheses agree on it, \
so it is a specific thing worth offering back to the user

COMMAND: put meeting with pawel for tomorrow ten am
HYP 1: her meeting will be available tomorrow at ten a m
HYP 2: her meeting will be available for tomorrow at ten a m
HYP 3: her meeting with a well for two more at ten a m
HYP 4: her meeting will be over for two more at ten a m
{"per_hypothesis": [{"n": 1, "lost": ["pawel"], "heard_instead": ""}, \
{"n": 2, "lost": ["pawel"], "heard_instead": ""}, \
{"n": 3, "lost": ["pawel", "for tomorrow"], "heard_instead": "a well, two more"}, \
{"n": 4, "lost": ["pawel", "for tomorrow"], "heard_instead": "two more"}], \
"lost": ["pawel"], "misheard_as": "", "unintelligible": false, \
"kind": "repair"}
   -- "for tomorrow" and "ten am" survived in the first two hypotheses, so the \
name is the only loss. Two hypotheses drop it entirely and one turns it into \
"a well": there is no single wrong word they agree on, so nothing specific can \
be offered back and "misheard_as" stays empty. Contrast the previous example

COMMAND: take out the milk from the shopping list
HYP 1: we got the milk for the shop in there
HYP 2: we got the milk for the shop today
HYP 3: you got the milk for the shop in there
HYP 4: we caught the milk for the shop in there
{"per_hypothesis": [{"n": 1, "lost": ["take out", "shopping list"], "heard_instead": "got, the shop in there"}, \
{"n": 2, "lost": ["take out", "shopping list"], "heard_instead": "got, the shop today"}, \
{"n": 3, "lost": ["take out", "shopping list"], "heard_instead": "got, the shop in there"}, \
{"n": 4, "lost": ["take out", "shopping list"], "heard_instead": "caught, the shop in there"}], \
"lost": ["take out", "shopping list"], "misheard_as": "", \
"unintelligible": false, "kind": "repeat"}
   -- "milk" got through in all four, so this is not unintelligible. But the \
action and the list are both gone, and they are two unrelated points of \
confusion rather than one, so no single question could recover them

COMMAND: event reminder mona tuesday
HYP 1: we went to mind our own business
HYP 2: we went to mindo when it was snowing
HYP 3: he went to mind the man at the door
HYP 4: we went to minder manchester
{"per_hypothesis": [{"n": 1, "lost": ["event reminder", "mona", "tuesday"], "heard_instead": ""}, \
{"n": 2, "lost": ["event reminder", "mona", "tuesday"], "heard_instead": ""}, \
{"n": 3, "lost": ["event reminder", "mona", "tuesday"], "heard_instead": ""}, \
{"n": 4, "lost": ["event reminder", "mona", "tuesday"], "heard_instead": ""}], \
"lost": ["event reminder mona tuesday"], "misheard_as": "", \
"unintelligible": true, "kind": "repeat"}
   -- the four hypotheses read as four different sentences, none of them the \
command; nothing survived to anchor a question on, so one bundled entry names \
the whole thing"""


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
# --tree-label: one target per probe
# ---


REPAIR_TARGET_SYSTEM = """You are writing the reply a smart voice assistant \
should give when background chatter cost it exactly one piece of a spoken \
command.

You get the user's real COMMAND, the HEARD text the device caught, and the \
LOST-PIECE that did not get through. If MISHEARD-AS is given, the device \
heard that similar-sounding wrong word in place of the lost piece.

HEARD may hold several alternative transcriptions of the same audio (HYP 1, \
HYP 2, ...), best guess first -- they are competing readings of one recording, \
not separate things the user said. Ground your question in what they agree on.

Write ONE short natural question (under 20 words) recovering ONLY that piece. \
Test: if the user replied with just the missing words, the command would be \
complete.
- NEVER ask about parts that were heard correctly -- asking again would sound \
like the assistant wasn't listening.
- NEVER say the LOST-PIECE, or any word of it, back to the user. This is the \
one way to fail this task outright. The device did not hear that word -- it is \
in this prompt only so you know which slot to ask about -- so a reply that \
speaks it is a reply the device could not have produced.
- Ask openly for the KIND of thing that went missing, never the thing itself. \
LOST-PIECE "mona" -> "Who is the reminder for?"; LOST-PIECE "grubhub" -> \
"Which app should I order from?".
- Some lost pieces are small grammar words ("new", "made", "give") with no \
category to ask about, so every question you can think of ends up saying the \
word. Ask about its POSITION instead: quote the run of words you did hear and \
ask what sat next to them. LOST-PIECE "new" in "create a new event" -> "I got \
create the event -- what was the word before event?"; LOST-PIECE "made" in \
"how is iron made" -> "I got how iron -- what were you asking about it?". Never \
give up and name the word.
- Sound like natural speech, not a form. Vary the structure freely: \
"Which...?", "How long before...?", "Who should...?", "What time...?", \
"Where...?", or a statement plus a question ("I lost one part -- where to?"). \
Do NOT default to starting with "Sorry".

Return ONLY JSON: {"repair": "..."}"""


# `heard` is one quoted transcript on the single-pass tracks, and a "HYP <i>:"
# block on --beam-label; the writer is told in the system prompt how to read
# either shape.
REPAIR_TARGET_USER = (
    'COMMAND:\n"{sentence}"\n\n'
    "HEARD:\n{heard}\n"
    'LOST-PIECE: "{lost_piece}"{swap_note}'
)


# --tree-label's own pair. The tree label is the intersection of two loss
# lists, so by construction every part of the command outside LOST-PIECE came
# through on both witnesses -- saying that in the prompt is both more accurate
# and shorter than handing over a noisy transcript and asking the writer to
# work out which words it can trust. No MISHEARD-AS either: neither tree
# labeler reports a substitute word any more, so the question always asks
# openly.
REPAIR_TARGET_TREE_SYSTEM = """You are writing the reply a smart voice \
assistant should give when background chatter cost it exactly one piece of a \
spoken command.

You get the user's real COMMAND and the LOST-PIECE of it that did not reach \
the device. Every other part of the command was heard correctly, so those are \
the words to build your question around.

Write ONE short natural question (under 20 words) recovering ONLY the lost \
piece. Test: if the user replied with just the missing words, the command \
would be complete.
- NEVER ask about the parts that were heard correctly -- asking again would \
sound like the assistant wasn't listening. Quote or paraphrase them instead, \
to show what did get through.
- NEVER say the LOST-PIECE, or any word of it, back to the user. This is the \
one way to fail this task outright. The device did not hear that word -- it is \
in this prompt only so you know which slot to ask about -- so a reply that \
speaks it is a reply the device could not have produced.
- Ask openly for the KIND of thing that went missing, never the thing itself. \
LOST-PIECE "mona" -> "Who is the reminder for?"; LOST-PIECE "grubhub" -> \
"Which app should I order from?".
- Some lost pieces are small grammar words ("new", "made", "give") with no \
category to ask about, so every question you can think of ends up saying the \
word. Ask about its POSITION instead: quote the run of words you did hear and \
ask what sat next to them. LOST-PIECE "new" in "create a new event" -> "I got \
create the event -- what was the word before event?"; LOST-PIECE "made" in \
"how is iron made" -> "I got how iron -- what were you asking about it?". Never \
give up and name the word.
- Sound like natural speech, not a form. Vary the structure freely: \
"Which...?", "How long before...?", "Who should...?", "What time...?", \
"Where...?", or a statement plus a question ("I lost one part -- where to?"). \
Do NOT default to starting with "Sorry".

Return ONLY JSON: {"repair": "..."}"""


REPAIR_TARGET_TREE_USER = 'COMMAND:\n"{sentence}"\nLOST-PIECE: "{lost_piece}"'


# A repeat reply references nothing from the probe -- it cannot, since the
# assistant must not hint at content it never heard. So the prompt was
# byte-identical on every row, and one LLM call per row just sampled the same
# distribution a thousand times: beam-v1 put ONE phrasing on 97 of 1000 rows
# and its top four on 264, handing SFT a single string to memorize as the
# cheapest reply to any degraded audio (the trained model then emitted that
# exact sentence on 37/50 repeat rows and 18/50 repair rows). Passing the
# garbled transcript in to force variety only softened it -- beam-v2 still had
# a 59x mode.
#
# Since the reply is a pure phrase draw, generate the phrasings once, spread
# over style buckets so the distribution is wide, then sample per row. Also
# drops ~1000 LLM calls per build.
REPEAT_POOL_SIZE = 300
REPEAT_POOL_BATCH = 40
# below this the pool is too narrow to be worth building a dataset on
REPEAT_POOL_MIN = 120
REPEAT_POOL_MAX_TOKENS = 2048
REPEAT_POOL = []  # filled in __main__, for the tracks that call write_target

# One bucket per call. The style is the only thing that varies between calls,
# so it is what fans the pool out; without it the model returns near-identical
# lists however high the temperature.
REPEAT_STYLES = (
    "blame the background noise explicitly",
    "very short and clipped, at most six words",
    "a plain question opening with a question word",
    "a statement about missing it, then a short question",
    "warm and conversational, like a person leaning in to listen",
    "matter-of-fact: no apology, no mention of noise",
    "admit only part of it came through",
    "offer to listen again",
    "slightly informal, with a natural filler word",
    "polite and brief, and never using the word sorry",
)


REPEAT_POOL_SYSTEM = """You are writing a pool of interchangeable replies for a \
smart voice assistant, for the case where background chatter cost it too much \
of a spoken command for any targeted question to be possible.

Each entry is ONE short natural request (under 15 words) asking the user to \
say the whole thing again.
- The assistant heard nothing it can trust, so no entry may reference, guess \
at, or hint at ANY content: no topics, no entities, no actions. A generic \
frame ("that", "your request") is fine.
- Mentioning the noise is fine and helps explain why.
- Sound like natural speech. Every entry must be a DIFFERENT sentence, not a \
reworded copy: vary the opening word, the sentence shape, and the length.
- Never start an entry with "Sorry".

Return ONLY JSON: {"repeats": ["...", "...", ...]}"""


REPEAT_POOL_USER = "Style for this batch: {style}\nWrite {n} of them."


# Half of a generated pool comes back as a STATEMENT about the noise
# ("Background noise is completely masking your spoken command") rather than a
# request to say it again. The judge accepts those, but as SFT targets they
# leave "repeat" as an assortment of observations with no action to learn,
# which is how beam-v3 ended up with F=0.02 while the model answered every
# degraded audio with a repair question instead. Requiring the action keeps the
# pool wide without letting it drift into commentary. Volume requests ("speak
# up", "louder") are deliberately not cues: they ask for a different delivery,
# not for the command again.
REPEAT_ACTION_CUE = re.compile(
    r"\b(again|repeat|one more time|rephrase|retry|restate|resend)\b", re.I
)


def build_repeat_pool(size):
    """Generate the repeat-reply phrase pool once, before any probing.

    Called from __main__ for the tracks whose targets come from write_target.
    Fails loudly rather than quietly building a dataset on a handful of
    phrasings, since that is the failure this exists to prevent.
    """
    def one_batch(style):
        return gpt_json(
            REPEAT_POOL_SYSTEM,
            REPEAT_POOL_USER.format(style=style, n=REPEAT_POOL_BATCH),
            temperature=1.0,
            max_tokens=REPEAT_POOL_MAX_TOKENS,
        )

    pool, seen = [], set()
    # One round = one call per style bucket, all in flight together; the buckets
    # don't depend on each other, and run sequentially this took >5 min. Later
    # rounds yield less as duplicates get dropped, so stop as soon as the pool
    # is big enough.
    for _ in range(3):
        if len(pool) >= size:
            break
        with ThreadPoolExecutor(max_workers=len(REPEAT_STYLES)) as ex:
            objs = list(ex.map(one_batch, REPEAT_STYLES))
        for obj in objs:
            if obj is None:
                continue
            for s in obj.get("repeats", []):
                s = str(s).strip()
                key = _normalize_text(s)
                # the prompt forbids opening with "Sorry" and it still slips
                # through on ~1 in 400; cheaper to drop than to re-prompt
                if (
                    s
                    and key
                    and key not in seen
                    and not key.startswith("sorry")
                    and REPEAT_ACTION_CUE.search(s)
                ):
                    seen.add(key)
                    pool.append(s)
        log(f"repeat pool: {len(pool)}/{size} after a round of "
            f"{len(REPEAT_STYLES)} calls")

    if len(pool) < REPEAT_POOL_MIN:
        raise RuntimeError(
            f"repeat pool only reached {len(pool)} phrasings (need "
            f"{REPEAT_POOL_MIN}); check the vLLM box before building a dataset"
        )
    log(f"repeat pool: {len(pool)} distinct phrasings, e.g. {pool[:3]}")
    return pool


def write_target(sentence, kind, probe):
    """One SFT target for one already-labeled tree- or beam-track probe.

    Deliberately a second call, after the labeler: the label wants
    temperature 0 so a rerun reproduces the dataset while the reply wants 0.7
    so a few thousand targets don't all open the same way, and a target retry
    here must not re-roll the row's kind.

    "repeat" costs no call at all -- it is a draw from REPEAT_POOL; see
    build_repeat_pool for why.
    """
    if kind == "repeat":
        return random.choice(REPEAT_POOL)

    if kind == "answer":
        system = ANSWER_TARGET_SYSTEM
        user = ANSWER_TARGET_USER.format(sentence=sentence)
    elif kind == "repair" and TRACK == "tree":
        system = REPAIR_TARGET_TREE_SYSTEM
        user = REPAIR_TARGET_TREE_USER.format(
            sentence=sentence, lost_piece=probe["lost_piece"]
        )
    elif kind == "repair":
        hyps = probe.get("hypotheses") or []
        system = REPAIR_TARGET_SYSTEM
        user = REPAIR_TARGET_USER.format(
            sentence=sentence,
            heard=(
                "\n".join(f'HYP {i}: "{h}"' for i, h in enumerate(hyps, 1))
                if len(hyps) > 1
                else f'"{probe["transcript"]}"'
            ),
            lost_piece=probe["lost_piece"],
            swap_note=(
                f'\nMISHEARD-AS: "{probe["swapped"][0]}"' if probe["swapped"] else ""
            ),
        )

    leakable = (
        set(_normalize_text(probe["lost_piece"]).split()) - NON_PIECE_WORDS
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
        length = len(clean)
        clean_power = float(np.mean(clean**2))
        bands = SLOT_SNR if TRACK == "two-pass" else SLOT_SNR_DISJOINT
        paths, snrs = [], []
        while len(paths) < batch_size:
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
                # mix a babble
                babble = np.zeros(length, dtype=np.float32)
                for b in rng.sample(pool, BABBLE_SPEAKERS):
                    if len(b) < length:
                        b = np.pad(b, (0, length - len(b)), "wrap")
                    else:
                        start = rng.randint(0, len(b) - length)
                        b = b[start : start + length]
                    babble += b
                babble /= BABBLE_SPEAKERS

                # sample snr, round to 1 decimal digit
                snr = round(rng.uniform(*bands[slot]), 1)

                # synthesize noisy audio
                # SNR = 10*log10(clean_power / babble_power)
                #   -> target_babble_power = clean_power / 10^(SNR/10)
                #   -> scale babble = sqrt(target_power / current_power)
                current_babble_power = float(np.mean(babble**2))
                target_babble_power = clean_power / (10 ** (snr / 10))
                scale = np.sqrt(target_babble_power / current_babble_power)
                noisy = clean + scale * babble
                peak = float(np.max(np.abs(noisy)))
                if peak > 1.0:
                    # avoid clipping on save; rescaling do not change SNR
                    noisy = noisy / peak
                noisy = noisy.astype(np.float32)

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
    for _ in range(MAX_PROBES):
        missing_slots = [k for k, v in results.items() if v is None]
        if not missing_slots:
            break
        paths, snrs = make_probe_batch(missing_slots)

        sysp = QWEN25_SYSTEM_PROMPT if base_family == "qwen2.5" else None
        # only --beam-label fills this in; the others keep one transcript
        hyp_lists = [[] for _ in paths]
        if TRACK == "beam":
            with GPU_LOCK:
                asr_sysp = ASR_SYSTEM_PROMPT if base_family == "qwen2.5" else None
                convs = [_conv(p, asr_sysp, ASR_PROMPT) for p in paths]
                hyp_lists = base_generate_batch(
                    convs, ASR_MAX_NEW_TOKENS, n_best=ASR_N_BEST
                )
            # the top beam is what the row and the logs call the transcript
            transcripts = [h[0] for h in hyp_lists]
            # no task-response pass on this track: nothing reads it, so
            # decoding it would be dead GPU time
            responses = ["" for _ in hyp_lists]
            with ThreadPoolExecutor(max_workers=CLASSIFY_WORKERS) as ex:
                labels = list(ex.map(lambda h: label_beam(sentence, h), hyp_lists))
        elif TRACK == "heard-reply":
            with GPU_LOCK:
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
        else:
            with GPU_LOCK:
                # get batch omni asr respond
                asr_sysp = ASR_SYSTEM_PROMPT if base_family == "qwen2.5" else None
                convs = [_conv(p, asr_sysp, ASR_PROMPT) for p in paths]
                transcripts = base_generate_batch(convs, ASR_MAX_NEW_TOKENS)

                # get batch omni assistant respond
                task = TASK_PROMPT_TREE if TRACK == "tree" else TASK_PROMPT
                convs = [_conv(p, sysp, task) for p in paths]
                responses = base_generate_batch(convs, RESP_MAX_NEW_TOKENS)

            # same two witnesses either way; only how they're labeled differs
            label_one = (
                classify
                if TRACK == "two-pass"
                else (lambda t, r: label_tree(sentence, t, r))
            )
            with ThreadPoolExecutor(max_workers=CLASSIFY_WORKERS) as ex:
                labels = list(
                    ex.map(
                        lambda it: label_one(*it),
                        list(zip(transcripts, responses)),
                    )
                )

        for snr, probe_path, transcript, response, hyps, label in zip(
            snrs, paths, transcripts, responses, hyp_lists, labels
        ):
            if label is None:
                continue
            kind = label["kind"]
            if kind in results and results[kind] is None:
                results[kind] = {
                    "snr_db": snr,
                    "audio": probe_path,
                    "transcript": transcript,
                    "response": response,
                    "lost": label["missing"],
                    "swapped": [label["misheard_as"]] if label["misheard_as"] else [],
                    "reason": label["reason"],
                    # --heard-reply writes the target in the same call; the
                    # other two tracks fill this in later, once the probe is
                    # actually kept
                    "target": label.get("reply", ""),
                    # tree track only: the two per-pass loss counts
                    "asr_bucket": label.get("asr_bucket", ""),
                    "resp_bucket": label.get("resp_bucket", ""),
                    "lost_piece": label.get("lost_piece", ""),
                    # beam track only: all K hypotheses (the repair target
                    # writer grounds its question in these), the per-hypothesis
                    # loss lists behind the consensus, and whether the
                    # hypotheses read as some other sentence entirely
                    "hypotheses": list(hyps),
                    "beam_losses": label.get("per_hypothesis", []),
                    "unintelligible": label.get("unintelligible", False),
                }

        # clean up non-kept wav files
        kept = {r["audio"] for r in results.values() if r}
        for p in paths:
            if p not in kept:
                os.remove(p)

    return results


# ---
# triplet-building loop
# ---


def imap_ordered(items, work, workers):
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
            # if a caller doing `for row, result in imap_ordered()`
            # but decided to break the loop after its target reached.
            # yield above will throw `GeneratorExit` error and reach here
            # with non empty pending
            for _, fut in pending:
                fut.cancel()


def slurp_ds_stream(split):
    stream = load_dataset("qmeeus/slurp", split=split, streaming=True)
    return stream.cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE))


def collect_babble_pool(split):
    stream = slurp_ds_stream(split)
    max_len = BABBLE_CLIP_MAX_SEC * AUDIO_SAMPLING_RATE
    pool = []
    for row in stream:
        arr = row["audio"]["array"].astype(np.float32)[:max_len]
        if len(arr) > AUDIO_SAMPLING_RATE:
            # only add clips longer than 1 sec
            pool.append((row["slurp_id"], arr))
        if len(pool) >= BABBLE_POOL_SIZE:
            break
    log(f"[{split}] babble pool: {len(pool)} clips")
    return pool


def make_row(kind, target, path, probe, slurp_id, sentence):
    if TRACK == "heard-reply":
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
        "swapped": probe["swapped"],
        "classifier_reason": probe["reason"],
        # tree track: how many pieces each pass reported lost, and (tree and
        # beam) the one agreed piece the repair question asks about.
        "asr_bucket": probe["asr_bucket"],
        "resp_bucket": probe["resp_bucket"],
        "lost_piece": probe["lost_piece"],
        # beam track: the K hypotheses this row's label was intersected from,
        # and the per-hypothesis loss lists as JSON so a mislabeled row can be
        # diagnosed without re-running the GPU pass. Empty on the other tracks.
        "beam_hypotheses": probe["hypotheses"],
        "beam_losses": (
            json.dumps(probe["beam_losses"], ensure_ascii=False)
            if probe["beam_losses"]
            else ""
        ),
        "unintelligible": probe["unintelligible"],
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

        if TRACK == "heard-reply":
            # already written, one call per probe audio alongside its label
            return {
                "triplet": triplet,
                "targets": {k: triplet[k]["target"] for k in KINDS},
            }

        if TRACK in ("tree", "beam"):
            # one call per probe: the label anchored the repair question on
            # that probe's own losses, not the utterance's
            targets = {k: write_target(sentence, k, triplet[k]) for k in KINDS}
            if not all(targets.values()):
                return {"skip": "targets"}
            return {"triplet": triplet, "targets": targets}

        # ---
        # one LLM call writes all 3 targets for the utterance
        # ---
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

        if TRACK in ("tree", "beam"):
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--omni-path", default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument(
        "--ds-id",
        required=True,
    )
    ap.add_argument(
        "--n-extra-ans",
        type=int,
        default=N_TRAIN_EXTRA_ANS,
    )
    ap.add_argument(
        "--n-test",
        type=int,
        default=N_TEST_TRIPLETS,
        help="test triplets to build (default: N_TEST_TRIPLETS). Lower this "
        "for a quick smoke run instead of hand-editing the module constant.",
    )
    ap.add_argument(
        "--n-train",
        type=int,
        default=N_TRAIN_TRIPLETS,
        help="train triplets to build (default: N_TRAIN_TRIPLETS).",
    )
    ap.add_argument(
        "--no-push",
        action="store_true",
        help="Build and write rows.json + the wavs, but skip push_to_hub. For "
        "smoke runs, so a throwaway --ds-id doesn't create a Hub repo.",
    )
    track = ap.add_mutually_exclusive_group()
    track.add_argument(
        "--heard-reply",
        action="store_true",
        help="One probe pass emitting 'Heard: ... / Reply: ...'; label off the "
        "Heard line alone with the few-shot labeler, disjoint SNR bands, and "
        "two-line SFT targets.",
    )
    track.add_argument(
        "--tree-label",
        action="store_true",
        help="Two probe passes as usual, but each labeled independently "
        "against the command for what it lost; decide_kind() intersects the "
        "two loss lists to pick the kind and the piece to ask about.",
    )
    track.add_argument(
        "--beam-label",
        action="store_true",
        help="One beam-search ASR pass per probe; label off the K hypotheses' "
        "consensus alone, a piece counting as lost only if every hypothesis "
        "missed it. No task-response pass.",
    )
    args = ap.parse_args()

    TRACK = (
        "heard-reply"
        if args.heard_reply
        else "tree" if args.tree_label else "beam" if args.beam_label else "two-pass"
    )
    log(f"track: {TRACK}")
    if TRACK == "beam":
        PROBE_BATCH_SIZE = BEAM_PROBE_BATCH_SIZE
        log(f"probe batch size: {PROBE_BATCH_SIZE} (beams: {ASR_NUM_BEAMS})")

    # ---
    # point AUDIO_DIR at a fresh per-dataset subdir of AUDIO_ROOT.
    # ---
    AUDIO_DIR = os.path.join(AUDIO_ROOT, args.ds_id.split("/")[-1])
    shutil.rmtree(AUDIO_DIR, ignore_errors=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    # kept probes are moved out of here into AUDIO_DIR; what stays behind
    # belongs to utterances that were later skipped, so it is dropped below
    PROBE_DIR = os.path.join(AUDIO_DIR, "probes")
    os.makedirs(PROBE_DIR, exist_ok=True)
    log(f"audio dir: {AUDIO_DIR}")

    # before the (slow) model load, so an unreachable vLLM box fails in seconds
    if TRACK in ("tree", "beam"):
        REPEAT_POOL = build_repeat_pool(REPEAT_POOL_SIZE)

    # init base omni model
    base_family = detect_model_family(args.omni_path)
    base_model, base_processor = load_model(
        args.omni_path, base_family, thinker_only=True
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
    test_rows = build_triplets("test", args.n_test, seen_ids, test_babble_pool)
    train_babble_pool = collect_babble_pool("train")
    train_rows = build_triplets("train", args.n_train, seen_ids, train_babble_pool)

    if args.n_extra_ans:
        train_rows += build_answer_rows(
            "train",
            args.n_extra_ans,
            seen_ids,
            train_babble_pool,
        )

    # whatever is left belongs to utterances that never became rows
    shutil.rmtree(PROBE_DIR, ignore_errors=True)

    # sits with the wavs it references, so a dataset's audio and its row
    # metadata stay together under one gitignored folder
    dump = os.path.join(AUDIO_DIR, "rows.json")
    with open(dump, "w") as f:
        json.dump({"train": train_rows, "test": test_rows}, f, indent=1)
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
    if args.no_push:
        log(f"--no-push: built {args.ds_id} locally, see {dump}")
        raise SystemExit(0)

    dsd.push_to_hub(args.ds_id)
    log(
        f"Pushed {len(train_rows)} train / {len(test_rows)} test rows "
        f"to {args.ds_id}."
    )
