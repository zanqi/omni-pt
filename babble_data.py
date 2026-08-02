"""
Build babble-noise slurp dataset for conversational-repair training.

Three tracks, selected by --heard-reply / --tree-label:

Default (two-pass): for each noisy probe the omni base model produces an ASR
transcript and, separately, a task response. The LLM sees (original sentence,
transcript, response) and labels the probe:
- "answer":     every detail needed to perform the task survived the noise;
                filler only loss
- "repair":     exactly ONE key piece was lost or misheard;
                the rest can be trusted
- "repeat":     more than one key piece lost in both passes

--tree-label: the same two probe passes, but neither labeler sees the other's
pass. label_asr diffs the transcript against the command; label_resp reads
what the reply demonstrates it heard. decide_kind() then resolves the pair of
labels with a fixed table, which also names the one piece a repair question
should anchor on. A clean result from EITHER pass alone means "answer";
"repeat" needs both passes to have failed badly.

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
import shutil
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
from util import QWEN25_SYSTEM_PROMPT, detect_model_family, load_model
from prompts import HEARD_PREFILL, TASK_PROMPT, split_heard_reply, task_prompt

skip = Counter()


def log(*args):
    """print-compatible logging that doesn't break the tqdm bar."""
    tqdm.write(" ".join(str(a) for a in args))


AUDIO_SAMPLING_RATE = 16000

# defend against single long audio causing oom
MAX_AUDIO_SECONDS = 30

N_TRAIN_TRIPLETS = 1000
N_TEST_TRIPLETS = 50
N_TRAIN_EXTRA_ANS = 3000

# Classification + Target generation are served by the local vLLM judge
# box. Its slurm job records the node it landed on in VLLM_HOST_FILE.
TARGET_MODEL = "Qwen/Qwen3.5-122B-A10B-FP8" # "Qwen/Qwen3.6-35B-A3B-FP8"
VLLM_HOST_FILE = "/gscratch/sciencehub/zanqil/vllm_judge/vllm_judge_host.txt"
MASK_DS_ID = "keylazy/slurp-ear-sft"
AUDIO_ROOT = "babble_audio"
AUDIO_DIR = None  # set in __main__ basedon ds name
SEED = 42
ROW_ID = itertools.count(1)

BABBLE_POOL_SIZE = 300
BABBLE_SPEAKERS = 3
BABBLE_CLIP_MAX_SEC = 10  # trim pool clips to save memory

PROBE_BATCH_SIZE = 16
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
TARGET_RETRIES = 3
CLASSIFY_WORKERS = 8  # parallel classifier calls to vLLM

ASR_MAX_NEW_TOKENS = 64
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
ASR_PROMPT = "Transcribe the English audio into text without any punctuation marks." # from Qwen2.5-Omni github cookbooks


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
def base_generate_batch(convs, max_new_tokens, prefill=None):
    """prefill, if given, is literal text appended to the rendered prompt so
    the model continues from there instead of opening with something else
    (e.g. --heard-reply forces HEARD_PREFILL so a refusal on bad audio can't
    skip the required format). The prefill is prepended back onto the decoded
    text, so callers see the same string they'd get if the model had opened
    with it spontaneously."""
    # the omni processor logs a root-logger warning per conversation whenever
    # the system prompt isn't the talker default (our ASR prompt never is). We
    # only decode text, so mute it for just this call -- an int compare in
    # isEnabledFor, cheaper than filtering the emitted records by message.
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
    out = base_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=IM_END_ID,
        pad_token_id=IM_END_ID,
    )
    gen = out[:, inputs["input_ids"].shape[1] :]
    decoded = [
        t.lower().strip()
        for t in base_processor.batch_decode(gen, skip_special_tokens=True)
    ]
    if prefill is not None:
        decoded = [f"{prefill}{d}" for d in decoded]
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


# ---
# --tree-label: two independent per-pass labelers + a decision table
# ---

# The default track hands both passes to ONE classifier call and asks it to
# reconcile them, which needs a long rubric of tie-breaks ("a correct reply
# always overrides a bad transcript") and still lets a good pass launder a bad
# one. Here each pass is labeled against the command alone, by a labeler that
# never sees the other pass, and decide_kind() resolves the pair in code.

ASR_LOSS_SYSTEM = """You are labeling noisy-audio data for a smart voice \
assistant.

You get the user's real spoken COMMAND, and HEARD -- what a speech recognizer \
caught after loud background chatter. Compare them and list the key pieces of \
the command that did not survive: entities, names, places, times, dates, \
quantities, titles, and the requested action. Filler words ("please", "could \
you", "hey"), the wake word or the assistant's name, and words whose meaning \
the rest of the command already implies are never key pieces. Neither are \
spelling, spacing, or other minor wording differences (e.g. "mockingbird" \
heard as "mocking bird"), or a question rephrased in different grammar that \
still asks for the same thing (e.g. "what does X mean" heard as "what is X", \
or "do you think" heard as "you think") -- these are never lost pieces, even \
when a real key piece is lost alongside them.

An auxiliary or light verb that only frames the request is not a key piece on \
its own -- "does", "do", "is", "can you", "give me", "tell me", "get me". The \
key piece is what is being asked FOR, not the words wrapping the asking. \
Dropping "give" from "give me a current traffic report" loses nothing: it \
still reads as a request for the traffic report. Words that change WHAT is \
asked are a different matter and do count -- "how many", "how long", "where", \
"when" carry the actual question.

Judge HEARD and nothing else. Do not reason about what a reply to this audio \
might have recovered; another labeler handles that separately.

Return ONLY JSON: {"lost": [...], "misheard_as": "...", \
"unintelligible": true|false}

Quote "lost" using the words of the real COMMAND. "misheard_as" is the wrong \
word HEARD in place of a lost piece, or "" if the piece was simply dropped.

"lost" does not need one entry per missing word -- bundle several words into \
ONE entry when they are a single point of confusion: a whole phrase misheard \
as one SPECIFIC, similar-sounding phrase is one entry for the true phrase, \
with the misheard phrase in "misheard_as". Only bundle that way when HEARD is \
a genuine close call -- most of the words or their sounds carry over, so the \
misheard phrase is a plausible thing to guess and offer back. Keep entries \
separate when the losses are unrelated to each other (a mishearing in one \
spot, a different piece dropped elsewhere).

Set "unintelligible" to true when HEARD is destroyed past telling pieces \
apart -- it reads as a different, unrelated sentence, shares barely any words \
or sounds, or says nothing at all -- so there is no plausible guess to offer. \
Give one "lost" entry naming everything unclear and set the flag: the flag, \
not the length of the list, is what marks this case. Otherwise it is false.

Examples:

COMMAND: find some classical music by beethoven and play it
HEARD:   find some classical music by beethoven and play it
{"lost": [], "misheard_as": "", "unintelligible": false}

COMMAND: hey olly are there any alarms set
HEARD:   hey ollie are there any alarms
{"lost": [], "misheard_as": "", "unintelligible": false}
   -- the wake word and "set" are not key pieces

COMMAND: hey olly play playlist tactics from music
HEARD:   a r i play playlist tactics for music
{"lost": [], "misheard_as": "", "unintelligible": false}
   -- the wake word is mangled beyond recognition and it STILL is not a lost \
piece. Never list it, however garbled or absent: the assistant is already \
listening, so nothing about the task depends on hearing its own name

COMMAND: do you think it's going to rain tomorrow
HEARD:   you think it is going to rain tomorrow
{"lost": [], "misheard_as": "", "unintelligible": false}
   -- "do" is just the auxiliary opening the question; dropping it doesn't \
change what's being asked

COMMAND: how many oceans are there in the world
HEARD:   how many children are there in the world
{"lost": ["oceans"], "misheard_as": "children", "unintelligible": false}

COMMAND: give me a current traffic report
HEARD:   me a current traffic report come in and
{"lost": [], "misheard_as": "", "unintelligible": false}
   -- "give" only frames the request; what is being asked for survived intact

COMMAND: does artificial intelligence have consciousness
HEARD:   thus artificial intelligence have consciousness
{"lost": [], "misheard_as": "", "unintelligible": false}
   -- "does" only opens the question, so a wrong word in its place changes \
nothing about what is being asked

COMMAND: tell me why relationships are so hard
HEARD:   why relationship is so hard
{"lost": [], "misheard_as": "", "unintelligible": false}
   -- "tell me" only frames the request, and singular/plural is a minor \
wording difference

COMMAND: i have a meeting by two pm today please remind me
HEARD:   i have a meeting at two p m today
{"lost": ["remind me"], "misheard_as": "", "unintelligible": false}

COMMAND: play mocking bird by eminem
HEARD:   play mockingbird by edna meyer
{"lost": ["eminem"], "misheard_as": "edna meyer", "unintelligible": false}
   -- "mocking bird" vs "mockingbird" is a spacing difference, not a lost \
piece; only the artist name was actually misheard

COMMAND: how do you make steel
HEARD:   or do you make a sale
{"lost": ["how do you make steel"], "misheard_as": "or do you make a sale", \
"unintelligible": false}
   -- multiple words changed together as one coherent near-miss of the whole \
phrase, so it's one bundled entry, and it's still a close call worth guessing at

COMMAND: skip to next episode
HEARD:   get to make copies
{"lost": ["skip to next episode"], "misheard_as": "", "unintelligible": true}
   -- one phrase versus one phrase again, but it reads as a different, \
unrelated sentence, so there is nothing specific enough to guess at

COMMAND: turn on the radio on this channel
HEARD:   anywhere on the radio
{"lost": ["turn on", "this channel"], "misheard_as": "", "unintelligible": false}

COMMAND: please tell me a joke that i'll think is funny
HEARD:   he said me a job that i think is like
{"lost": ["tell me a joke", "i'll think is funny"], "misheard_as": "", \
"unintelligible": false}

COMMAND: please turn on the radio
HEARD:   yes
{"lost": ["turn on the radio"], "misheard_as": "", "unintelligible": true}
   -- nothing distinguishable survived"""


ASR_LOSS_USER = "COMMAND: {sentence}\nHEARD:   {transcript}"


RESP_LOSS_SYSTEM = """You are labeling noisy-audio data for a smart voice \
assistant.

You get the user's real spoken COMMAND, and REPLY -- what the assistant said \
after hearing that same command over loud background chatter. The assistant \
never saw the command in text; REPLY is your only evidence of what reached it.

You are reading REPLY as EVIDENCE, not grading it. A rambling, unhelpful, \
evasive or plain wrong reply can still prove the assistant heard the command \
perfectly, and that is all you are here to establish. Never downgrade a reply \
for being a bad reply; ask only which pieces of the command its wording shows \
got through.

Judge REPLY and nothing else. A separate labeler reads the speech \
recognizer's transcript; you must not speculate about it.

First set "form", the shape of the reply:
- "repair": it asks a targeted question about ONE specific piece the command \
did state, signalling "I may not have caught this right" while treating the \
rest as heard. The rest of the command must actually be treated as heard for \
this to apply.
- "repeat": it asks for the whole command again, or says it could not catch \
it, committing to no specific content at all. This INCLUDES the very common \
"i'm not sure what you mean by <garbled phrase>, could you clarify?" -- \
quoting mangled audio back and asking what it meant is a request to say the \
whole thing again, not a targeted question, however specific the quoted \
phrase looks. It is only "repair" if the question names a piece the command \
really did contain and leaves the rest standing.
- "bad": REPLY is empty, or is pure noise with no interpretable content.
- "answer": EVERYTHING ELSE, whatever its quality. Carrying out the task, \
answering directly, saying it is looking it up, declining for a capability \
reason, handing the job back to the user ("you could open your calendar \
app"), wandering off around the topic, or confidently asserting something \
wrong -- all of these are "answer". Usefulness, correctness and good \
behaviour are somebody else's problem.

Then list in "lost" the key pieces of the command the reply does NOT show it \
heard. Key pieces are entities, names, places, times, dates, quantities, \
titles, and the requested action. Filler words, the wake word, the \
assistant's name, and words the rest of the command already implies are never \
key pieces.
- A piece counts as heard when the reply uses it correctly, judged by meaning \
rather than wording. A natural paraphrase counts, and so does a decline or a \
hand-off that refers to the piece correctly.
- A reply that wanders off the point still proves it heard whatever pieces it \
did name along the way. Credit those; list only the rest.
- A reply that asserts a WRONG value for a piece has lost that piece -- put \
the wrong value in "misheard_as".
- A reply that addresses only part of a multi-part command has lost the part \
it dropped.
- A reply that names nothing from the command at all -- a boilerplate refusal \
like "i'm sorry, i can't help with that request", or an answer to some \
entirely different question -- has lost EVERY key piece. List them all; do \
not shorten the list because the reply was short.
For "repeat" and "bad", "lost" is always an empty list: the table downstream \
already treats those as carrying no evidence either way.

"misheard_as" is the wrong word the reply used in place of a lost piece, or \
"". "asked_about" is filled in only when "form" is "repair": name the piece \
the question targets, quoted in the words of the real COMMAND, or "" if the \
question targets nothing the command actually said.

Return ONLY JSON: {"form": "...", "lost": [...], "misheard_as": "...", \
"asked_about": "..."}

Examples:

COMMAND: set an alarm for seven am tomorrow
REPLY:   i'm setting your alarm for 7 am tomorrow.
{"form": "answer", "lost": [], "misheard_as": "", "asked_about": ""}

COMMAND: what was the score of the mariners game last night
REPLY:   i can't check last night's game score for you.
{"form": "answer", "lost": [], "misheard_as": "", "asked_about": ""}
   -- a capability decline, but it names the game score and last night, so \
those pieces got through

COMMAND: play my most recent playlist
REPLY:   i'm not sure how to play your playlist. you might need to open the \
music app on your device and find the playlist there.
{"form": "answer", "lost": [], "misheard_as": "", "asked_about": ""}
   -- it hands the job back to the user, which is a poor reply and \
irrelevant here: it names playing the playlist, so every piece got through

COMMAND: does artificial intelligence have consciousness
REPLY:   well, you know, artificial intelligence can be really good at making \
us feel more in control. for example, it can help us manage our schedules.
{"form": "answer", "lost": ["have consciousness"], "misheard_as": "", \
"asked_about": ""}
   -- it wanders off the question, but it plainly heard "artificial \
intelligence"; credit that piece and list only the one it never engages with

COMMAND: add milk to my shopping list and remind me at six
REPLY:   i'm adding milk to your shopping list.
{"form": "answer", "lost": ["remind me at six"], "misheard_as": "", \
"asked_about": ""}
   -- only half the command is addressed, so the other half was not heard

COMMAND: play mocking bird by eminem
REPLY:   playing mockingbird by edna meyer now.
{"form": "answer", "lost": ["eminem"], "misheard_as": "edna meyer", \
"asked_about": ""}
   -- asserting a wrong artist is still an attempt at the task; it heard the \
song, so only the artist is lost

COMMAND: how many oceans are there in the world
REPLY:   i'm sorry, i can't help with that request.
{"form": "answer", "lost": ["how many", "oceans", "in the world"], \
"misheard_as": "", "asked_about": ""}
   -- boilerplate that would decline any request equally, so it evidences \
nothing: every key piece goes in the list

COMMAND: remind me about the dentist at four thirty
REPLY:   i've got the dentist reminder -- what time did you say?
{"form": "repair", "lost": ["four thirty"], "misheard_as": "", \
"asked_about": "four thirty"}

COMMAND: turn on the radio on this channel
REPLY:   it's really loud in here -- what was that?
{"form": "repeat", "lost": [], "misheard_as": "", "asked_about": ""}

COMMAND: event reminder mona tuesday
REPLY:   i'm not sure what you mean by "event to mind up money." could you \
please clarify? are you asking about a way to save money for an event?
{"form": "repeat", "lost": [], "misheard_as": "", "asked_about": ""}
   -- it quotes mangled audio back and asks what the whole thing meant. \
Nothing from the real command is treated as heard, so this is a repeat \
request, not a targeted question

COMMAND: brighten the lights a little bit
REPLY:   i'm not sure what you mean by "write in the lights a little bit." \
could you be referring to a specific task?
{"form": "repair", "lost": ["brighten"], "misheard_as": "write in", \
"asked_about": "brighten"}
   -- contrast with the previous example: here "the lights a little bit" is \
carried over correctly, so only one piece is in doubt"""


RESP_LOSS_USER = "COMMAND: {sentence}\nREPLY:   {response}"

RESP_FORMS = ("answer", "repair", "repeat", "bad")

# SLURP's assistant is "olly" (~150 sentences open with it). Both labeler
# prompts say the wake word is never a key piece, and both still occasionally
# list it -- so a lost piece made only of these is dropped before the table
# sees it. Otherwise a garbled wake word becomes a repair anchor and the
# target asks the user to re-confirm their assistant's name.
WAKE_WORDS = {"hey", "ok", "okay", "olly", "ollie", "alexa", "siri", "google",
              "assistant", "computer"}


def drop_wake_only(pieces):
    return [p for p in pieces if set(_normalize_text(p).split()) - WAKE_WORDS]

# a repair anchor covering this fraction of the command's content words or
# more is not targeted at one piece -- treat the probe as "repeat" instead
ANCHOR_BREADTH_MAX = 0.6

# dropped before intersecting two free-text quotes of the same lost piece
PIECE_STOPWORDS = {
    "a", "an", "and", "any", "are", "at", "be", "by", "do", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "please", "s", "that",
    "the", "this", "to", "what", "you", "your",
}


def label_asr(sentence, transcript):
    """(command, ASR transcript) -> {lost, misheard_as, unintelligible} | None."""
    if not transcript:
        return None
    obj = gpt_json(
        ASR_LOSS_SYSTEM,
        ASR_LOSS_USER.format(
            sentence=_normalize_text(sentence), transcript=_normalize_text(transcript)
        ),
        temperature=CLASSIFY_TEMPERATURE,
        max_tokens=CLASSIFY_MAX_TOKENS,
    )
    if obj is None:
        return None
    return {
        "lost": drop_wake_only(
            [str(s).strip() for s in obj.get("lost", []) if str(s).strip()]
        ),
        "misheard_as": str(obj.get("misheard_as", "")).strip(),
        "unintelligible": bool(obj.get("unintelligible", False)),
    }


def label_resp(sentence, response):
    """(command, task reply) -> {form, lost, misheard_as, asked_about} | None."""
    if not response:
        return None
    obj = gpt_json(
        RESP_LOSS_SYSTEM,
        RESP_LOSS_USER.format(
            sentence=_normalize_text(sentence), response=_normalize_text(response)
        ),
        temperature=CLASSIFY_TEMPERATURE,
        max_tokens=CLASSIFY_MAX_TOKENS,
    )
    if obj is None:
        return None
    form = str(obj.get("form", "")).strip().lower()
    if form not in RESP_FORMS:
        return None
    return {
        "form": form,
        "lost": drop_wake_only(
            [str(s).strip() for s in obj.get("lost", []) if str(s).strip()]
        ),
        "misheard_as": str(obj.get("misheard_as", "")).strip(),
        "asked_about": str(obj.get("asked_about", "")).strip(),
    }


def decide_kind(sentence, asr, resp):
    """Independent per-pass labels -> {kind, anchor, misheard_as, buckets}.

    The decision table. `resp=None` falls back to the ASR pass alone, which is
    both what happens when the reply labeler errors out and the entry point
    for an ASR-only classifier variant -- pass resp=None and skip label_resp.

    Returns None when the table lands on "repair" with no piece to anchor the
    question on; such a row is unusable, so the caller drops the probe.
    """

    def overlap(a, b):
        # `lost` entries are free-text quotes, so "the 7 am alarm" and "7 am"
        # have to intersect. One shared content token is enough -- this only
        # ever adjudicates a1's single entry against the reply's list.
        ta = set(_normalize_text(a).split()) - PIECE_STOPWORDS
        tb = set(_normalize_text(b).split()) - PIECE_STOPWORDS
        return bool(ta & tb)

    lost = asr["lost"]
    if asr["unintelligible"]:
        # nothing nameable survived, however few entries that took to say
        a = "aN"
    else:
        a = "a0" if not lost else "a1" if len(lost) == 1 else "aN"

    if resp is None:
        kind = {"a0": "answer", "a1": "repair", "aN": "repeat"}[a]
        r = ""
        anchor = lost[0] if kind == "repair" else ""
        misheard = asr["misheard_as"] if kind == "repair" else ""
    else:
        form, r_lost = resp["form"], resp["lost"]
        if form == "repair":
            r = "rREPAIR"
        elif form == "repeat":
            r = "rREPEAT"
        elif form == "bad":
            r = "rBAD"
        else:
            r = "r0" if not r_lost else "r1" if len(r_lost) == 1 else "rN"

        # An empty or uninterpretable reply demonstrates exactly as much as a
        # repeat request does: nothing. Routing it that way makes the row fall
        # back to the ASR pass alone -- identical to what resp=None does
        # above. Kept as its own bucket rather than rewriting `form`, so a
        # built dataset still shows how often the reply pass abstained.
        route = "rREPEAT" if r == "rBAD" else r

        if a == "a0" or route == "r0":
            # either pass coming through clean clears the audio, whatever the
            # other one did
            kind, anchor, misheard = "answer", "", ""
        elif route == "rREPAIR":
            kind, anchor, misheard = "repair", resp["asked_about"], resp["misheard_as"]
        elif route == "r1":
            kind, anchor, misheard = "repair", r_lost[0], resp["misheard_as"]
        elif route == "rREPEAT":
            # the reply says nothing, so only the ASR pass is speaking
            if a == "a1":
                kind, anchor, misheard = "repair", lost[0], asr["misheard_as"]
            else:
                kind, anchor, misheard = "repeat", "", ""
        elif a == "a1" and any(overlap(p, lost[0]) for p in r_lost):
            # rN, and both passes agree on one piece
            kind = "repair"
            anchor = lost[0]
            misheard = asr["misheard_as"] or resp["misheard_as"]
        elif a == "a1":
            # rN, disjoint: the passes disagree about what was lost, so treat
            # neither loss as real
            kind, anchor, misheard = "answer", "", ""
        else:
            kind, anchor, misheard = "repeat", "", ""

    # A "targeted" question covering most of the command is a repeat request
    # wearing a repair's clothes -- it asks the user to say nearly everything
    # again. Three separate paths can produce one: the reply labeler reading
    # "i'm not sure what you mean by <garbled>" as a repair, either labeler
    # bundling several unrelated losses into one entry, or a command whose
    # only key piece is the lost one. Checking the final anchor catches all
    # three at once.
    if kind == "repair":
        cmd = set(_normalize_text(sentence).split()) - PIECE_STOPWORDS
        anc = set(_normalize_text(anchor).split()) - PIECE_STOPWORDS
        if cmd and len(anc & cmd) / len(cmd) >= ANCHOR_BREADTH_MAX:
            kind, anchor, misheard = "repeat", "", ""

    if kind == "repair" and not anchor:
        return None

    # "missing" rather than "lost", to match what the other two labelers return
    if kind == "repair":
        pieces = [anchor]
    elif kind == "repeat":
        pieces = list(dict.fromkeys(lost + (resp["lost"] if resp else [])))
    else:
        pieces = []

    return {
        "kind": kind,
        "anchor": anchor,
        "misheard_as": misheard,
        "missing": pieces,
        "asr_bucket": a,
        "resp_bucket": r,
        "reason": (
            f"{a}x{r or '-'} | asr lost: {'; '.join(lost) or 'none'}"
            f"{' (unintelligible)' if asr['unintelligible'] else ''}"
            + (
                f" | reply {resp['form']}, lost: {'; '.join(resp['lost']) or 'none'}"
                if resp
                else ""
            )
        ),
    }


def label_tree(sentence, transcript, response):
    """Label both passes independently, then resolve them with the table.

    A reply labeler that errors out leaves resp=None, which decide_kind reads
    as the ASR-only rule rather than failing the probe.
    """
    asr = label_asr(sentence, transcript)
    if asr is None:
        return None
    return decide_kind(sentence, asr, label_resp(sentence, response))


# ---
# --tree-label: one target per probe
# ---

# The table anchors the repair question on the individual probe's losses, so
# targets are written per probe rather than once per utterance triplet.
# Answers reuse ANSWER_TARGET_SYSTEM above: the base model's own reply to a
# fully-intelligible probe is a correct answer but a bad target, since it
# habitually offloads the task back onto the user ("you could open your music
# app and search for it"), which is the opposite of what we want trained in.

REPAIR_TARGET_SYSTEM = """You are writing the reply a smart voice assistant \
should give when background chatter cost it exactly one piece of a spoken \
command.

You get the user's real COMMAND, the HEARD text the device caught, and the \
LOST-PIECE that did not get through. If MISHEARD-AS is given, the device \
heard that similar-sounding wrong word in place of the lost piece.

Write ONE short natural question (under 20 words) recovering ONLY that piece. \
Test: if the user replied with just the missing words, the command would be \
complete.
- NEVER ask about parts that were heard correctly -- asking again would sound \
like the assistant wasn't listening.
- Ground the question in the parts that did survive, so it is clear the \
assistant followed everything except this one piece.
- Do not reveal the missing words. ONE exception: when MISHEARD-AS is given \
you may ask a confirmation question offering the true word AND the misheard \
word as alternatives ("did you say saved or shared?") -- never the true word \
alone.
- Sound like natural speech, not a form. Vary the structure freely: \
"Which...?", "How long before...?", "Who should...?", "What time...?", \
"Where...?", or a statement plus a question ("I lost one part -- where to?"). \
Do NOT default to starting with "Sorry".

Return ONLY JSON: {"repair": "..."}"""


REPAIR_TARGET_USER = (
    'COMMAND:\n"{sentence}"\n\n'
    'HEARD:\n"{transcript}"\n'
    'LOST-PIECE: "{anchor}"{swap_note}'
)


REPEAT_TARGET_SYSTEM = """You are writing the reply a smart voice assistant \
should give when background chatter cost it too much of a spoken command for \
a targeted question to be possible.

Write ONE short natural request (under 15 words) asking the user to say the \
whole thing again.
- Do NOT reference, guess, or hint at ANY content from the real command or \
from the garbled text -- the assistant cannot trust any of it.
- Mentioning the noise is fine and helps explain why.
- Sound like natural speech and vary the phrasing ("It's really loud here -- \
what was that?", "I couldn't catch that over the noise, could you say it \
again?"). Do NOT default to starting with "Sorry".

Return ONLY JSON: {"repeat": "..."}"""


REPEAT_TARGET_USER = 'GARBLED:\n"{transcript}"'


def write_target(sentence, kind, probe):
    """One SFT target for one already-labeled tree-track probe."""
    if kind == "answer":
        system = ANSWER_TARGET_SYSTEM
        user = ANSWER_TARGET_USER.format(sentence=sentence)
    elif kind == "repair":
        system = REPAIR_TARGET_SYSTEM
        user = REPAIR_TARGET_USER.format(
            sentence=sentence,
            transcript=probe["transcript"],
            anchor=probe["anchor"],
            swap_note=(
                f'\nMISHEARD-AS: "{probe["swapped"][0]}"' if probe["swapped"] else ""
            ),
        )
    else:
        system = REPEAT_TARGET_SYSTEM
        user = REPEAT_TARGET_USER.format(transcript=probe["transcript"])

    for attempt in range(TARGET_RETRIES):
        obj = gpt_json(system, user, temperature=0.7, max_tokens=TARGET_MAX_TOKENS)
        if obj is None:
            time.sleep(2**attempt)
            continue
        target = str(obj.get(kind, "")).strip()
        if target:
            return target
    return ""


# ---
# SNR probing -> classify its kind
# ---


def probe_by_kinds(clean, pool, sentence, kinds_need, batch_size, rng):
    def make_probe_batch(kinds_need):
        length = len(clean)
        clean_power = float(np.mean(clean**2))
        bands = SLOT_SNR if TRACK == "two-pass" else SLOT_SNR_DISJOINT
        audios, snrs = [], []
        while len(audios) < batch_size:
            weights = [SLOT_WEIGHTS[k] for k in kinds_need]
            slot = rng.choices(kinds_need, weights=weights, k=1)[0]

            if (
                TRACK != "two-pass"
                and slot == "answer"
                and rng.random() < CLEAN_ANSWER_PROB
            ):
                # keep noise-free audio in the answer band; snr_db=None reads
                # as "clean" everywhere downstream
                audios.append(clean)
                snrs.append(None)
                continue

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

            audios.append(noisy)
            snrs.append(snr)
        return audios, snrs

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
        audios, snrs = make_probe_batch(missing_slots)

        sysp = QWEN25_SYSTEM_PROMPT if base_family == "qwen2.5" else None
        if TRACK == "heard-reply":
            with GPU_LOCK:
                convs = [_conv(a, sysp, task_prompt(True)) for a in audios]
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
                convs = [_conv(a, asr_sysp, ASR_PROMPT) for a in audios]
                transcripts = base_generate_batch(convs, ASR_MAX_NEW_TOKENS)

                # get batch omni assistant respond
                convs = [_conv(a, sysp, TASK_PROMPT) for a in audios]
                responses = base_generate_batch(convs, RESP_MAX_NEW_TOKENS)

            # same two witnesses either way; only how they're labeled differs
            label_one = classify if TRACK == "two-pass" else (
                lambda t, r: label_tree(sentence, t, r)
            )
            with ThreadPoolExecutor(max_workers=CLASSIFY_WORKERS) as ex:
                labels = list(
                    ex.map(
                        lambda it: label_one(*it),
                        list(zip(transcripts, responses)),
                    )
                )

        for snr, noisy, transcript, response, label in zip(
            snrs, audios, transcripts, responses, labels
        ):
            if label is None:
                continue
            kind = label["kind"]
            if kind in results and results[kind] is None:
                results[kind] = {
                    "snr_db": snr,
                    "audio": noisy,
                    "transcript": transcript,
                    "response": response,
                    "lost": label["missing"],
                    "swapped": [label["misheard_as"]] if label["misheard_as"] else [],
                    "reason": label["reason"],
                    # --heard-reply writes the target in the same call; the
                    # other two tracks fill this in later, once the probe is
                    # actually kept
                    "target": label.get("reply", ""),
                    # tree track only: the table cell and the piece it picked
                    "asr_bucket": label.get("asr_bucket", ""),
                    "resp_bucket": label.get("resp_bucket", ""),
                    "anchor": label.get("anchor", ""),
                }

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
    reply = target
    if TRACK == "heard-reply":
        target = f"Heard: {probe['transcript']}\nReply: {reply}"
    return {
        "id": next(ROW_ID),
        "kind": kind,
        "target": target,
        "target_reply": reply,
        "audio": path,
        "snr_db": probe["snr_db"],
        "asr_transcript": probe["transcript"],
        "omni_response": probe["response"],
        "lost": probe["lost"],
        "swapped": probe["swapped"],
        "classifier_reason": probe["reason"],
        # tree track: which table cell produced this row, and the piece the
        # repair question was anchored on. "" on the other two tracks.
        "asr_bucket": probe["asr_bucket"],
        "resp_bucket": probe["resp_bucket"],
        "anchor": probe["anchor"],
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
        if any(v is None for v in triplet.values()):
            return {"skip": "probe"}

        if TRACK == "heard-reply":
            # already written, one call per probe audio alongside its label
            return {"triplet": triplet, "targets": {k: triplet[k]["target"] for k in KINDS}}

        if TRACK == "tree":
            # one call per probe: the table anchored the repair question on
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
            sf.write(path, probe["audio"], AUDIO_SAMPLING_RATE)
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

        if TRACK == "tree":
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
        sf.write(path, built["probe"]["audio"], AUDIO_SAMPLING_RATE)
        rows.append(
            make_row("answer", built["target"], path, built["probe"], slurp_id, sentence)
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
        "against the command and resolved by decide_kind()'s table, which "
        "also names the piece a repair question anchors on.",
    )
    args = ap.parse_args()

    TRACK = (
        "heard-reply" if args.heard_reply else "tree" if args.tree_label else "two-pass"
    )
    log(f"track: {TRACK}")

    # ---
    # point AUDIO_DIR at a fresh per-dataset subdir of AUDIO_ROOT.
    # ---
    AUDIO_DIR = os.path.join(AUDIO_ROOT, args.ds_id.split("/")[-1])
    shutil.rmtree(AUDIO_DIR, ignore_errors=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    log(f"audio dir: {AUDIO_DIR}")

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
