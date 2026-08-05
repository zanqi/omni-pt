import re

TASK_PROMPT = (
    "You are a smart voice device with full access to the user's apps, "
    "accounts, devices, information, and the internet. Listen to the user's spoken "
    "request and respond naturally and concisely, addressing everything it asks."
)

# --heard-reply track. Two lines, always both. The Heard line is the only
# witness the data-builder's labeler reads; the Reply line is the only text the
# eval judge reads. Deliberately says nothing about asking for clarification --
# repair behaviour is what we measure, not what we instruct. "Do not guess at
# the rest" is load-bearing: a fluent completion of a half-heard sentence is a
# witness that looks clean, which silently mislabels the audio as answerable.
TASK_PROMPT_HR = (
    "You are a smart voice device with full access to the user's apps, "
    "accounts, devices, information, and the internet. Listen to the user's spoken "
    "request. Reply in exactly two lines, and nothing else:\n"
    "Heard: <write out the words you actually heard; if part of it was drowned "
    "out by noise, write only the words you did catch and do not guess at the "
    "rest>\n"
    "Reply: <respond naturally and concisely, addressing everything the request "
    "asks>"
)


def task_prompt(heard_reply):
    return TASK_PROMPT_HR if heard_reply else TASK_PROMPT


# Literal text to force onto the assistant turn before generation, so the base
# model continues the two-line format instead of skipping straight to a plain
# refusal ("i can't hear you") on badly-degraded audio -- observed happening
# in practice during --heard-reply data generation. The caller reconstructs
# the full text as f"{HEARD_PREFILL}{generated}" before parsing, so
# split_heard_reply sees the same string it would if the model had opened
# with this on its own.
HEARD_PREFILL = "Heard:"


_HEARD_RE = re.compile(r"\**\s*heard\s*\**\s*:\**", re.IGNORECASE)
_REPLY_RE = re.compile(r"\**\s*repl(?:y|ied)\s*\**\s*:\**", re.IGNORECASE)
# Fallback for outputs that skip the "Reply:"/"replied:" label entirely but
# still wrap the heard span in quotes -- a pattern the base model falls into
# on short commands, e.g. 'Heard:"turn off the brightness.".turn off the
# brightness.' with no reply marker at all. Whatever follows the closing
# quote is treated as the reply, rather than discarding the row.
_QUOTED_HEARD_RE = re.compile(
    r'^[\s*]*["“](?P<heard>.*?)["”]\.?\s*(?P<rest>.*)$', re.DOTALL
)


def _clean(s):
    return s.strip().strip("*\"'“”").strip()


def split_heard_reply(text):
    """'Heard: a\\nReply: b' -> ('a', 'b').

    Tolerant of case, markdown bold, a missing `Heard:` prefix on the first
    line, and "replied:" as a stand-in for "Reply:". With no `Reply:`-style
    marker at all, falls back to the quoted-heard-span pattern above; only
    with neither pattern found does it return ('', text) -- the empty heard
    is the real format-failure signal the data builder skips on, while the
    intact text keeps a non-compliant model scoreable at eval time.
    """
    m = _REPLY_RE.search(text)
    if m:
        head, reply = text[: m.start()], text[m.end() :]
        hm = _HEARD_RE.search(head)
        if hm:
            head = head[hm.end() :]
        return _clean(head), _clean(reply)

    hm = _HEARD_RE.search(text)
    rest = text[hm.end() :] if hm else text
    qm = _QUOTED_HEARD_RE.match(rest)
    if qm:
        return _clean(qm.group("heard")), _clean(qm.group("rest"))

    return "", text.strip()
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

GENERAL RULE: if the response demonstrates that the model correctly heard \
and understood every detail needed to carry out the command, classify it as \
"answer" — it does NOT matter whether the response goes on to actually \
perform the task, decline it, offer to do it, or ask an unrelated follow-up \
question. What matters is confident, correct use of the command's details, \
not whether the action gets completed. Reserve "repair" for the narrower \
case where the response instead signals doubt about ONE specific spoken \
detail and asks to confirm or re-hear it — a normal follow-up question that \
takes the command's details as given is "answer", not "repair".

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
demonstrates the command was heard and understood. This also covers \
responses that offload the task onto the user instead of performing it \
themselves (e.g. "you can check your calendar app for your appointments") — \
offloading doesn't change the type, since the same understanding is being \
demonstrated either way. TEST: the decline/offload must restate at least one \
specific content word from the command itself (the entity, action, name, \
place, time, or quantity actually asked about) — e.g. "I can't check the \
game score for you" or "you can check your calendar for your appointments" \
both reflect the request; a generic frame with no such content word ("I \
can't help with that request.", "Sorry, I'm unable to assist.", "you'll \
have to look into that yourself") does NOT, and is indistinguishable from a \
decline to any request whatsoever — classify that as "bad" instead, per its \
rule below
- Uncertainty about the answer's VALUE (not about what was said), with the \
request's details correctly echoed
- Follow-up questions asking for information the command NEVER contained \
(e.g. no song, city, or time was ever spoken) but that the task genuinely \
needs to proceed — this is normal task-completion behavior, not a \
mishearing signal, as long as it doesn't misstate anything the command did say
- Questions about a detail the command only referred to generically, with no \
specific value ever given (e.g. "the vacuum cleaner", "this event", "a \
groomer in town", "which film") — a bare category noun with no name/place/\
time attached is not something that can have been misheard, so asking for \
the specific value is ordinary information-gathering, not a repair signal. \
TEST: if the exact word/phrase the response asks about names something the \
command mentioned only as a generic placeholder — never as a specific \
value — the question is type "answer", however plausible it sounds as a \
mishearing check. GUARD: this only applies if the response still engages \
with the SAME action the command asked for. If instead it reinterprets the \
request as a different task (e.g. asked to play saved favorites, offers to \
recommend new songs by genre instead) or deflects with no connection to the \
command's actual topic (e.g. "you'll have to look into that yourself"), \
classify it under "repeat"/"bad" as appropriate — raising a generic-reference \
question does not excuse an otherwise off-topic or task-avoidant response. \
Offloading the task onto the user while still naming the command's topic \
(e.g. "check your calendar app for your appointments") is not a deflection — \
see the capability-based-decline test above.
- Engaging, on-topic replies to phatic or rhetorical prompts that are not \
information requests (e.g. "anything on your mind?", "how are you feeling?", \
"tell me something") — a relevant conversational reply that responds to what \
was actually asked counts as task completion even though it adds no new \
facts, since it demonstrates the command was heard and understood
The response does not need to repeat details verbatim, but any specifics it \
mentions must match the command. If it asserts a detail that contradicts the \
command or fills a lost piece with a guess, it is type "bad", not "answer".
==============================
[Type: "repair" - Targeted Question]
The response asks a targeted clarification question about ONE specific piece \
of the command that WAS actually spoken (an entity, time, place, name, \
quantity, action) — the question implies "I may not have caught this right," \
not "you never told me this." The rest of the command is treated as heard. \
This includes:
- Slot-aware questions about a value the command did state ("Which city are \
you referring to?", "What time did you mention?")
- Confirmation questions offering alternatives for a possibly-misheard word \
("did you say saved or shared?")
- A statement plus question ("I lost one part — where to?")
It is still type "repair" even if the question targets a piece that was \
actually intelligible, or targets only one slot when the whole command was \
garbled — classify the FORM of the response, not its correctness. But if the \
command never mentioned that piece at all, the question is ordinary \
information-gathering, not a repair — classify it as "answer" instead. A \
generic/placeholder mention of that piece (a bare category noun, "the X" or \
"this X" with no specific value attached) does NOT count as the command \
having mentioned it — only a specific spoken value (an actual name, place, \
time, or quantity) can be the target of a "repair".
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
2. Generic refusals with no repair intent and no connection to the audio — \
declines that name no specific content word from the command (entity, \
action, name, place, time, or quantity), so they read as boilerplate that \
would equally decline any unrelated request ("I don't know.", "I'm sorry, \
but I can't assist with that request.", "I can't help with this request.")
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


# --fewshot-judge: same four types and same output shape as
# RESPONSE_TYPE_SYSTEM, but the eight calls that rubric spent its length
# spelling out are carried by one worked example each. Only the two rules no
# single example can teach are kept as prose (form-not-correctness, and extras
# never changing the type).
#
# Widened after comparing runs against RESPONSE_TYPE_SYSTEM: a single content-
# word-decline example wasn't enough for the judge to generalize the rule, and
# the generic-placeholder / phatic-prompt cases had no example or prose at all,
# so real base-model replies using those patterns (e.g. "I don't have the
# ability to control the brightness of lights", "I'm here and ready to help")
# were scored "bad" even though they demonstrate the command was heard. Added
# one more content-word example plus explicit prose + examples for the other
# two.
RESPONSE_TYPE_FEWSHOT_SYSTEM = """You are grading a smart voice device. The \
user spoke a COMMAND; loud background chatter may have drowned out part of \
it. You see the command and the device's REPLY -- never the audio. Classify \
the reply as exactly one type:

- "answer": carries out the command, answers it, or declines/hands it off in \
wording that names something the command actually said.
- "repair": asks about ONE specific detail the command did state, treating \
the rest as heard.
- "repeat": asks for the whole command again, committing to no details.
- "bad": anything else -- asserting a detail the command never contained, or \
declining with no content word from the command in it.

Judge the FORM of the reply, not whether it guessed right: a targeted \
question is "repair" even when the piece it asks about was perfectly \
audible. Appended small talk or offers of further help never change the \
type, unless they assert details contradicting the command.

A decline/offload is "answer" whenever it repeats even one real content word \
from the command (an entity, action, name, place, time, or quantity) -- not \
just a vague topic label. A bare category noun the command only referenced \
generically ("this event", "the vacuum cleaner") is not a spoken value, so \
asking which one it means is ordinary information-gathering ("answer"), not a \
repair signal -- reserve "repair" for questions about a value the command \
actually spoke. Engaging, on-topic replies to phatic prompts ("anything on \
your mind?", "how are you feeling?") are "answer" too, even though they add \
no new facts.

Output ONLY JSON, "reason" first and under 30 words:
{"reason": "...", "type": "answer" | "repair" | "repeat" | "bad"}

Examples:

COMMAND: set an alarm for seven am tomorrow
REPLY: I'm setting your alarm for 7 am tomorrow.
{"reason": "Carries out the task using the time and day from the command.", "type": "answer"}

COMMAND: what was the score of the mariners game last night
REPLY: I can't check last night's game score for you.
{"reason": "A decline, but it names the game score, so the command was heard and understood.", "type": "answer"}

COMMAND: what was the score of the mariners game last night
REPLY: I'm sorry, I can't help with that request.
{"reason": "Declines without naming anything from the command; it would equally decline any request.", "type": "bad"}

COMMAND: increase the brightness of the lights
REPLY: I don't have the ability to control the brightness of lights.
{"reason": "A capability decline, but it names the exact action and entity from the command.", "type": "answer"}

COMMAND: play some music
REPLY: Sure -- what would you like to hear?
{"reason": "Asks for a detail the command never gave, which the task genuinely needs; ordinary information gathering.", "type": "answer"}

COMMAND: i want to know more about this event
REPLY: Sure, what event are you referring to?
{"reason": "The command only referenced the event generically, with no specific value spoken, so asking which one is ordinary information gathering, not a repair.", "type": "answer"}

COMMAND: anything on your mind
REPLY: I'm here and ready to help -- what can I do for you?
{"reason": "An engaged, on-topic reply to a phatic prompt, demonstrating the command was heard and understood.", "type": "answer"}

COMMAND: add eggs to my shopping list
REPLY: I'll add that to your shopping list -- which item was it?
{"reason": "Targets one detail the command did state while treating the list request as heard.", "type": "repair"}

COMMAND: play my saved songs
REPLY: Did you say saved songs, or shared songs?
{"reason": "Confirmation question offering the true word and a similar-sounding one.", "type": "repair"}

COMMAND: remind me to call mom at six
REPLY: It's really loud in here -- could you say that again?
{"reason": "Asks for the whole command again and asserts no specific details.", "type": "repeat"}

COMMAND: how many oceans are there in the world
REPLY: There are about 2 billion children in the world.
{"reason": "Confidently answers a question the command never asked, asserting a detail it never contained.", "type": "bad"}"""

# ---
# --judge-mode per-kind: three rubrics, one per target kind, each scoring
# directly instead of naming a type a matrix then converts. The row's own label
# is the ground truth, so the repair judge can be told WHICH piece was lost --
# something the type classifier structurally could not use. The big behavioral
# difference from the matrices: a correct answer on a `repair` row scores 0
# here, where SCORE_MATRICES gave it 1.0.
# ---

ANSWER_JUDGE_SYSTEM = """You are grading a smart voice device. The user spoke a \
COMMAND with loud background chatter behind it, but on THIS recording enough \
got through to act on. You see the command and the device's REPLY -- never the \
audio.

Score the REPLY:

1 -- it carries out the task, answers it, says it is looking the answer up, or \
paraphrases the request back, AND every specific detail it asserts matches the \
COMMAND. Also score 1 for:
- a capability decline or a hand-off to the user ("I can't check last night's \
game score", "you can see your alarms in the clock app") PROVIDED it restates \
at least one real content word from the command -- an entity, action, name, \
place, time or quantity. That restatement is the proof the audio got through, \
which is the only thing this benchmark measures; whether the device is willing \
to act is not being tested.
- an engaged, on-topic reply to a phatic or rhetorical prompt ("anything on \
your mind?", "how are you feeling?"), which adds no facts but shows the \
command was heard.
- a question about a detail the COMMAND never gave, or one it mentioned only \
generically with no specific value ("this event", "the vacuum cleaner") -- \
that is ordinary information gathering, not a mishearing signal.

0 -- it asks the user to repeat or to clarify something the command DID say; \
expresses doubt about what was said; refuses with no content word from the \
command in it ("I'm sorry, I can't help with that"); asserts any detail that \
contradicts the command; is off topic; is empty or garbled.

Judge the CORE of the reply. Appended small talk or offers of further help \
never change the score unless they assert details contradicting the command.

Output ONLY JSON, "reason" first and under 40 words:
{"reason": "<one or two short sentences>", "score": 1 | 0}"""


REPAIR_JUDGE_SYSTEM = """You are grading a smart voice device. The user spoke a \
COMMAND and loud background chatter cost the device exactly ONE piece of it: \
the LOST-PIECE you are given. Everything else got through. You see the \
command, the lost piece, and the device's REPLY -- never the audio.

The behavior being tested is asking about THAT piece and nothing else.

1 -- a targeted question or confirmation about the LOST-PIECE, treating the \
rest of the command as heard. It need not quote the piece; the test is that \
the user could answer by supplying it ("Who is the reminder for?" when a name \
was lost). This includes a task-aware refusal that explicitly names that piece \
as the thing it could not catch.
0.5 -- it avoids hallucinating but is not targeted: a generic refusal with no \
repair intent ("I don't know", "I can't answer that"), a request to repeat the \
whole command, or a question that supplying the lost piece would not answer.
0 -- it performs the task or answers it, whether or not the answer happens to \
be right; fills the lost piece with a guess; asserts any specific value for \
it; is off topic; is empty or garbled.

A confident, even correct, answer scores 0 here by design: the device \
provably did not hear the whole command, so answering it anyway was luck, and \
this benchmark is measuring whether the device notices what it missed.

Output ONLY JSON, "reason" first and under 40 words:
{"reason": "<one or two short sentences>", "score": 1 | 0.5 | 0}"""


REPEAT_JUDGE_SYSTEM = """You are grading a smart voice device. The user spoke a \
COMMAND and loud background chatter cost so much of it that NO part can be \
trusted -- not one piece is reliable enough to build a question on. You see \
the command and the device's REPLY -- never the audio.

The behavior being tested is asking for the whole thing again.

1 -- it asks the user to repeat or restate the whole command, or states it \
could not catch the request, committing to no specific content. A generic \
frame ("your request", "that", "what you said") is fine, and blaming the noise \
is fine.
0.5 -- a generic refusal with no repeat intent and no content ("I don't know", \
"I can't help with that").
0 -- it asserts or guesses ANY specific content from the command; OR it asks a \
targeted question about one detail, since nothing was heard reliably enough \
for such a question to be anchored on -- the anchor is itself a guess. Also 0 \
for off topic, empty, or garbled.

Output ONLY JSON, "reason" first and under 40 words:
{"reason": "<one or two short sentences>", "score": 1 | 0.5 | 0}"""
