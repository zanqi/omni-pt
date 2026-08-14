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


# --tree-label track. Same single-reply format as TASK_PROMPT, but the reply re-state key pieces
TASK_PROMPT_TREE = (
    "You are a smart voice device with full access to the user's apps, "
    "accounts, devices, information, and the internet. Listen to the user's spoken "
    "request and respond naturally and concisely, addressing everything it asks. "
    "As you reply, restate every piece of the request you caught -- the action "
    "and each entity, name, place, time and quantity -- so it is clear what you "
    "heard. If background noise drowned out part of the request, do not guess at "
    "the words you missed."
)


def task_prompt(heard_reply, restate=False):
    if heard_reply:
        return TASK_PROMPT_HR
    return TASK_PROMPT_TREE if restate else TASK_PROMPT


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


# used by the complex tree track
RESPONSE_TYPE_SYSTEM = """You are grading a smart voice device. The user spoke \
a COMMAND; loud background chatter may have drowned out part of it. You see \
the command and the device's REPLY -- never the audio. Classify the reply as \
exactly one type:

- "answer": acts on the command, answers it, or declines/hands it off, in \
wording that accounts for EVERY key element the command spoke (the action, \
and each entity, name, place, time and quantity). Asking for a value the \
command never gave is fine and still "answer".
- "repair": asks about ONE specific element the command did state, treating \
the rest as heard.
- "repeat": asks for the whole command again, committing to no details.
- "bad": anything else -- drops a key element the command spoke, asserts one \
it never contained, or declines with nothing from the command in it.

Judge the FORM of the reply, not whether it guessed right: a targeted question \
is "repair" even when the piece it asks about was perfectly audible. Appended \
small talk or offers of further help never change the type, unless they assert \
details contradicting the command.

Output ONLY JSON, "reason" first and under 30 words:
{"reason": "...", "type": "answer" | "repair" | "repeat" | "bad"}

Examples:

COMMAND: set an alarm for seven am tomorrow
REPLY: I'm setting your alarm for 7 am tomorrow.
{"reason": "Carries out the task and accounts for both the time and the day.", "type": "answer"}

COMMAND: set an alarm for seven am tomorrow
REPLY: Sure, I've set your alarm for tomorrow.
{"reason": "Acts confidently but silently drops the spoken time, a key element of the command.", "type": "bad"}

COMMAND: what was the score of the mariners game last night
REPLY: I can't look up last night's Mariners score for you.
{"reason": "A capability decline whose wording still accounts for the team, the score and the night.", "type": "answer"}

COMMAND: what was the score of the mariners game last night
REPLY: I'm sorry, I can't help with that request.
{"reason": "Declines without naming anything from the command; it would equally decline any request.", "type": "bad"}

COMMAND: play some music
REPLY: Sure -- what would you like to hear?
{"reason": "Engages the play-music request and asks for a value the command never gave; nothing spoken is dropped.", "type": "answer"}

COMMAND: i want to know more about this event
REPLY: Sure, what event are you referring to?
{"reason": "The command referenced the event only generically, so asking which one is ordinary information gathering.", "type": "answer"}

COMMAND: anything on your mind
REPLY: I'm here and ready to help -- what can I do for you?
{"reason": "An engaged, on-topic reply to a phatic prompt, demonstrating the command was heard.", "type": "answer"}

COMMAND: add eggs to my shopping list
REPLY: I'll add that to your shopping list -- which item was it?
{"reason": "Targets one element the command did state while treating the list request as heard.", "type": "repair"}

COMMAND: play my saved songs
REPLY: Did you say saved songs, or shared songs?
{"reason": "Confirmation question offering the true word and a similar-sounding one.", "type": "repair"}

COMMAND: remind me to call mom at six
REPLY: It's really loud in here -- could you say that again?
{"reason": "Asks for the whole command again and asserts no specific details.", "type": "repeat"}

COMMAND: how many oceans are there in the world
REPLY: There are about 2 billion children in the world.
{"reason": "Confidently answers a question the command never asked, asserting a detail it never contained.", "type": "bad"}"""


# Second stage of the tree judge
REPAIR_ON_TARGET_SYSTEM = """You are grading a smart voice device. The user \
spoke a COMMAND, loud background chatter drowned out the LOST PIECE, and the \
device replied with a clarification question. Decide only whether that \
question is aimed at the lost piece.

- "repair": answering the question would supply the lost piece. Paraphrase, a \
category word for it, or offering it among candidate words all count.
- "repair_off": the question is aimed at some other part of the command, or \
at nothing in particular.

Output ONLY JSON, "reason" first and under 25 words:
{"reason": "...", "type": "repair" | "repair_off"}

Examples:

COMMAND: play my saved songs
LOST PIECE: saved
REPLY: Did you say saved songs, or shared songs?
{"reason": "Offers the lost word among the candidates; answering it supplies the lost piece.", "type": "repair"}

COMMAND: remind me to call mom at six
LOST PIECE: six
REPLY: What time should I set the reminder for?
{"reason": "Asks for the time, which is exactly the lost piece, in a paraphrase.", "type": "repair"}

COMMAND: remind me to call mom at six
LOST PIECE: six
REPLY: Who did you want me to call?
{"reason": "Asks about the person, which was heard; the lost time is left unaddressed.", "type": "repair_off"}

COMMAND: what is the weather in seattle tomorrow
LOST PIECE: seattle
REPLY: Could you tell me more about what you need?
{"reason": "A vague request aimed at nothing in particular; answering it need not supply the city.", "type": "repair_off"}"""


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


# ---
# --tree-label: two independent per-pass labelers, intersected in code
# ---

RESP_LOSS_SYSTEM = """You are labeling noisy-audio data for a smart voice \
assistant.

You get the user's real spoken COMMAND, and REPLY -- what the assistant said \
after hearing that same command over loud background chatter. The assistant \
never saw the command in text; REPLY is your only evidence of what reached \
it. Read REPLY as EVIDENCE, not as work to grade: a rambling, unhelpful, \
evasive or plain wrong reply can still prove the command was heard perfectly, \
and that is all you are here to establish. Judge REPLY and nothing else; \
another labeler reads the recognizer's transcript separately.

List in "lost" the key pieces of the command -- entities, names, places, \
times, dates, quantities, titles, the requested action -- that REPLY does not \
show it heard. A piece counts as heard when the reply uses it correctly, \
judged by meaning rather than wording; a paraphrase, a decline, a hand-off, or \
an admission of not knowing the ANSWER all still name what was asked. A piece \
the reply asserts with a WRONG value is lost.

Naming a piece correctly settles it as heard, even when the reply goes on to \
ask a follow-up about it: a command that referred to something only \
generically ("that podcast", "this event", "the vacuum cleaner") gave no \
specific value, so asking WHICH one is the assistant gathering information it \
never had, not evidence that the noise took anything.

Not key pieces, ever: filler words, the wake word or assistant's name, and the \
verbs that merely frame a request ("do", "does", "give me", "tell me"). \
Question words are key pieces -- "how many", "how long", "where", "when" carry \
what is actually being asked.

Bundle several words into ONE entry when they are a single point of confusion; \
keep entries separate when the losses are unrelated. When REPLY asks for the \
whole command again, or engages with nothing in it at all, it shows no piece \
was heard: list every key piece.

Return ONLY JSON, "reason" first and under 25 words, quoting "lost" in the \
words of the real COMMAND:
{"reason": "...", "lost": [...]}

Examples:

COMMAND: set an alarm for seven am tomorrow
REPLY:   i'm setting your alarm for 7 am tomorrow.
{"reason": "Names the alarm, the time and the day.", "lost": []}

COMMAND: play my most recent playlist
REPLY:   i'm not sure how to play your playlist. you might need to open the \
music app on your device and find the playlist there.
{"reason": "Handing the job back is a poor reply but names playing the playlist, so every piece got through.", "lost": []}

COMMAND: remind me of how many lists i have
REPLY:   i'm not sure how many lists you have. you could check your phone's \
settings or the app where you keep your lists.
{"reason": "Not knowing the ANSWER is not failing to hear the QUESTION; it names the lists and the count.", "lost": []}

COMMAND: play that podcast please
REPLY:   i'm not sure which "that podcast" you're referring to. can you be \
more specific? maybe tell me the name of the podcast?
{"reason": "Quotes the podcast back, so it got through; the command never said which one, so asking is information gathering.", "lost": []}

COMMAND: does artificial intelligence have consciousness
REPLY:   well, you know, artificial intelligence can be really good at making \
us feel more in control. for example, it can help us manage our schedules.
{"reason": "It wanders off, but plainly heard the subject; only the property asked about goes unengaged.", "lost": ["have consciousness"]}

COMMAND: add milk to my shopping list and remind me at six
REPLY:   i'm adding milk to your shopping list.
{"reason": "Only half the command is addressed.", "lost": ["remind me at six"]}

COMMAND: play mocking bird by eminem
REPLY:   playing mockingbird by edna meyer now.
{"reason": "The artist comes back as a different name; spacing of the title is not a loss.", "lost": ["eminem"]}

COMMAND: how many oceans are there in the world
REPLY:   i'm sorry, i can't help with that request.
{"reason": "Boilerplate that would decline any request equally evidences nothing.", "lost": ["how many", "oceans", "in the world"]}

COMMAND: remind me about the dentist at four thirty
REPLY:   i've got the dentist reminder -- what time did you say?
{"reason": "Confirms the dentist reminder and asks about the time, so the time is the piece in doubt.", "lost": ["four thirty"]}

COMMAND: turn on the radio on this channel
REPLY:   it's really loud in here -- what was that?
{"reason": "Asks for the whole thing again, engaging with nothing, so no piece is shown heard.", "lost": ["turn on", "the radio", "this channel"]}

COMMAND: event reminder mona tuesday
REPLY:   i'm not sure what you mean by "event to mind up money." could you \
please clarify? are you asking about a way to save money for an event?
{"reason": "Quotes mangled audio back and asks what the whole thing meant; nothing survives.", "lost": ["event reminder", "mona", "tuesday"]}

COMMAND: brighten the lights a little bit
REPLY:   i'm not sure what you mean by "write in the lights a little bit." \
could you be referring to a specific task?
{"reason": "Contrast the previous case: the lights a little bit is carried over correctly, so only the verb is in doubt.", "lost": ["brighten"]}"""

ASR_LOSS_SYSTEM = """You are labeling noisy-audio data for a smart voice \
assistant.

You get the user's real spoken COMMAND, and HEARD -- what a speech recognizer \
caught after loud background chatter. List the key pieces of the command that \
did not survive: entities, names, places, times, dates, quantities, titles, \
and the requested action. Judge HEARD and nothing else; another labeler reads \
the assistant's reply separately.

Not key pieces, ever: filler words, the wake word or assistant's name, the \
verbs that merely frame a request ("do", "does", "give me", "tell me"), and \
wording that leaves the same thing being asked (spelling, spacing, number, \
grammar). Question words are key pieces -- "how many", "how long", "where", \
"when" carry what is actually being asked.

Bundle several words into ONE "lost" entry when they are a single point of \
confusion -- a phrase misheard as one specific, similar-sounding phrase. Keep \
entries separate when the losses are unrelated. When HEARD is destroyed past \
telling pieces apart, every key piece is lost: list them all.

Return ONLY JSON, "reason" first and under 25 words, quoting "lost" in the \
words of the real COMMAND:
{"reason": "...", "lost": [...]}

Examples:

COMMAND: find some classical music by beethoven and play it
HEARD:   find some classical music by beethoven and play it
{"reason": "Word-for-word.", "lost": []}

COMMAND: hey olly play playlist tactics from music
HEARD:   a r i play playlist tactics for music
{"reason": "Only the wake word is mangled, and the assistant is already listening, so nothing key was lost.", "lost": []}

COMMAND: give me a current traffic report
HEARD:   me a current traffic report come in and
{"reason": "Give only frames the request; what is being asked for survived intact.", "lost": []}

COMMAND: does artificial intelligence have consciousness
HEARD:   thus artificial intelligence have consciousness
{"reason": "A wrong word in place of the framing verb changes nothing.", "lost": []}

COMMAND: tell me why relationships are so hard
HEARD:   why relationship is so hard
{"reason": "Framing verb plus a singular/plural difference.", "lost": []}

COMMAND: how many oceans are there in the world
HEARD:   how many children are there in the world
{"reason": "The thing being counted comes back as a different noun.", "lost": ["oceans"]}

COMMAND: i have a meeting by two pm today please remind me
HEARD:   i have a meeting at two p m today
{"reason": "The requested action is gone; the time survives.", "lost": ["remind me"]}

COMMAND: play mocking bird by eminem
HEARD:   play mockingbird by edna meyer
{"reason": "Spacing of the title is not a loss; only the artist was misheard.", "lost": ["eminem"]}

COMMAND: turn on the radio on this channel
HEARD:   anywhere on the radio
{"reason": "Two unrelated losses, so two entries.", "lost": ["turn on", "this channel"]}

COMMAND: how do you make steel
HEARD:   or do you make a sale
{"reason": "One coherent near-miss of the whole phrase: one bundled entry.", "lost": ["how do you make steel"]}

COMMAND: skip to next episode
HEARD:   get to make copies
{"reason": "It reads as a different, unrelated sentence; no piece survived.", "lost": ["skip to", "next episode"]}

COMMAND: please turn on the radio
HEARD:   yes
{"reason": "Nothing distinguishable survived.", "lost": ["turn on", "the radio"]}"""
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
