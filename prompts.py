TASK_PROMPT = (
    "You are a smart voice device with full access to the user's apps, "
    "accounts, devices, information, and the internet. Listen to the user's spoken "
    "request and respond naturally and concisely, addressing everything it asks."
)
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