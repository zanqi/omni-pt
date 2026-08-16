# sent-loss tracks (`--sent-2` / `--sent-4`) — implementation guide

## What changes

Today's tree/beam labelers hand an LLM the *real command* and a *witness* (transcript, reply, or
hypothesis) and ask it to invent a free-text list of what went missing. Two problems follow from that:
the key-piece inventory is re-derived on every call (so the two witnesses can disagree about what the
pieces even *are*), and the resulting `lost` entries are free-text quotes that `decide_kind` has to
intersect with a token-overlap heuristic (`PIECE_STOPWORDS`).

sent-loss splits that into two stages:

1. **Stage 1 — key pieces.** One call sees *only* the ground-truth sentence and returns a numbered
   list of its key pieces. This is the canonical inventory for the utterance.
2. **Stage 2 — loss.** One call per witness sees the numbered list plus that witness, and returns
   `{"reason": ..., "lost": [ids]}`. Nothing but numbers come back.

Intersection is then exact set math on ids — no token overlap, no wake-word post-filter on the loss
lists (stage 1 already excluded wake words from the inventory).

Two new tracks, per the decisions:

| flag | `TRACK` | witnesses | stage-2 calls per probe |
| --- | --- | --- | --- |
| `--sent-2` | `"sent-2"` | ASR transcript + base-model task reply | 2 |
| `--sent-4` | `"sent-4"` | 4 ASR beam hypotheses | 4 (parallel) |

A piece is lost only if **every** witness on that track reports it lost. `misheard_as` and
`unintelligible` are dropped on both tracks — repair questions always ask openly
(`REPAIR_TARGET_TREE_SYSTEM`), and a probe that lost too much of the inventory to anchor a question
on is caught by `LOST_MAX_PCT` (step 5).

---

## Step 1 — `prompts.py`: `KEY_PIECES_SYSTEM`

Stage 1 owns the key-info definition for every downstream stage, so this prompt merges the exclusion
rules that are currently spread across `CLASSIFY_SYSTEM` (the "key info" paragraph, prompts.py:591),
`ASR_LOSS_SYSTEM` (prompts.py:406), and `BEAM_LOSS_SYSTEM` (prompts.py:676) — rewritten few-shot,
matching the house style (`COMMAND:` block, JSON with `reason` first, occasional `--` gloss line under
an example that teaches a rule no other example teaches).

Add near the other labeler prompts:

```python
# ---
# --sent-2 / --sent-4: stage 1, the canonical key-piece inventory
# ---

# The one place "key piece" is defined on these tracks. Every stage-2 call is
# scored against the list this returns, so the exclusions that used to be
# restated in each labeler (wake words, framing verbs, implied words, spelling
# and phonetic variants) live here and here only.
KEY_PIECES_SYSTEM = """You are labeling noisy-audio data for a smart voice \
assistant.

You get the user's real spoken COMMAND, clean and complete. Break it into its \
KEY PIECES: the pieces the assistant must have heard to carry the command out \
correctly. A piece is key ONLY if the task cannot be performed correctly \
without it.

Key pieces are entities, names, places, times, dates, quantities, titles, and \
the requested action or topic. Question words are key pieces -- "how many", \
"how long", "where", "when" carry what is actually being asked.

Never key pieces:
- filler and politeness words ("please", "could you", "a little bit")
- the wake word or the assistant's own name ("hey olly", "ok google", \
"assistant"). The assistant is already listening, so nothing about the task \
depends on it hearing its own name. Never list it, however prominent it is.
- auxiliary and light verbs that only frame the request ("do", "does", "is", \
"can you", "give me", "tell me", "get me", "put"). The key piece is what is \
being asked FOR, not the words wrapping the asking.
- a word whose meaning the rest of the command already implies ("set" in "are \
there any alarms set")

Bundle words into ONE piece when they are a single slot the user would supply \
as a unit -- "seven am", "my shopping list", "mocking bird". Keep pieces \
separate when they are unrelated things the user chose independently. A later \
labeler is told that exactly one piece went missing and must write one \
question recovering it, so a bundle spanning half the command makes that \
impossible.

Quote each piece in the words of the COMMAND, in the order they were spoken.

Return ONLY JSON, "reason" first and under 20 words:
{"reason": "...", "pieces": [...]}

Examples:

COMMAND: set an alarm for seven am tomorrow
{"reason": "The action plus the two values it needs.", "pieces": ["set an alarm", "seven am", "tomorrow"]}

COMMAND: hey olly play playlist tactics from music
{"reason": "The wake word is not a key piece.", "pieces": ["play", "playlist tactics", "music"]}

COMMAND: give me a current traffic report
{"reason": "Give me only frames the request; what is asked for is one thing.", "pieces": ["current traffic report"]}

COMMAND: does artificial intelligence have consciousness
{"reason": "Does is the framing auxiliary; the subject and the property asked about are the pieces.", "pieces": ["artificial intelligence", "have consciousness"]}

COMMAND: do you think it's going to rain tomorrow
{"reason": "Do you think only opens the question.", "pieces": ["rain", "tomorrow"]}

COMMAND: how many oceans are there in the world
{"reason": "The question word carries what is being asked, so it is a piece.", "pieces": ["how many", "oceans", "in the world"]}

COMMAND: play mocking bird by eminem
{"reason": "Action, title, artist.", "pieces": ["play", "mocking bird", "eminem"]}

COMMAND: add milk to my shopping list and remind me at six
{"reason": "Two requests joined; each keeps its own pieces.", "pieces": ["add", "milk", "my shopping list", "remind me", "six"]}

COMMAND: brighten the lights a little bit
{"reason": "A little bit is filler; the action and the device are the pieces.", "pieces": ["brighten", "the lights"]}

COMMAND: turn on the radio on this channel
{"reason": "Three independent slots, so three pieces.", "pieces": ["turn on", "the radio", "this channel"]}

COMMAND: hey olly are there any alarms set
{"reason": "Set is implied by the rest; the wake word never counts; one thing is being asked.", "pieces": ["are there any alarms"]}
   -- a one-piece command is normal. Do not pad the list to make it longer"""
```

The last example matters: a one-piece command that loses its piece is 100% of the inventory, so
`LOST_MAX_PCT` files it as `repeat` (see step 5), which is the behavior `CLASSIFY_SYSTEM` already
specifies. Padding the inventory would break it.

## Step 2 — `prompts.py`: the two stage-2 prompts

Both take a numbered `KEY PIECES` block and one witness, and return ids only. They differ only in
which witness they read and in how "survived" is evidenced.

`SENT_ASR_LOSS_SYSTEM` also serves `--sent-4`, one call per hypothesis — a hypothesis *is* a
`HEARD` line, so no third prompt is needed.

```python
SENT_ASR_LOSS_SYSTEM = """You are labeling noisy-audio data for a smart voice \
assistant.

You get KEY PIECES -- a numbered list of the pieces of the user's real spoken \
command that the assistant must have heard to carry it out -- and HEARD, what \
a speech recognizer caught of that same command after loud background chatter. \
Say which numbered pieces did not survive. Judge HEARD and nothing else.

A piece SURVIVED when HEARD carries the same thing, judged by meaning rather \
than wording:
- spelling, spacing, number or grammar differences ("mockingbird" for "mocking \
bird", "10 a m" for "ten am", "relationship" for "relationships")
- a synonym asking for the same thing ("raise" for "increase")
- a question rephrased in different grammar that still asks the same thing
- a name rendered as a close phonetic match ("powell" for "pawel", "deevya" \
for "divya"). The recognizer heard the name and spelled it its own way, so the \
slot is filled and the user has nothing to clarify.

A piece is LOST when HEARD drops it, garbles it, or puts a DIFFERENT word in \
its place rather than a spelling of the same sounds -- "monday" for "mona" (a \
weekday, not that name), "edna meyer" for "eminem", "children" for "oceans". \
When HEARD reads as some other sentence entirely, every piece is lost.

Words in HEARD that no key piece accounts for are not evidence of anything: \
the recognizer inventing extra words is normal at every noise level.

Return ONLY JSON, "reason" first and under 25 words. "lost" holds the NUMBERS \
of the lost pieces and nothing else:
{"reason": "...", "lost": [2, 3]}

Examples:

KEY PIECES:
1. find some classical music
2. beethoven
3. play it
HEARD: find some classical music by beethoven and play it
{"reason": "Word-for-word.", "lost": []}

KEY PIECES:
1. play
2. playlist tactics
3. music
HEARD: a r i play playlist tactics for music
{"reason": "Only the wake word is mangled, and it is not on the list.", "lost": []}

KEY PIECES:
1. current traffic report
HEARD: me a current traffic report come in and
{"reason": "What is asked for survived intact; the trailing junk accounts for nothing.", "lost": []}

KEY PIECES:
1. how many
2. oceans
3. in the world
HEARD: how many children are there in the world
{"reason": "The thing being counted comes back as a different noun.", "lost": [2]}

KEY PIECES:
1. a meeting
2. two pm
3. today
4. remind me
HEARD: i have a meeting at two p m today
{"reason": "The requested action is gone; the time and day survive as spelled variants.", "lost": [4]}

KEY PIECES:
1. play
2. mocking bird
3. eminem
HEARD: play mockingbird by edna meyer
{"reason": "Spacing of the title is not a loss; only the artist was misheard.", "lost": [3]}

KEY PIECES:
1. turn on
2. the radio
3. this channel
HEARD: anywhere on the radio
{"reason": "The radio survives; the action and the channel do not.", "lost": [1, 3]}

KEY PIECES:
1. skip to
2. next episode
HEARD: get to make copies
{"reason": "It reads as a different, unrelated sentence; no piece survived.", "lost": [1, 2]}

KEY PIECES:
1. turn on
2. the radio
HEARD: yes
{"reason": "Nothing distinguishable survived.", "lost": [1, 2]}"""
```

```python
SENT_RESP_LOSS_SYSTEM = """You are labeling noisy-audio data for a smart voice \
assistant.

You get KEY PIECES -- a numbered list of the pieces of the user's real spoken \
command that the assistant must have heard to carry it out -- and REPLY, what \
the assistant said after hearing that same command over loud background \
chatter. The assistant never saw the command in text; REPLY is your only \
evidence of what reached it.

Read REPLY as EVIDENCE, not as work to grade. A rambling, unhelpful, evasive \
or plain wrong reply can still prove the command was heard perfectly, and that \
is all you are here to establish. Judge REPLY and nothing else; another \
labeler reads the recognizer's transcript separately.

A piece SURVIVED when the reply uses it correctly, judged by meaning rather \
than wording. A paraphrase, a capability decline, a hand-off to the user, or \
an admission of not knowing the ANSWER all still name what was asked. Naming a \
piece correctly settles it as heard even when the reply then asks a follow-up \
about it: a command that referred to something only generically ("that \
podcast", "this event") gave no specific value, so asking WHICH one is the \
assistant gathering information it never had.

A piece is LOST when the reply neither uses it nor otherwise shows it got \
through, or asserts a WRONG value for it. When the reply asks for the whole \
command again, or engages with nothing in it at all, no piece is shown heard: \
list them all.

Return ONLY JSON, "reason" first and under 25 words. "lost" holds the NUMBERS \
of the lost pieces and nothing else:
{"reason": "...", "lost": [2, 3]}

Examples:

KEY PIECES:
1. set an alarm
2. seven am
3. tomorrow
REPLY: i'm setting your alarm for 7 am tomorrow.
{"reason": "Names the alarm, the time and the day.", "lost": []}

KEY PIECES:
1. play
2. my most recent playlist
REPLY: i'm not sure how to play your playlist. you might need to open the music app on your device and find the playlist there.
{"reason": "Handing the job back is a poor reply but names playing the playlist.", "lost": []}

KEY PIECES:
1. how many
2. lists
3. remind me
REPLY: i'm not sure how many lists you have. you could check your phone's settings.
{"reason": "Not knowing the ANSWER is not failing to hear the QUESTION; only the requested action goes unengaged.", "lost": [3]}

KEY PIECES:
1. play
2. that podcast
REPLY: i'm not sure which "that podcast" you're referring to. can you be more specific?
{"reason": "Quotes the podcast back; the command never said which one, so asking is information gathering.", "lost": []}

KEY PIECES:
1. artificial intelligence
2. have consciousness
REPLY: well, you know, artificial intelligence can be really good at making us feel more in control.
{"reason": "It wanders off, but plainly heard the subject; only the property asked about goes unengaged.", "lost": [2]}

KEY PIECES:
1. add
2. milk
3. my shopping list
4. remind me
5. six
REPLY: i'm adding milk to your shopping list.
{"reason": "Only half the command is addressed.", "lost": [4, 5]}

KEY PIECES:
1. play
2. mocking bird
3. eminem
REPLY: playing mockingbird by edna meyer now.
{"reason": "The artist comes back as a different name.", "lost": [3]}

KEY PIECES:
1. how many
2. oceans
3. in the world
REPLY: i'm sorry, i can't help with that request.
{"reason": "Boilerplate that would decline any request equally evidences nothing.", "lost": [1, 2, 3]}

KEY PIECES:
1. brighten
2. the lights
REPLY: i'm not sure what you mean by "write in the lights a little bit." could you be referring to a specific task?
{"reason": "The lights is carried over correctly, so only the verb is in doubt.", "lost": [1]}

KEY PIECES:
1. turn on
2. the radio
3. this channel
REPLY: it's really loud in here -- what was that?
{"reason": "Asks for the whole thing again, engaging with nothing.", "lost": [1, 2, 3]}"""
```

Then extend the import block at [babble_data.py:24-39](../babble_data.py#L24-L39) with
`KEY_PIECES_SYSTEM`, `SENT_ASR_LOSS_SYSTEM`, `SENT_RESP_LOSS_SYSTEM`.

## Step 3 — `babble_data.py`: stage 1, cached

Stage 1's only input is the sentence, and `slurp_id` is exactly the id of a distinct sentence (SLURP
streams several recordings of the same prompt back to back, all sharing one `slurp_id`) — so key the
cache on `slurp_id`. It is a short, already-unique key that needs no text normalization to be a
correct key, where the sentence string only becomes one after `_normalize_text` and only stays one as
long as nothing upstream re-cases or re-punctuates it.

This is not a cross-utterance dedupe either way: `build_triplets` / `build_answer_rows` claim each
`slurp_id` in `seen_slurp_ids` on first sight, so a build never labels the same sentence twice. The
cache earns its keep *within* one utterance — every kind and every `MAX_PROBES` redraw reuses one
stage-1 call.

Both call sites already hold `slurp_id` ([babble_data.py:1197](../babble_data.py#L1197),
[1316](../babble_data.py#L1316)) but `probe_by_kinds` does not take it, so step 7 threads it through.

Place it next to `label_loss` (around [babble_data.py:399](../babble_data.py#L399)):

```python
# stage 1 depends only on the sentence, so it is computed once and reused
# across every probe batch, every kind, and every MAX_PROBES redraw of the
# utterance. Keyed on slurp_id, which IS the id of a distinct sentence (the
# several recordings slurp streams of one prompt all share it), so no text
# normalization stands between the key and its identity.
_SLURP_ID_2_KEY_PIECES = {}
_KEY_PIECES_LOCK = threading.Lock()


def key_pieces(slurp_id, sentence):
    """slurp_id, sentence -> [piece text, ...] | None, one per key piece.

    The list index + 1 is the id every stage-2 call speaks in.
    """
    with _KEY_PIECES_LOCK:
        if slurp_id in _SLURP_ID_2_KEY_PIECES:
            return _SLURP_ID_2_KEY_PIECES[slurp_id]

    cmd = _normalize_text(sentence)

    obj = gpt_json(
        KEY_PIECES_SYSTEM,
        f"COMMAND: {cmd}",
        temperature=CLASSIFY_TEMPERATURE,
        max_tokens=CLASSIFY_MAX_TOKENS,
    )
    pieces = None
    if obj is not None:
        # the wake-word filter still runs: the prompt says never to list one and
        # every labeler prompt in this repo occasionally does anyway
        pieces = drop_wake_only(
            [str(s).strip() for s in obj.get("pieces", []) if str(s).strip()]
        )
        if not pieces:
            pieces = None

    with _KEY_PIECES_LOCK:
        _SLURP_ID_2_KEY_PIECES[slurp_id] = pieces
    return pieces
```

A `None` here (bad JSON, or every piece filtered away) means the utterance cannot be labeled at all —
handle it as a skip in step 6, not as an empty inventory.

## Step 4 — `babble_data.py`: stage 2

One function serves both tracks; it differs only in rubric and witness, exactly as `label_loss` does
today. Keep `label_loss` for the tree track and add alongside it:

```python
def lost_pieces(system: str, pieces: list[str], witness_line: str):
    """Ask LLM for lost within pieces.
    Returns a set for intersection downstream

    Ids outside 1..len(pieces) are dropped rather than retried: they are rare,
    and a hallucinated id would silently push a repair to a repeat.
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
```

## Step 5 — `babble_data.py`: `decide_kind_ids`

Same 0/1/≥2 mapping as `decide_kind` — agreed pieces to answer/repair/repeat — but agreement is
exact id intersection, and "how much was lost" is a fraction of the inventory instead of a
token-overlap ratio against the sentence. Written to take N sides so the two-witness and
4-hypothesis tracks share it.

Three departures from `decide_kind`:

- **One `lost` list, no `lost_piece` / `misheard_as`.** The pieces are already exact strings out of
  the stage-1 inventory, so the free-text pair (`lost_piece` for the target writer, `missing` for
  diagnostics) collapses to one list — on repair it is a 1-element list holding exactly what the
  target writer needs. `misheard_as` was always `""` on this track anyway (the sent rubrics report
  ids, never a substitute wording), and `REPAIR_TARGET_TREE_SYSTEM` never renders it. Step 7 adapts
  the probe-dict site to both.
- **`LOST_MAX_PCT` is measured in words of the inventory, not in pieces.** `1/len(pieces)` treats
  every piece as equally big; what actually matters is whether the agreed piece *is* most of the
  command. This is stricter on short inventories — a 2-piece command that lost the wordier piece
  files as `repeat` where the count rule kept it as `repair` — so it is the first thing to check if
  repair rows are starved (step 9).
- **No all-sides-wrecked override.** `decide_kind`'s second rule (every witness on its own lost most
  of the command ⇒ `repeat`, whatever they share) is deliberately not carried over. The exact-id
  intersection makes the coincidental-agreement case it guards against much rarer than it is between
  two free-text quotes, and the per-side counts are in `classifier_reason` if the built rows say
  otherwise.

```python
def decide_kind_ids(pieces, sides: list[set[int]]):
    agreed = sorted(set.intersection(*sides)) if sides else []
    kind = "answer" if not agreed else "repair" if len(agreed) == 1 else "repeat"
    lost = [pieces[i - 1] for i in agreed]
    total_wc = sum(len(p.split()) for p in pieces)
    if kind == "repair" and len(lost[0].split()) / total_wc >= LOST_MAX_PCT:
        kind = "repeat"

    return {
        "kind": kind,
        # on repeat this is the agreed pieces, not the union of the sides --
        # a diagnostic column only, since repeat targets come from REPEAT_POOL
        "lost": lost,
        "asr_bucket": f"a{len(sides[0])}",
        "resp_bucket": f"r{len(sides[-1])}",
        "reason": (
            f"{len(pieces)} pieces | "
            + " x ".join(str(sorted(s)) for s in sides)
            + f" -> agreed {agreed}"
        ),
    }
```

`drop_wake_only`, `lost_too_much`, and `PIECE_STOPWORDS` are untouched — the tree and beam tracks
still use them; the sent tracks simply never call them on loss lists.

## Step 6 — `babble_data.py`: the two labeler entry points

Next to `label_tree` / `label_beam`:

```python
def label_sent(slurp_id, sentence, transcript, response):
    """--sent-2: inventory once, then one loss call per witness."""
    if not transcript or not response:
        return None
    pieces = key_pieces(slurp_id, sentence)
    if pieces is None:
        return None
    asr_ids = lost_pieces(
        SENT_ASR_LOSS_SYSTEM, pieces, f"HEARD: {_normalize_text(transcript)}"
    )
    resp_ids = lost_pieces(
        SENT_RESP_LOSS_SYSTEM, pieces, f"REPLY: {_normalize_text(response)}"
    )
    if asr_ids is None or resp_ids is None:
        return None
    return {**decide_kind_ids(pieces, [asr_ids, resp_ids]), "pieces": pieces}


def label_sent_beam(slurp_id, sentence, hyps):
    """--sent-4: inventory once, then ASR_N_BEST loss calls in parallel."""
    hyps = [h for h in hyps if h and h.strip()]
    if not hyps:
        return None
    pieces = key_pieces(slurp_id, sentence)
    if pieces is None:
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
    return {**decide_kind_ids(pieces, sides), "pieces": pieces}
```

The nested `ThreadPoolExecutor` sits inside the one `probe_by_kinds` already opens with
`CLASSIFY_WORKERS`, so a beam probe batch has `CLASSIFY_WORKERS * ASR_N_BEST` calls in flight. If the
vLLM box queues badly, drop `CLASSIFY_WORKERS` for this track rather than serializing the hypotheses.

## Step 7 — wire the tracks

`prompts.py` needs nothing further; the rest is flag plumbing in `babble_data.py`.

**Two flags, two `TRACK` strings.** `TRACK` ([babble_data.py:131](../babble_data.py#L131)) is the
module-level track selector every branch below reads; the flag name and the string it resolves to
are independent (`--tree-label` → `"tree"`, `--beam-label` → `"beam"`). The two sent tracks are:

| flag | `TRACK` | witnesses |
| --- | --- | --- |
| `--sent-2` | `"sent-2"` | ASR transcript + base-model task reply, 2 stage-2 calls |
| `--sent-4` | `"sent-4"` | the `ASR_N_BEST` beam hypotheses, 4 stage-2 calls in parallel |

**Argparse** ([babble_data.py:1530-1557](../babble_data.py#L1530-L1557)) — two entries in the existing
mutually-exclusive group:

```python
track.add_argument("--sent-2", action="store_true")

track.add_argument("--sent-4", action="store_true")
```

**TRACK resolution** ([babble_data.py:1561](../babble_data.py#L1561)) — extend the chain with
`"sent-2" if args.sent_2 else "sent-4" if args.sent_4` before the `"two-pass"` fallback. Argparse
turns the dashes into underscores, so the attributes are `args.sent_2` / `args.sent_4`.

**Every `TRACK in (...)` / `TRACK ==` site.** Five places, all needing the sent tracks added:

| site | change |
| --- | --- |
| [1567](../babble_data.py#L1567) `if TRACK == "beam"` (batch size) | `in ("beam", "sent-4")` — only the beam-decoding track needs the smaller batch |
| [1583](../babble_data.py#L1583) repeat-pool build | `in ("tree", "beam", "sent-2", "sent-4")` |
| [946](../babble_data.py#L946) `write_target` repair branch | `TRACK in ("tree", "sent-2", "sent-4")` → `REPAIR_TARGET_TREE_SYSTEM` (open question, no `misheard_as`) |
| [1340](../babble_data.py#L1340), [1451](../babble_data.py#L1451) build/answer-row target path | `in ("tree", "beam", "sent-2", "sent-4")` |
| [1147](../babble_data.py#L1147) `task = TASK_PROMPT_TREE if TRACK == "tree"` | `in ("tree", "sent-2")` — the restating reply is what `SENT_RESP_LOSS_SYSTEM` reads, and `--sent-4` has no reply pass at all |

`SLOT_SNR_DISJOINT` and `CLEAN_ANSWER_PROB` are already gated on `TRACK != "two-pass"`
([998](../babble_data.py#L998), [1005](../babble_data.py#L1005)), so both sent tracks pick them up
with no edit.

**`probe_by_kinds` signature** ([babble_data.py:985](../babble_data.py#L985)) — the sent tracks' cache
key has to reach `key_pieces`, so take it next to the sentence it identifies:

```python
def probe_by_kinds(clean, pool, slurp_id, sentence, kinds_need, batch_size, rng):
```

Both callers already have `slurp_id` bound near the call — they use it for the self-exclusion filter
on `babble_pool` and for the per-utterance `random.Random` seed — so each is a one-line insertion.
The other tracks ignore the argument.

**Probe dispatch in `probe_by_kinds`** ([babble_data.py:1105-1163](../babble_data.py#L1105-L1163)):

- `--sent-4` joins the existing beam branch (same GPU pass, same `hyp_lists`, no task-response
  decode). Change the branch guard to `TRACK in ("beam", "sent-4")` and hoist the labeler out of the
  `ex.map` call:

  ```python
  label_hyps = (
      (lambda h: label_sent_beam(slurp_id, sentence, h))
      if TRACK == "sent-4"
      else (lambda h: label_beam(sentence, h))
  )
  ```
- `--sent-2` joins the final `else` branch alongside tree/two-pass. Extend the `label_one`
  selection to a three-way choice rather than a nested conditional expression — at three tracks the
  ternary chain stops reading:

  ```python
  if TRACK == "two-pass":
      label_one = classify
  elif TRACK == "sent-2":
      label_one = lambda t, r: label_sent(slurp_id, sentence, t, r)
  else:
      label_one = lambda t, r: label_tree(sentence, t, r)
  ```

**Probe result dict** ([1180-1216](../babble_data.py#L1180-L1216)) — the sent label dict carries
`lost` where the other labelers carry `missing` / `misheard_as` / `lost_piece`, so resolve it into a
local *above* the dict and read the fallbacks off that:

```python
kind = label["kind"]
# tree/beam/two-pass call it "missing" and quote it out of the command; the
# sent tracks call it "lost" and it is already exact key-piece text
lost = label.get("lost", label.get("missing", []))
if kind in results and results[kind] is None:
    results[kind] = {
        ...
        "lost": lost,
        "swapped": [label["misheard_as"]] if label.get("misheard_as") else [],
        ...
        # the sent tracks don't return this: on repair their `lost` IS the one
        # piece, so write_target's prompt and its leak filter read it off that
        "lost_piece": label.get(
            "lost_piece", lost[0] if kind == "repair" and lost else ""
        ),
        # sent tracks only: the stage-1 inventory the label's ids indexed into
        "pieces": label.get("pieces", []),
    }
```

The local is not just tidiness: `dict.get`'s default argument is evaluated eagerly, so writing
`label.get("lost_piece", label["lost"][0] ...)` inline raises `KeyError` on every tree/beam/two-pass
probe, whose labels have no `lost` key at all.

Nothing else needs to know which track it is: `write_target`'s repair branch, its `leakable` leak
filter, and `make_row`'s `lost_piece` column all keep reading `probe["lost_piece"]`, and
`probe["swapped"]` stays empty on these tracks so `REPAIR_TARGET_TREE_SYSTEM`'s open question is the
only thing that can be rendered.

## Step 8 — row schema

In `make_row` ([babble_data.py:1263](../babble_data.py#L1263)), add one column, keeping the additive
pattern the other tracks use (empty on tracks that don't fill it):

```python
# sent tracks: the stage-1 key-piece inventory the label's ids indexed into,
# so a mislabeled row can be re-read without re-running stage 1
"key_pieces": probe["pieces"],
```

`kind`, `target`, `lost_piece`, `classifier_reason` keep their existing meanings, so
`sft_qwen.py`, `babble_eval_qwen.py`, and `results/viz.ipynb` need no changes.

## Step 9 — run it

```bash
conda activate qwen25omni

# two-witness variant, small
python babble_data.py --ds-id keylazy/slurp-sent2-v0 --sent-2 \
    --n-test 5 --n-train 5 --n-extra-ans 0 --no-push

# 4-hypothesis variant
python babble_data.py --ds-id keylazy/slurp-sent4-v0 --sent-4 \
    --n-test 5 --n-train 5 --n-extra-ans 0 --no-push
```

Read `babble_audio/slurp-sent2-v0/rows.json` and check, in this order:

1. **`key_pieces`** — is the inventory right for the sentence? Wrong inventories poison everything
   downstream, and they are the one thing stage 2 cannot recover from. Watch for wake words surviving
   `drop_wake_only`, framing verbs listed as pieces, and one-piece commands padded to three.
2. **`classifier_reason`** — the per-side id sets and the agreed set are printed there. A track where
   the sides never intersect means the two rubrics disagree about what "survived" means, not that the
   audio was clean.
3. **kind distribution** — the `Counter` the build logs. If `repair` is starved, `LOST_MAX_PCT`
   against a short inventory is the first suspect: it is measured in words, so a 2-piece command
   that lost the wordier of the two is already at or past the 0.6 threshold and files as `repeat`.
   `classifier_reason` prints `<n> pieces | ...` so the inventory length is right there.
4. **`lost_piece` vs `target`** — the repair question must not speak the lost piece; `skip` counts
   `target-leak` for the ones that did and retried.

Then a full build with the usual `--n-train` / `--n-test`, push, and run `sft.sh` / `eval.sh` against
the new `--ds-id` as with any other track.
