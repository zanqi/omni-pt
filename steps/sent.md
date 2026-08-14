# sent-loss track — implementation guide

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

| flag | witnesses | stage-2 calls per probe |
| --- | --- | --- |
| `--sent-loss` | ASR transcript + base-model task reply | 2 |
| `--sent-beam` | 4 ASR beam hypotheses | 4 (parallel) |

A piece is lost only if **every** witness on that track reports it lost. `misheard_as` and
`unintelligible` are dropped on both tracks — repair questions always ask openly
(`REPAIR_TARGET_TREE_SYSTEM`), and the `LOST_MAX_PCT` override covers the wrecked case.

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
# --sent-loss / --sent-beam: stage 1, the canonical key-piece inventory
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

The last example matters: with an exact-count `decide_kind`, a one-piece command that loses its piece
becomes `repeat` (see step 5), which is the behavior `CLASSIFY_SYSTEM` already specifies. Padding the
inventory would break it.

## Step 2 — `prompts.py`: the two stage-2 prompts

Both take a numbered `KEY PIECES` block and one witness, and return ids only. They differ only in
which witness they read and in how "survived" is evidenced.

`SENT_ASR_LOSS_SYSTEM` also serves `--sent-beam`, one call per hypothesis — a hypothesis *is* a
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

Stage 1's only input is the sentence, so cache on the normalized sentence rather than on `slurp_id`:
SLURP streams several recordings of the same prompt back to back, so a sentence key also dedupes
across utterances, and no call site has to thread a `slurp_id` through.

Place it next to `label_loss` (around [babble_data.py:399](../babble_data.py#L399)):

```python
# stage 1 depends only on the sentence, so it is computed once per distinct
# sentence and reused across every probe batch, every kind, and every
# MAX_PROBES redraw. SLURP repeats the same prompt across recordings, so the
# key is the sentence, not the slurp_id.
_PIECES_CACHE = {}
_PIECES_LOCK = threading.Lock()

# a command with more pieces than this is not a repair candidate under any
# labeling, and a long inventory makes the id space noisy; clamp rather than
# skip, since decide_kind's LOST_MAX_PCT will file it as repeat anyway
PIECES_MAX = 12


def key_pieces(sentence):
    """sentence -> [piece text, ...] | None, one entry per key piece.

    The list index + 1 is the id every stage-2 call speaks in.
    """
    cmd = _normalize_text(sentence)
    with _PIECES_LOCK:
        if cmd in _PIECES_CACHE:
            return _PIECES_CACHE[cmd]

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
        )[:PIECES_MAX]
        if not pieces:
            pieces = None

    with _PIECES_LOCK:
        _PIECES_CACHE[cmd] = pieces
    return pieces
```

A `None` here (bad JSON, or every piece filtered away) means the utterance cannot be labeled at all —
handle it as a skip in step 6, not as an empty inventory.

## Step 4 — `babble_data.py`: stage 2

One function serves both tracks; it differs only in rubric and witness, exactly as `label_loss` does
today. Keep `label_loss` for the tree track and add alongside it:

```python
def sent_loss(system, pieces, witness_line):
    """(rubric, inventory, rendered witness line) -> set of lost ids | None.

    Ids outside 1..len(pieces) are dropped rather than retried: they are rare,
    and a hallucinated id would silently push a repair to a repeat.
    """
    listing = "\n".join(f"{i}. {p}" for i, p in enumerate(pieces, 1))
    obj = gpt_json(
        system,
        f"KEY PIECES:\n{listing}\n{witness_line}",
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

Same rules as `decide_kind` — 0/1/≥2 agreed pieces map to answer/repair/repeat, plus the
`LOST_MAX_PCT` wrecked override — but agreement is exact id intersection, and "how much was lost"
is now a fraction of the inventory instead of a token-overlap ratio. Written to take N sides so the
two-witness and 4-hypothesis tracks share it:

```python
def decide_kind_ids(pieces, sides):
    """(inventory, [set of lost ids per witness]) -> label dict.

    `sides` is [asr_ids, resp_ids] on --sent-loss and the four per-hypothesis
    id sets on --sent-beam. A piece counts lost only when EVERY side reports
    it: one witness getting a piece right proves it was audible, however badly
    the others mangled it. This is an intersection, not a vote -- the rule
    BEAM_LOSS_SYSTEM states in prose, now enforced in code.

    Two overrides file an otherwise-repair probe as repeat:
      - every side on its own lost most of the inventory: an intersection of
        one there is a coincidence between two wrecks, not a clean hole
      - the agreed piece is >= LOST_MAX_PCT of the whole inventory, which is
        how a one-piece command that lost its piece lands (nothing survives to
        anchor a question on)
    """
    agreed = sorted(set.intersection(*sides)) if sides else []
    kind = "answer" if not agreed else "repair" if len(agreed) == 1 else "repeat"
    lost_piece = pieces[agreed[0] - 1] if kind == "repair" else ""

    wrecked = all(len(s) / len(pieces) >= LOST_MAX_PCT for s in sides)
    if wrecked or (kind == "repair" and 1 / len(pieces) >= LOST_MAX_PCT):
        kind, lost_piece = "repeat", ""

    return {
        "kind": kind,
        "lost_piece": lost_piece,
        # no witness of a similar-sounding substitute on these tracks, so a
        # repair question always asks openly
        "misheard_as": "",
        "missing": (
            [lost_piece]
            if kind == "repair"
            else [pieces[i - 1] for i in sorted(set().union(*sides))]
            if kind == "repeat"
            else []
        ),
        "asr_bucket": f"a{len(sides[0])}",
        "resp_bucket": f"r{len(sides[-1])}",
        "reason": (
            f"{len(pieces)} pieces | "
            + " x ".join(str(sorted(s)) for s in sides)
            + f" -> agreed {agreed}"
            + (" (all wrecked)" if wrecked else "")
        ),
    }
```

`drop_wake_only`, `lost_too_much`, and `PIECE_STOPWORDS` are untouched — the tree and beam tracks
still use them; sent-loss simply never calls them on loss lists.

## Step 6 — `babble_data.py`: the two labeler entry points

Next to `label_tree` / `label_beam`:

```python
def label_sent(sentence, transcript, response):
    """--sent-loss: inventory once, then one loss call per witness."""
    if not transcript or not response:
        return None
    pieces = key_pieces(sentence)
    if pieces is None:
        return None
    asr_ids = sent_loss(
        SENT_ASR_LOSS_SYSTEM, pieces, f"HEARD: {_normalize_text(transcript)}"
    )
    resp_ids = sent_loss(
        SENT_RESP_LOSS_SYSTEM, pieces, f"REPLY: {_normalize_text(response)}"
    )
    if asr_ids is None or resp_ids is None:
        return None
    return {**decide_kind_ids(pieces, [asr_ids, resp_ids]), "pieces": pieces}


def label_sent_beam(sentence, hyps):
    """--sent-beam: inventory once, then ASR_N_BEST loss calls in parallel."""
    hyps = [h for h in hyps if h and h.strip()]
    if not hyps:
        return None
    pieces = key_pieces(sentence)
    if pieces is None:
        return None
    with ThreadPoolExecutor(max_workers=len(hyps)) as ex:
        sides = list(
            ex.map(
                lambda h: sent_loss(
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

**Argparse** ([babble_data.py:1404-1425](../babble_data.py#L1404-L1425)) — two entries in the existing
mutually-exclusive group:

```python
track.add_argument(
    "--sent-loss",
    action="store_true",
    help="Two probe passes as usual, but labeled in two stages: one call "
    "lists the command's key pieces, then one call per witness says which "
    "numbered pieces it lost. A piece counts lost only if both witnesses "
    "report it.",
)
track.add_argument(
    "--sent-beam",
    action="store_true",
    help="As --sent-loss, but the witnesses are the ASR_N_BEST beam "
    "hypotheses, labeled in parallel against the same key-piece list. No "
    "task-response pass.",
)
```

**TRACK resolution** ([babble_data.py:1428](../babble_data.py#L1428)) — extend the chain to
`"sent"` / `"sent-beam"`.

**Every `TRACK in (...)` / `TRACK ==` site.** Five places, all needing the sent tracks added:

| site | change |
| --- | --- |
| [1434](../babble_data.py#L1434) `if TRACK == "beam"` (batch size) | `in ("beam", "sent-beam")` |
| [1451](../babble_data.py#L1451) repeat-pool build | `in ("tree", "beam", "sent", "sent-beam")` |
| [820](../babble_data.py#L820) `write_target` repair branch | `TRACK in ("tree", "sent", "sent-beam")` → `REPAIR_TARGET_TREE_SYSTEM` (open question, no `misheard_as`) |
| [1214](../babble_data.py#L1214), [1325](../babble_data.py#L1325) build/answer-row target path | `in ("tree", "beam", "sent", "sent-beam")` |
| [1021](../babble_data.py#L1021) `task = TASK_PROMPT_TREE if TRACK == "tree"` | `in ("tree", "sent")` — the restating reply is what `SENT_RESP_LOSS_SYSTEM` reads |

`SLOT_SNR_DISJOINT` and `CLEAN_ANSWER_PROB` are already gated on `TRACK != "two-pass"`
([872](../babble_data.py#L872), [880](../babble_data.py#L880)), so both sent tracks pick them up with
no edit.

**Probe dispatch in `probe_by_kinds`** ([babble_data.py:979-1037](../babble_data.py#L979-L1037)):

- `--sent-beam` joins the existing beam branch (same GPU pass, same `hyp_lists`, no task-response
  decode). Change the branch guard to `TRACK in ("beam", "sent-beam")` and pick the labeler:
  `label_beam` vs `lambda h: label_sent_beam(sentence, h)`.
- `--sent-loss` joins the final `else` branch alongside tree/two-pass. Extend the `label_one`
  selection to a three-way choice rather than a nested conditional expression — at three tracks the
  ternary chain stops reading:

  ```python
  if TRACK == "two-pass":
      label_one = classify
  elif TRACK == "sent":
      label_one = lambda t, r: label_sent(sentence, t, r)
  else:
      label_one = lambda t, r: label_tree(sentence, t, r)
  ```

**Probe result dict** ([1046-1069](../babble_data.py#L1046-L1069)) — add
`"pieces": label.get("pieces", [])` so the inventory reaches the row.

## Step 8 — row schema

In `make_row` ([babble_data.py:1129](../babble_data.py#L1129)), add one column, keeping the additive
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
python babble_data.py --ds-id keylazy/slurp-sent-v0 --sent-loss \
    --n-test 5 --n-train 5 --n-extra-ans 0 --no-push

# 4-hypothesis variant
python babble_data.py --ds-id keylazy/slurp-sentbeam-v0 --sent-beam \
    --n-test 5 --n-train 5 --n-extra-ans 0 --no-push
```

Read `babble_audio/slurp-sent-v0/rows.json` and check, in this order:

1. **`key_pieces`** — is the inventory right for the sentence? Wrong inventories poison everything
   downstream, and they are the one thing stage 2 cannot recover from. Watch for wake words surviving
   `drop_wake_only`, framing verbs listed as pieces, and one-piece commands padded to three.
2. **`classifier_reason`** — the per-side id sets and the agreed set are printed there. A track where
   the sides never intersect means the two rubrics disagree about what "survived" means, not that the
   audio was clean.
3. **kind distribution** — the `Counter` the build logs. If `repair` is starved, `LOST_MAX_PCT`
   against a short inventory is the first suspect: a 2-piece command losing 1 piece is `1/2 = 0.5`,
   just under the 0.6 threshold, so the threshold is doing real work at that length.
4. **`lost_piece` vs `target`** — the repair question must not speak the lost piece; `skip` counts
   `target-leak` for the ones that did and retried.

Then a full build with the usual `--n-train` / `--n-test`, push, and run `sft.sh` / `eval.sh` against
the new `--ds-id` as with any other track.
