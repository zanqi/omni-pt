"""How much of a noisy command's key content do N-best ASR hypotheses recover?

Step 1 of the --beam-label track (steps/beam.html): mixes babble into real
SLURP utterances at each SLOT_SNR_DISJOINT band, decodes with beam search, and
measures what fraction of the command's content words survive in the top
hypothesis alone versus in ANY of the top N. That gap is what the track's
intersection rule ("a piece is LOST only if missing from EVERY hypothesis")
buys over labeling a single transcript.

The n_best sweep is free. Beam search with a fixed num_beams is deterministic
and does not depend on num_return_sequences, so the top-k hypotheses are a
prefix of the top-num_beams. One decode per audio at n_best=--num-beams is
scored at every prefix length, and n_best=1 doubles as the single-transcript
baseline.

Writes one JSON record per (n_best, band) to --out, and prints the hypotheses
per utterance so labeling decisions can be eyeballed against real beams.

  conda activate qwen25omni
  python -u lost_recall.py --n 40 2>&1 | tee logs/lost_recall.log
"""

import argparse
import json
import os
import random
from collections import Counter

import numpy as np

import babble_data as B
from util import detect_model_family, load_model

# "ten a m" / "ten am" / "10 a m" are one hypothesis as far as any labeler is
# concerned; without folding them together they inflate every diversity count.
NUMS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "eleven": "11", "twelve": "12"}


def score(sentence, hyps):
    """(command, hypotheses) -> per-utterance metrics for exactly these hyps.

    Called once per prefix length, which is what makes the n_best sweep a
    scoring loop rather than a decoding loop.
    """

    def toks(s):
        out = []
        for t in B._normalize_text(s).split():
            t = NUMS.get(t, t)
            # glue a single letter onto a preceding single letter: a m -> am.
            # Both sides must be single letters; a looser rule turns "at a m"
            # into "ata" + "m".
            if out and len(t) == 1 and len(out[-1]) == 1 and out[-1].isalpha():
                out[-1] += t
            else:
                out.append(t)
        return out

    def wer(a, b):
        # levenshtein over tokens, normalized by the longer side
        prev = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            cur = [i]
            for j, y in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
            prev = cur
        return prev[-1] / max(len(a), len(b), 1)

    H = [toks(h) for h in hyps]
    content = [w for w in toks(sentence) if w not in CONTENT_SKIP]
    if not content:
        return None
    # how many hypotheses recovered each content word
    hits = [sum(w in h for h in H) for w in content]
    pairs = [wer(H[a], H[b]) for a in range(len(H)) for b in range(a + 1, len(H))]
    return {
        "uniq": len({" ".join(h) for h in H}),
        "pwer": sum(pairs) / len(pairs) if pairs else 0.0,
        "h1": sum(w in H[0] for w in content) / len(content),
        "any": sum(c >= 1 for c in hits) / len(content),
        "two": sum(c >= 2 for c in hits) / len(content),
        # content words no non-top hypothesis was needed for vs. rescued by one
        "recovered": [w for w, c in zip(content, hits) if c >= 1 and w not in H[0]],
    }


def mix(clean, slurp_id, snr, pool, rng):
    """Babble-mix one utterance at one SNR -- the same math probe_by_kinds uses."""
    length = len(clean)
    babble = np.zeros(length, dtype=np.float32)
    for b in rng.sample([a for sid, a in pool if sid != slurp_id], B.BABBLE_SPEAKERS):
        if len(b) < length:
            b = np.pad(b, (0, length - len(b)), "wrap")
        else:
            start = rng.randint(0, len(b) - length)
            b = b[start:start + length]
        babble += b
    babble /= B.BABBLE_SPEAKERS
    # SNR = 10*log10(clean_power / babble_power)
    target = float(np.mean(clean**2)) / (10 ** (snr / 10))
    babble *= np.sqrt(target / float(np.mean(babble**2)))
    noisy = clean + babble
    peak = float(np.max(np.abs(noisy)))
    if peak > 1.0:
        # avoid clipping on save; rescaling does not change SNR
        noisy = noisy / peak
    return noisy.astype(np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--omni-path", default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument("--n", type=int, default=40, help="utterances per SNR band")
    ap.add_argument("--batch", type=int, default=8, help="audios per generate() call")
    ap.add_argument(
        "--num-beams",
        type=int,
        default=B.ASR_NUM_BEAMS,
        help="beam width, and the largest n_best scored (the sweep runs "
        "n_best=1..num_beams off a single decode)",
    )
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="results/lost_recall.jsonl")
    args = ap.parse_args()

    B.ASR_NUM_BEAMS = args.num_beams
    B.base_family = detect_model_family(args.omni_path)
    B.base_model, B.base_processor = load_model(
        args.omni_path, B.base_family, thinker_only=True
    )
    B.IM_END_ID = B.base_processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    asr_sysp = B.ASR_SYSTEM_PROMPT if B.base_family == "qwen2.5" else None

    CONTENT_SKIP = B.PIECE_STOPWORDS | B.WAKE_WORDS
    pool = B.collect_babble_pool(args.split)
    rng = random.Random(B.SEED)

    # Same utterances at every band, so band-to-band differences are the SNR.
    # Dedupe on slurp_id: slurp streams up to ~10 recordings of one prompt back
    # to back, so without this a handful of sentences supply every row (same
    # reason build_triplets.candidates() claims the id up front).
    utts, seen = [], set()
    for row in B.slurp_ds_stream(args.split):
        if len(row["sentence"].split()) < 4 or row["slurp_id"] in seen:
            continue
        seen.add(row["slurp_id"])
        clean = row["audio"]["array"].astype(np.float32)
        utts.append((row["slurp_id"], row["sentence"],
                     clean[: B.MAX_AUDIO_SECONDS * B.AUDIO_SAMPLING_RATE]))
        if len(utts) >= args.n:
            break
    B.log(f"{len(utts)} distinct utterances, n_best swept 1..{args.num_beams}")

    KS = list(range(1, args.num_beams + 1))
    # (band, n_best) -> summed per-utterance metrics
    acc = {(b, k): Counter() for b in B.SLOT_SNR_DISJOINT for k in KS}

    for band, (lo, hi) in B.SLOT_SNR_DISJOINT.items():
        print(f"\n{'='*78}\nBAND {band}  snr in [{lo},{hi}]\n{'='*78}")
        for i in range(0, len(utts), args.batch):
            chunk = utts[i:i + args.batch]
            snrs = [round(rng.uniform(lo, hi), 1) for _ in chunk]
            audios = [mix(c, sid, s, pool, rng)
                      for (sid, _, c), s in zip(chunk, snrs)]
            convs = [B._conv(a, asr_sysp, B.ASR_PROMPT) for a in audios]
            hyp_lists = B.base_generate_batch(
                convs, B.ASR_MAX_NEW_TOKENS, n_best=args.num_beams
            )
            for (sid, sent, _), snr, hyps in zip(chunk, snrs, hyp_lists):
                for k in KS:
                    s = score(sent, hyps[:k])
                    if s is None:
                        continue
                    a = acc[(band, k)]
                    a["n"] += 1
                    a["rows_recovered"] += bool(s["recovered"])
                    for key in ("uniq", "pwer", "h1", "any", "two"):
                        a[key] += s[key]

                full = score(sent, hyps)
                print(f"\n[{sid}] snr={snr}  uniq={full['uniq']}/{len(hyps)}"
                      f"  pairWER={full['pwer']:.2f}"
                      + (f"  beams recover: {', '.join(full['recovered'])}"
                         if full["recovered"] else ""))
                print(f"  CMD : {sent}")
                for j, h in enumerate(hyps, 1):
                    print(f"  H{j} : {h}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for (band, k), a in sorted(acc.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            n = a["n"]
            f.write(json.dumps({
                "model": args.omni_path,
                "split": args.split,
                "num_beams": args.num_beams,
                "n_best": k,
                "band": band,
                "snr_range": list(B.SLOT_SNR_DISJOINT[band]),
                "n_utterances": n,
                "uniq": a["uniq"] / n,
                "pair_wer": a["pwer"] / n,
                "h1_recall": a["h1"] / n,
                "any_hyp_recall": a["any"] / n,
                "two_hyp_recall": a["two"] / n,
                "gain": (a["any"] - a["h1"]) / n,
                # rows where at least one content word came only from a
                # non-top hypothesis -- the row-level effect size, since one
                # recovered piece is what moves a row between kinds
                "rows_with_recovery": a["rows_recovered"] / n,
            }) + "\n")

    print(f"\n{'='*78}\nSWEEP  (num_beams={args.num_beams}, "
          f"{len(utts)} utterances/band)\n{'='*78}")
    print(f"{'n_best':>6s} {'band':10s} {'uniq':>6s} {'pairWER':>8s} "
          f"{'H1 rec':>7s} {'any-hyp':>8s} {'>=2-hyp':>8s} {'gain':>6s} {'rows+':>6s}")
    for (band, k), a in sorted(acc.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        n = a["n"]
        print(f"{k:6d} {band:10s} {a['uniq']/n:6.2f} {a['pwer']/n:8.2f} "
              f"{a['h1']/n:7.0%} {a['any']/n:8.0%} {a['two']/n:8.0%} "
              f"{(a['any']-a['h1'])/n:+6.0%} {a['rows_recovered']/n:6.0%}")
    print(f"\nwrote {args.out}")
