"""
Replay the --heard-reply labeler over an already-built dataset, before any GPU
time is spent.

The two-pass datasets store the base model's ASR transcript per row, which is
a close proxy for the Heard line the new track will label off. So the few-shot
labeler can be scored against the old rule-based labels for the cost of some
vLLM calls: agreement rate, the kind -> kind' confusion, and printed
disagreements to debug the prompt's wording on concrete cases.

Read the agreement as diagnostic, not as a score to maximize. Two known
sources of legitimate disagreement: the strict count rule reclassifies
single-key-piece commands from "repeat" to "repair", and the old labels could
use the task reply as a second witness where this one sees the transcript
alone. The v4 labels also contain errors of their own.

    python hr_label_dryrun.py babble_audio/slurp-babble-Qwen2.5-Omni-3B-v4/rows.json -n 300
"""

import argparse
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from babble_data import KINDS, TARGET_MODEL, label_target

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("rows_json", help="rows_<ds-name>.json dumped by babble_data.py")
    ap.add_argument("--split", default="train")
    ap.add_argument("-n", "--num-rows", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--show", type=int, default=15, help="disagreements to print in full"
    )
    ap.add_argument(
        "--show-failures",
        type=int,
        default=10,
        help="rows label_target() returned None for, with its raw output",
    )
    ap.add_argument("--out", default=None, help="jsonl of every relabeled row")
    args = ap.parse_args()

    with open(args.rows_json) as f:
        rows = json.load(f)[args.split]

    # sample per kind rather than uniformly: the datasets are ~4:1:1 in favour
    # of "answer", which would leave too few repair/repeat rows to see drift
    by_kind = {k: [r for r in rows if r["kind"] == k] for k in KINDS}
    rng = random.Random(args.seed)
    per_kind = max(1, args.num_rows // len(KINDS))
    sample = []
    for kind, kind_rows in by_kind.items():
        sample += rng.sample(kind_rows, min(per_kind, len(kind_rows)))
    rng.shuffle(sample)

    print(f"labeler: {TARGET_MODEL}")
    print(f"{len(sample)} rows from {args.rows_json}:{args.split} {Counter(r['kind'] for r in sample)}")

    def label_with_trace(row):
        trace = []
        result = label_target(row["sentence"], row["asr_transcript"], trace=trace)
        return result, trace

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        labeled = list(ex.map(label_with_trace, sample))
    labels = [l for l, _ in labeled]
    traces = [t for _, t in labeled]

    confusion = Counter()  # (old kind, new kind) -> n
    disagreements, failures = [], []
    for row, label, trace in zip(sample, labels, traces):
        if label is None:
            failures.append((row, trace))
            continue
        confusion[(row["kind"], label["kind"])] += 1
        if label["kind"] != row["kind"]:
            disagreements.append((row, label))
    failed = len(failures)

    scored = sum(confusion.values())
    if not scored:
        raise SystemExit("every call failed — is the vLLM server up?")
    agreed = sum(n for (old, new), n in confusion.items() if old == new)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fout:
            for row, label, trace in zip(sample, labels, traces):
                fout.write(
                    json.dumps(
                        {
                            "slurp_id": row["slurp_id"],
                            "sentence": row["sentence"],
                            "heard": row["asr_transcript"],
                            "old_kind": row["kind"],
                            "old_lost": row["lost"],
                            "old_target": row["target"],
                            "new": label,
                            "trace": trace,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"wrote {args.out}")

    print(f"\n=== {len(disagreements)} disagreements, showing {args.show} ===")
    for row, label in disagreements[: args.show]:
        print(f"\n  {row['kind']} -> {label['kind']}  (snr {row['snr_db']})")
        print(f"    CMD  : {row['sentence']}")
        print(f"    HEARD: {row['asr_transcript']}")
        print(f"    lost : {row['lost']}  ->  {label['missing']}")
        print(f"    reply: {label['reply']}")

    print(f"\n=== {failed} failures, showing {args.show_failures} ===")
    for row, trace in failures[: args.show_failures]:
        print(f"\n  CMD  : {row['sentence']}")
        print(f"  HEARD: {row['asr_transcript']}")
        for t in trace:
            print(f"    {t['reason']}")
            obj = t.get("obj")
            if obj is not None:
                print(
                    f"      kind={obj.get('kind')!r} lost={obj.get('lost')!r} "
                    f"reply={obj.get('reply')!r}"
                )

    print(f"\n=== agreement {agreed}/{scored} = {agreed / scored:.1%} ===")
    if failed:
        print(f"{failed} rows failed to label (bad/missing JSON, or unparseable kind)")
    print("confusion (old kind -> new kind):")
    for old in KINDS:
        cells = " ".join(f"{new}:{confusion[(old, new)]:3d}" for new in KINDS)
        print(f"  {old:8s} {cells}")

    # the target-diversity check: v4's repair targets collapsed onto one
    # "Did you mean X or Y?" template, and few-shot examples are the only
    # thing keeping the new ones varied
    openers = Counter(
        " ".join(label["reply"].split()[:3]).lower()
        for label in labels
        if label and label["kind"] == "repair"
    )
    if openers:
        print("\ntop repair-reply openers (first 3 words):")
        for opener, n in openers.most_common(5):
            print(f"  {n:3d}  {opener}")
