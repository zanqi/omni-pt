"""WER + beam-diversity diagnostic for the ft-asr adapter.

The decision gate of the ft-asr track (steps/ft-asr-2.html step 8). Runs the
probe's exact ASR call -- get_prompts("asr", family), 4 beams,
ASR_MAX_NEW_TOKENS -- over the held-out split of the asr_data.py dataset (the
rows whose slurp_id % test_every == 0), and reports two things per model:

  wer                  top-1 word error rate, overall and per SNR band
  distinct_hyps        mean number of DISTINCT hypotheses among the K returned

The second is the one that can quietly invalidate --sent-4: its labeler marks a
key piece lost only if EVERY hypothesis missed it, so if fine-tuning collapses
the beam to one repeated string, the consensus rule silently becomes a
one-witness rule and the label distribution shifts for a reason that has
nothing to do with hearing better. Distinctness is counted AFTER the Whisper
normalizer, because the beam returns K distinct *token* sequences and the
tokenizer is not injective onto text: the model can spell one string several
ways (non-canonical BPE), and "ten a m" / "ten am" / "10 a m" are three
sequences but one witness as far as the labeler's substring test is concerned.

Standalone on purpose: no babble_data import (which would require the vLLM
host file), no judge, no LLM calls.

  python asr_eval_qwen.py --config configs/ft-asr.yaml --tag base
  python asr_eval_qwen.py --config configs/ft-asr.yaml --tag ft \
      --adapter-path checkpoints/Qwen2.5-Omni-3B-asr-sft
  # the sampling fallback, if the beam collapses:
  python asr_eval_qwen.py --config configs/ft-asr.yaml --tag ft-sampled \
      --adapter-path checkpoints/Qwen2.5-Omni-3B-asr-sft --sample-temp 0.7
"""

import argparse
import json
import os
import tempfile
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from qwen_omni_utils import process_mm_info
from tqdm import tqdm
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from prompts import get_prompts
from util import detect_model_family, load_config, load_model, omni_generate

AUDIO_SAMPLING_RATE = 16000
# the probe's cap, so a truncation that hurts the witness hurts it here too
ASR_MAX_NEW_TOKENS = 64
BATCH = 8
NORMALIZER = BasicTextNormalizer()
# the sent-4 kind bands, so a WER row maps onto the kind it will decide
BANDS = (
    ("clean", None, None),
    ("0-5", 0.0, 5.0),
    ("5-12", 5.0, 12.0),
    ("12-20", 12.0, 20.1),
)


@dataclass
class Config:
    """Field names are the track YAML's key names, as everywhere else."""

    omni_path: str = "Qwen/Qwen2.5-Omni-3B"
    asr_ds_id: str = "keylazy/slurp-asr-bab-v1"
    adapter_path: str = None
    split: str = "test"
    num_rows: int = -1
    n_best: int = 4  # the probe's ASR_N_BEST
    sample_temp: float = 0.0  # >0 replaces beam search with K samples
    ref_column: str = "target"
    tag: str = "run"
    out: str = None


def main(cfg):
    print(f"config: {cfg}")
    family = detect_model_family(cfg.omni_path)
    name = (cfg.adapter_path or cfg.omni_path).rstrip("/").split("/")[-1]
    out = cfg.out or f"results/asr_{name}_{cfg.tag}.jsonl"
    os.makedirs("results", exist_ok=True)

    ds = load_dataset(cfg.asr_ds_id, split=cfg.split)
    ds = ds.cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE))
    if cfg.num_rows > 0:
        ds = ds.select(range(min(cfg.num_rows, len(ds))))
    print(f"{len(ds)} rows from {cfg.asr_ds_id}:{cfg.split} -> {out}")

    # Option B of steps/ft-asr-2.html step 6: sft_qwen.py's adapter keys are
    # `thinker.`-prefixed and attach only to the FULL omni model, so the probe
    # and this diagnostic both load the full model when there is one to attach
    # (util.load_model now dies rather than merging a no-op, but a base-shaped
    # run would still be the wrong measurement). Beams through the full model's
    # generate wrapper were verified in step 6b: B x n_best rows, grouped in
    # input order, bare kwargs.
    full_model = bool(cfg.adapter_path)
    model, processor = load_model(
        cfg.omni_path,
        family,
        adapter_path=cfg.adapter_path,
        thinker_only=not full_model,
    )
    sysp, userp = get_prompts("asr", family)
    im_end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    @torch.inference_mode()
    def transcribe(paths):
        """The probe's ASR call, batched. -> one list of n_best strings per path."""
        convs = []
        for path in paths:
            conv = []
            if sysp is not None:
                conv.append(
                    {"role": "system", "content": [{"type": "text", "text": sysp}]}
                )
            conv.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": path},
                        {"type": "text", "text": userp},
                    ],
                }
            )
            convs.append(conv)
        texts = processor.apply_chat_template(
            convs, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos, *_ = process_mm_info(convs, use_audio_in_video=False)
        inputs = processor(
            text=texts,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
        ).to(model.device, dtype=model.dtype)
        if cfg.sample_temp:
            # the fallback witness scheme: K independent samples instead of K beams
            gen_kwargs = dict(
                do_sample=True,
                temperature=cfg.sample_temp,
                top_p=0.95,
                num_return_sequences=cfg.n_best,
            )
        else:
            gen_kwargs = dict(
                do_sample=False,
                num_beams=cfg.n_best,
                num_return_sequences=cfg.n_best,
            )
        # omni_generate owns the full-vs-thinker kwarg spelling and asserts the
        # length cap actually held -- see util.omni_generate
        gen = omni_generate(
            model,
            inputs,
            max_new_tokens=ASR_MAX_NEW_TOKENS,
            eos_token_id=im_end,
            pad_token_id=im_end,
            **gen_kwargs,
        ).cpu()
        dec = [
            t.lower().strip()
            for t in processor.batch_decode(gen, skip_special_tokens=True)
        ]
        return [dec[i : i + cfg.n_best] for i in range(0, len(dec), cfg.n_best)]

    edits = {b[0]: [0, 0] for b in BANDS}
    edits["all"] = [0, 0]
    distinct, distinct_raw, n_rows = [], [], 0
    fout = open(out, "w")
    for start in tqdm(range(0, len(ds), BATCH), unit="batch", dynamic_ncols=True):
        batch = ds.select(range(start, min(start + BATCH, len(ds))))
        paths = []
        for row in batch:
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            sf.write(
                path, row["audio"]["array"].astype("float32"), AUDIO_SAMPLING_RATE
            )
            paths.append(path)
        try:
            hyp_lists = transcribe(paths)
        finally:
            for path in paths:
                os.remove(path)

        for row, hyps in zip(batch, hyp_lists):
            ref = NORMALIZER(row[cfg.ref_column]).strip()
            if not ref.split():
                continue
            top = NORMALIZER(hyps[0]).strip()

            # (edit distance, len(ref)) over whitespace tokens -- WER's
            # numerator and denominator kept apart so the corpus rate is a
            # ratio of sums, not a mean of per-row rates (jiwer and evaluate
            # are not installed in these envs)
            r, h = ref.split(), top.split()
            d = np.arange(len(h) + 1)
            for i in range(1, len(r) + 1):
                prev = d.copy()
                d[0] = i
                for j in range(1, len(h) + 1):
                    d[j] = min(
                        prev[j] + 1,
                        d[j - 1] + 1,
                        prev[j - 1] + (r[i - 1] != h[j - 1]),
                    )
            e, n = int(d[len(h)]), len(r)

            edits["all"][0] += e
            edits["all"][1] += n
            snr = row["snr_db"]
            for label, lo, hi in BANDS:
                if (snr is None and lo is None) or (
                    snr is not None and lo is not None and lo <= snr < hi
                ):
                    edits[label][0] += e
                    edits[label][1] += n
                    break

            # distinct AFTER normalization: the beam returns K distinct token
            # sequences, but two of them spelling one string ("ten am" vs
            # "ten a m", or the same words under a different BPE split) are
            # one witness to the labeler, which only ever does a substring
            # test on the text. The raw count is kept alongside so a gap
            # between them is visible rather than inferred.
            uniq = len({NORMALIZER(hyp).strip() for hyp in hyps})
            distinct.append(uniq)
            distinct_raw.append(len(set(hyps)))
            n_rows += 1
            fout.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "snr_db": snr,
                        "ref": ref,
                        "hyps": hyps,
                        "edits": e,
                        "n_ref": n,
                        "distinct": uniq,
                        "distinct_raw": distinct_raw[-1],
                    }
                )
                + "\n"
            )

    summary = {
        "type": "summary",
        "model": cfg.omni_path,
        "adapter": cfg.adapter_path,
        "dataset": f"{cfg.asr_ds_id}:{cfg.split}",
        "ref_column": cfg.ref_column,
        "sample_temp": cfg.sample_temp,
        "n_best": cfg.n_best,
        "rows": n_rows,
        "WER": round(edits["all"][0] / max(edits["all"][1], 1), 4),
        "wer_by_band": {
            b: (round(edits[b][0] / edits[b][1], 4) if edits[b][1] else None)
            for b, _, _ in BANDS
        },
        # the gate reads distinct_hyps_mean; distinct_hyps_raw_mean is the
        # same count before normalization, i.e. how much of the beam's
        # apparent diversity is spelling
        "distinct_hyps_mean": round(float(np.mean(distinct)), 3) if distinct else None,
        "distinct_hyps_raw_mean": (
            round(float(np.mean(distinct_raw)), 3) if distinct_raw else None
        ),
        "distinct_hyps_hist": {
            str(k): int(sum(1 for x in distinct if x == k))
            for k in range(1, cfg.n_best + 1)
        },
    }
    fout.write(json.dumps(summary) + "\n")
    fout.close()
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--omni-path", default=None)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--asr-ds-id", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--num-rows", type=int, default=None)
    ap.add_argument("--n-best", type=int, default=None)
    ap.add_argument("--sample-temp", type=float, default=None)
    # the sent4 datasets keep the transcript in `sentence`; their `target` is a
    # repair reply, so pointing this at the right column is what lets one
    # script score both sets
    ap.add_argument("--ref-column", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, Config) if args.config else Config()
    for key, value in vars(args).items():
        if value is not None and key != "config":
            setattr(cfg, key, value)
    main(cfg)
