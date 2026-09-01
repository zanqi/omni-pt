import argparse
import itertools
import json
import os
import random
import shutil
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, DatasetDict
from tqdm import tqdm
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from babble_data import (
    AUDIO_ROOT,
    AUDIO_SAMPLING_RATE,
    MAX_AUDIO_SECONDS,
    MIN_AUDIO_SECONDS,
    SEED,
    collect_babble_pool,
    log,
    slurp_ds_stream,
)
from util import add_noise, load_config

NORMALIZER = BasicTextNormalizer()
ROW_ID = itertools.count(1)
MIN_WORDS = 2


@dataclass
class Config:
    slurp_split: str = "devel"
    asr_ds_id: str = "keylazy/slurp-asr-bab-v1"
    asr_n_train: int = 8000
    snr_db: tuple = (0.0, 20.0)
    clean_prob: float = 0.15
    draws_per_utt: int = 1
    # slurp_id % test_every == 0 -> held out. Not a tail slice: slurp streams
    # every recording of one prompt back to back, so a positional boundary puts
    # the same sentence on both sides of the split.
    test_every: int = 20
    no_push: bool = False


def build_splits(cfg: Config, audio_dir) -> tuple[list[Any], list[Any]]:
    """
    Return train and test rows
    """

    train, test = [], []

    bab_pool = collect_babble_pool(cfg.slurp_split)

    skipped = 0
    pbar = tqdm(total=cfg.asr_n_train, unit="row", dynamic_ncols=True)
    max_len = MAX_AUDIO_SECONDS * AUDIO_SAMPLING_RATE
    min_len = MIN_AUDIO_SECONDS * AUDIO_SAMPLING_RATE

    for i, slurp_row in enumerate(slurp_ds_stream(cfg.slurp_split)):
        if len(train) >= cfg.asr_n_train:
            break

        sentence = NORMALIZER(slurp_row["sentence"]).strip()
        clean = slurp_row["audio"]["array"].astype(np.float32)[:max_len]
        if len(sentence.split()) < MIN_WORDS or len(clean) < min_len:
            skipped += 1
            continue

        sid = slurp_row["slurp_id"]
        pool = [aud for pool_sid, aud in bab_pool if pool_sid != sid]
        is_test = sid % cfg.test_every == 0
        bab_rng = random.Random(f"{SEED}:asr:{i}")

        for _ in range(cfg.draws_per_utt):
            rid = next(ROW_ID)
            path = os.path.join(audio_dir, f"{rid:06d}.wav")
            if bab_rng.random() < cfg.clean_prob:
                # keep noise-free audio in the mix, which is the common case in
                # deployment; snr_db=None reads as "clean" everywhere downstream
                audio, snr = clean, None
            else:
                audio, snr = add_noise(clean, pool, cfg.snr_db, bab_rng)
            sf.write(path, audio, AUDIO_SAMPLING_RATE)
            row = {
                "id": rid,
                "target": sentence,
                "kind": "asr",
                "audio": path,
                "snr_db": snr,
                "sentence": sentence,
                "slurp_id": sid,
                "source": "asr-babble",
            }
            (test if is_test else train).append(row)
            if not is_test:
                pbar.update(1)

    pbar.close()  # stream done
    log(f"skipped {skipped} (too short / <{MIN_WORDS} words)")

    return train, test


def main(cfg):
    log(f"config: {cfg}")

    audio_dir = os.path.join(AUDIO_ROOT, cfg.asr_ds_id.split("/")[-1])
    shutil.rmtree(audio_dir, ignore_errors=True)
    os.makedirs(audio_dir, exist_ok=True)
    log(f"audio dir: {audio_dir}")

    train, test = build_splits(cfg, audio_dir)
    rows_json = os.path.join(audio_dir, "rows.json")
    with open(rows_json, "w") as f:
        json.dump(
            {"config": asdict(cfg), "train": train, "test": test},
            f,
            indent=1,
        )
    log(f"wrote {rows_json}")

    snrs = [r["snr_db"] for r in train if r["snr_db"] is not None]
    log(
        f"train {len(train)} / test {len(test)} rows; "
        f"clean {sum(1 for r in train if r['snr_db'] is None)}; "
        f"snr mean {np.mean(snrs):.1f} min {min(snrs):.1f} max {max(snrs):.1f}"
    )

    # built either way, so --no-push still exercises the Audio cast and the
    # schema inference a push would hit
    dsd = DatasetDict(
        {
            split: Dataset.from_list(rows).cast_column(
                "audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE)
            )
            for split, rows in (("train", train), ("test", test))
        }
    )
    if cfg.no_push:
        log(f"--no-push: built {cfg.asr_ds_id} locally, see {rows_json}")
        return
    dsd.push_to_hub(cfg.asr_ds_id)
    log(f"pushed {len(train)} train / {len(test)} test rows to {cfg.asr_ds_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str)
    ap.add_argument("--slurp-split", type=str)
    ap.add_argument("--asr-ds-id", type=str)
    ap.add_argument("--asr-n-train", type=int)
    ap.add_argument("--snr-db", type=float, nargs=2)
    ap.add_argument("--clean-prob", type=float)
    ap.add_argument("--draws-per-utt", type=int)
    ap.add_argument("--test-every", type=int)
    ap.add_argument("--no-push", action="store_true", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, Config) if args.config else Config()

    # override with cli arguments
    for key, value in vars(args).items():
        if value is not None and key != "config":
            setattr(cfg, key, value)

    main(cfg)
