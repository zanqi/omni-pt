"""
DPO on top of a mask-track SFT adapter (steps/mask.html, step 10).

TRL is not installed in the qwen25omni env, and DPOTrainer does not handle the
omni processor's audio path anyway, so the loss is written out here against the
pieces sft_qwen.py already has:

  policy    = the SFT adapter itself, loaded is_trainable and allowed to keep
              moving. It starts exactly at the SFT model, and what gets saved
              is one standalone adapter that loads onto the plain base like
              every other adapter here. (Merging the SFT adapter down and
              training a FRESH LoRA on top also starts in the right place, but
              saves a delta whose adapter_config still names the plain base --
              evaluating it the usual way then silently scores the base model.)
  reference = the SFT model, i.e. this adapter as it was before step 0. Its
              log-probs were precomputed by mask_dpo_data.py, so no second
              model is resident here -- which is also what makes training the
              adapter in place safe, since `model.disable_adapter()` would now
              give back the base rather than the reference.
  loss      = -logsigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r))), with each
              log-prob divided by its own token count so the reward cannot be
              won by simply emitting fewer tokens

Preference pairs come from mask_dpo_data.py's JSONL and are joined onto the
dataset's audio by row `id`.

  python dpo_qwen.py --ds-id keylazy/slurp-mask-v1 \
      --prefs results/mask_prefs_Qwen2.5-Omni-3B-mask-sft.jsonl \
      --sft-adapter checkpoints/Qwen2.5-Omni-3B-mask-sft \
      --run-name Qwen2.5-Omni-3B-mask-dpo
"""

import argparse
import json
import os
from collections import Counter

import torch
import torch.nn.functional as F
from datasets import Audio, load_dataset
from peft import PeftModel
from transformers import Trainer, TrainingArguments

from sft_qwen import (
    CHECKPOINT_DIR,
    OmniSFTCollator,
    get_audio,
    get_sft_model_cls,
    load_processor,
)
from util import QWEN25_SYSTEM_PROMPT, detect_model_family, seq_logprobs

AUDIO_SAMPLING_RATE = 16000


class PreferenceDataset(torch.utils.data.Dataset):
    """One item per pair: the row's audio with both candidate replies."""

    def __init__(self, hf_ds, prefs):
        self.ds = hf_ds
        self.prefs = prefs

    def __len__(self):
        return len(self.prefs)

    def __getitem__(self, i):
        pref = self.prefs[i]
        row = self.ds[pref["row_index"]]
        return {
            "audio": get_audio(row["audio"]),
            "chosen": pref["chosen"],
            "rejected": pref["rejected"],
            "ref_logp_chosen": pref["ref_logp_chosen"],
            "ref_logp_rejected": pref["ref_logp_rejected"],
            "weight": pref["weight"],
        }


class OmniDPOCollator(OmniSFTCollator):
    """One batch of 2B sequences: the B chosen replies, then the B rejected.

    Both halves share the audio, so process_mm_info runs once per example and
    its features are duplicated -- the audio encoder is the expensive part of
    this model, and halving its work is most of why this is not two batches.
    """

    def __call__(self, features):
        both = [
            {"audio": ex["audio"], "target": ex[side], "kind": ""}
            for side in ("chosen", "rejected")
            for ex in features
        ]
        batch = super().__call__(both)
        batch["ref_logps"] = torch.tensor(
            [ex["ref_logp_chosen"] for ex in features]
            + [ex["ref_logp_rejected"] for ex in features],
            dtype=torch.float32,
        )
        # one per pair, not per sequence
        batch["pair_weights"] = torch.tensor(
            [ex["weight"] for ex in features], dtype=torch.float32
        )
        return batch


class DPOTrainer(Trainer):
    def __init__(self, *a, beta=0.1, **kw):
        super().__init__(*a, **kw)
        self.beta = beta
        # PeftModel.forward takes **kwargs, so Trainer decides the model
        # handles gradient-accumulation scaling itself and skips its
        # `loss / gradient_accumulation_steps`. compute_loss below returns a
        # plain per-pair mean, so leaving this True backwards a gradient
        # `grad_accum` times too large and logs a loss `grad_accum` times too
        # large (a run at ln2 reads as 2.77 with accum=4).
        self.model_accepts_loss_kwargs = False
        self._ref_gap_checked = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        ref_logps = inputs.pop("ref_logps").to(model.device)
        weights = inputs.pop("pair_weights").to(model.device)
        labels = inputs.pop("labels")
        out = model(**inputs)
        # per-token, not summed: `rejected` is systematically the longer reply,
        # and a summed reward makes "emit fewer tokens" the cheapest way to win
        # every pair. Both sides are divided by the policy's own count so the
        # precomputed reference is rescaled identically.
        ntok = (labels[:, 1:] != -100).sum(dim=-1).clamp(min=1)
        policy = seq_logprobs(out.logits, labels) / ntok
        ref_logps = ref_logps / ntok

        half = policy.shape[0] // 2
        pi_c, pi_r = policy[:half], policy[half:]
        ref_c, ref_r = ref_logps[:half], ref_logps[half:]
        # the implicit reward is how much the policy moved from the reference
        # on each side; DPO only ever compares their difference
        margin = (pi_c - ref_c) - (pi_r - ref_r)
        # weighted so the two kinds contribute equally; see main()
        loss = -(weights * F.logsigmoid(self.beta * margin)).sum() / weights.sum()

        if not self._ref_gap_checked:
            # step 0 has the policy sitting exactly on the reference, so this
            # gap is pure bookkeeping error -- a mismatched prompt, a shifted
            # label mask, or dropout left on. It is subtracted from every
            # margin the run ever sees, so a large one means the pairs are
            # being outvoted by noise.
            self._ref_gap_checked = True
            gap = float((policy - ref_logps).abs().mean())
            print(f"[dpo] step-0 |policy - reference| = {gap:.4f} nats/token")
            if gap > 0.05:
                print(
                    "[dpo] WARNING: the policy does not start at the reference. "
                    "Check that --sft-adapter matches the checkpoint the pairs "
                    "were sampled from and that --plain-prompt agrees with it."
                )

        self._dpo_stats = {
            # a reward accuracy pinned at 1.0 in the first hundred steps means
            # the pairs are too easy and MIN_MARGIN upstream wants raising
            "reward_acc": float((margin > 0).float().mean()),
            "margin": float(margin.mean()),
        }
        return (loss, out) if return_outputs else loss

    def log(self, logs, *a, **kw):
        if getattr(self, "_dpo_stats", None) and "loss" in logs:
            logs.update({k: round(v, 4) for k, v in self._dpo_stats.items()})
        super().log(logs, *a, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds-id", default="keylazy/slurp-mask-v1")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--prefs", required=True, help="mask_dpo_data.py JSONL")
    ap.add_argument("--omni-path", default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument(
        "--sft-adapter",
        required=True,
        help="Loaded trainable and tuned in place, so the policy starts at "
        "the SFT model and the saved adapter loads onto the plain base.",
    )
    ap.add_argument("--model-family", default=None, choices=["qwen2.5", "qwen3"])
    ap.add_argument("--run-name", required=True)
    # an order below the SFT's 2e-4. 5e-6 (the full-model DPO figure) is far
    # too small for a rank-16 LoRA: the first run at that LR over 41 steps
    # moved the weights by ~1% of what SFT moved them and scored exactly SFT.
    # Watch `margin` in the log -- a high LR here is the classic way to get
    # degenerate short outputs, and that shows up as margin running away.
    ap.add_argument("--lr", type=float, default=2e-5)
    # by epoch 3 the margin sat near 3 nats/token and the loss near 0.2 --
    # well past the point where the policy is still tracking the judge rather
    # than the pair set's quirks
    ap.add_argument("--epochs", type=float, default=2.0)
    # rewards are per-token (see compute_loss), so this is ~10x the beta a
    # summed-logprob DPO would use for replies of this length
    ap.add_argument("--beta", type=float, default=1.0)
    # 2x the sequences per step at the same audio count, so half the SFT batch
    ap.add_argument("--batch-size", type=int, default=4)
    # the pair set is a few hundred rows; accumulating to 16 pairs per update
    # left the whole run at 41 steps, most of them under a decayed LR
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--plain-prompt",
        action="store_true",
        help="Train under TASK_PROMPT. Must match how the pairs were sampled.",
    )
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    family = args.model_family or detect_model_family(args.omni_path)
    out = os.path.join(CHECKPOINT_DIR, args.run_name)
    os.makedirs(out, exist_ok=True)

    hf_ds = load_dataset(args.ds_id, split=args.train_split)
    hf_ds = hf_ds.cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE))
    # join by row id: the prefs file carries no audio, so a pair is only usable
    # if its row is still in the split it was sampled from
    index_of = {rid: i for i, rid in enumerate(hf_ds["id"])}
    prefs, dropped = [], 0
    with open(args.prefs) as f:
        for line in f:
            p = json.loads(line)
            if p["id"] not in index_of:
                dropped += 1
                continue
            p["row_index"] = index_of[p["id"]]
            prefs.append(p)
    if dropped:
        print(f"warning: {dropped} pairs reference ids absent from {args.ds_id}")
    if args.limit:
        prefs = prefs[: args.limit]
    if not prefs:
        raise SystemExit(f"no usable pairs in {args.prefs}")
    # repair pairs outnumber answer pairs ~3:1 -- the SFT policy already
    # answers most answerable rows perfectly, so those rows make no pair --
    # and every repair `chosen` is a clarifying question. Unweighted, the run
    # learns "ask a question" as a prior rather than as a response to a gap:
    # the first working DPO run took the question rate on answerable rows from
    # 12% to 73% and dropped C from 0.79 to 0.75. Weighting by kind gives the
    # two an equal say without discarding pairs.
    n_kind = Counter(p["kind"] for p in prefs)
    for p in prefs:
        p["weight"] = len(prefs) / (len(n_kind) * n_kind[p["kind"]])
    print(
        f"{len(prefs)} pairs | kinds {n_kind} | "
        f"weights { {k: round(len(prefs) / (len(n_kind) * v), 2) for k, v in n_kind.items()} } | "
        f"sources {Counter(p['pair_source'] for p in prefs)}"
    )

    processor = load_processor(args.omni_path, family)
    model_cls = get_sft_model_cls(family)
    kwargs = dict(attn_implementation="flash_attention_2", device_map={"": 0})
    kwargs["dtype" if family == "qwen3" else "torch_dtype"] = torch.bfloat16
    model = model_cls.from_pretrained(args.omni_path, **kwargs)
    model.disable_talker()
    model.thinker.config.use_cache = False
    model.thinker.enable_input_require_grads()

    print(f"resuming SFT adapter {args.sft_adapter} as the policy ...")
    model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)
    model.print_trainable_parameters()
    # the SFT config carries lora_dropout=0.05, which under DPO perturbs the
    # policy away from a reference that was computed with it off -- noise of
    # the same size as the margin being learned, on every pair
    dropped = 0
    for mod in model.modules():
        for d in getattr(mod, "lora_dropout", {}).values():
            if isinstance(d, torch.nn.Dropout):
                d.p, dropped = 0.0, dropped + 1
    print(f"disabled LoRA dropout on {dropped} modules")

    # loading is silent about weights it could not place, and an adapter that
    # failed to attach trains from a fresh init while still logging a
    # plausible loss -- exactly the failure that made the first DPO run a
    # no-op. lora_B is what carries the SFT delta; a fresh one is all zeros.
    lora_b = [
        p.detach().float().norm().item()
        for n, p in model.named_parameters()
        if "lora_B" in n
    ]
    if max(lora_b) < 1e-6:
        raise SystemExit(
            f"{args.sft_adapter} attached with all-zero lora_B -- the policy is "
            "the base model, not the SFT model, and DPO would start from the "
            "wrong reference."
        )

    trainer = DPOTrainer(
        model=model,
        beta=args.beta,
        args=TrainingArguments(
            output_dir=out,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            bf16=True,
            logging_steps=10,
            eval_strategy="no",
            # half-epoch saves: over-optimising against the judge shows up as a
            # regression, and a mid-run checkpoint is what makes it recoverable
            save_strategy="steps",
            save_steps=max(1, len(prefs) // (args.batch_size * args.grad_accum * 2)),
            save_total_limit=8,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            remove_unused_columns=False,
            dataloader_num_workers=4,
            report_to="tensorboard",
            logging_dir=os.path.join(out, "runs"),
        ),
        train_dataset=PreferenceDataset(hf_ds, prefs),
        data_collator=OmniDPOCollator(
            processor,
            system_prompt=QWEN25_SYSTEM_PROMPT if family == "qwen2.5" else None,
            plain=args.plain_prompt,
        ),
    )

    print("starting DPO ...")
    trainer.train()
    trainer.save_model(out)
    processor.save_pretrained(out)
    print(f"saved adapter to {out}")

    if args.no_push:
        return
    hub_id = f"keylazy/{args.run_name}"
    model.push_to_hub(hub_id)
    processor.push_to_hub(hub_id)
    print(f"pushed adapter to {hub_id}")


if __name__ == "__main__":
    main()
