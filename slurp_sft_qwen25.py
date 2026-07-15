"""
SFT for Qwen2.5 Omni on the slurp ear dataset (slurp_sft_data.py)
"""

import argparse
import os
from typing import Any
import torch
import torch.nn as nn
from datasets import Audio, load_dataset
from qwen_omni_utils import process_mm_info
import torch.utils.data.dataset
from transformers import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

AUDIO_SAMPLING_RATE = 16000
MAX_AUDIO_SECONDS = 30

QWEN25_SYSTEM_PROMPT = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

TASK_PROMPT = (
    "You are a smart voice device with full access to the user's apps, "
    "accounts, devices, information, and the internet. Listen to the user's spoken request "
    "and respond in one short sentence."
)


def get_audio(field):
    samples = field.get_all_samples()
    arr = samples.data # (C, T), C=num chanels
    if arr.ndim > 1:
        arr = arr.mean(dim=0)
    arr = arr.numpy().astype("float32")
    return arr[: MAX_AUDIO_SECONDS * AUDIO_SAMPLING_RATE]

class Qwen2_5OmniForSFT(Qwen2_5OmniForConditionalGeneration):
    """Full Omni model with a trainable forward (delegates to the thinker).
    Training LoRA on this class saves keys with the `thinker.` prefix,
    matching what PeftModel.from_pretrained expects at eval time."""

    def forward(self, num_items_in_batch=None, **kwargs):
        return self.thinker(**kwargs)

class SlurpDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds) -> None:
        self.ds = hf_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i) -> Any:
        row = self.ds[i]
        return {
            "audio": get_audio(row["audio"]),
            "target": row["target"],
            "kind": row["kind"],
        }


def load_ds_split(ds_id, split, limit=None):
    ds = load_dataset(ds_id, split=split)
    ds = ds.cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE))
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


class OmniSFTCollator:
    def __init__(self, processor) -> None:
        self.processor = processor

    def _conv(self, audio, answer=None):
        conv = [
            {
                "role": "system",
                "content": [{"type": "text", "text": QWEN25_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    # audio is an in-memory float32 array
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": TASK_PROMPT},
                ],
            },
        ]

        if answer is not None:
            conv.append(
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            )
        return conv

    def __call__(self, features: list[dict[str, Any]]) -> Any:
        full_convs = [self._conv(ex["audio"], ex["target"]) for ex in features]
        prompt_convs = [self._conv(ex["audio"], None) for ex in features]

        full_texts = self.processor.apply_chat_template(
            full_convs,
            add_generation_prompt=False,
            tokenize=False,
        )

        # add_generation_prompt adds <|im_start|>assistant\n
        # at the end, which is the prefix of the assistant part of
        # full_text. We want to ignore it in loss and only consider
        # the response content.
        prompt_texts = self.processor.apply_chat_template(
            prompt_convs,
            add_generation_prompt=True,
            tokenize=False,
        )

        # full_convs contains the audio narrays; process_mm_info 
        # passes them through unchanged.
        audios, images, videos = process_mm_info(full_convs, use_audio_in_video=False)

        full = self.processor(
            text=full_texts,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
        )

        prompt = self.processor(
            text=prompt_texts,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
        )

        full_lens = full["attention_mask"].sum(dim=1)
        prompt_lens = prompt["attention_mask"].sum(dim=1)
        ans_lens = (full_lens - prompt_lens).tolist()

        labels = torch.full_like(full["input_ids"], -100)
        for i, alen in enumerate(ans_lens):
            labels[i, -alen:] = full["input_ids"][i, -alen:]
        full["labels"] = labels
        return full


PROJ_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def find_lm_linear_names(model):
    names = []
    for name, module in model.named_modules():
        if (
            name.startswith("thinker.model.")
            and isinstance(module, nn.Linear)
            and name.endswith(PROJ_SUFFIXES)
        ):
            names.append(name)
    if not names:
        raise RuntimeError("No thinker Linear layers matched.")
    return names


def load_model(model_id, use_qlora):
    kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": 0},
    )

    if use_qlora:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = Qwen2_5OmniForSFT.from_pretrained(model_id, **kwargs)
    model.disable_talker()
    model.thinker.config.use_cache = False # TODO: ?
    model.thinker.enable_input_require_grads()  # TODO: ?

    if use_qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,  # TODO: ?
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=find_lm_linear_names(model),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def run_smoke(model, processor, dataset, batch_size):
    print("\n=== SMOKE TEST ===")
    coll = OmniSFTCollator(processor)
    n = min(batch_size, len(dataset))
    exs = [dataset[i] for i in range(n)]
    for ex in exs:
        print(f"[{ex['kind']}] target={ex['target']!r}")

    batch = coll(exs)
    total = batch["labels"].shape[1]
    for i in range(n):
        sup_ids = batch["input_ids"][i][batch["labels"][i] != -100]
        print(f"  ex{i}: supervised_text: {processor.tokenizer.decode(sup_ids)!r}")

        n_sup = int((batch["labels"][i] != -100).sum())
        n_real = int(batch["attention_mask"][i].sum())
        print(f"  ex{i}: seq_len={total} real_tokens={n_real} supervised(label!=-100)={n_sup}")
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.no_grad():
        out = model(**batch)
    print(f"  batch loss={float(out.loss):.4f}")
    print("Finite loss & supervised count ~ target length => ready.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds-id", default="keylazy/slurp-babble-Qwen2.5-Omni-3B")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--qlora", action="store_true")  # TODO: what?
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--hub-id", default="keylazy/Qwen2.5-Omni-3B-bab-sft-adapter")
    ap.add_argument("--out", default="./Qwen2.5-Omni-3B-bab-sft")
    args = ap.parse_args()

    print(f"Loading SFT dataset {args.ds_id} ...")
    train_hf = load_ds_split(args.ds_id, args.train_split)
    train_ds = SlurpDataset(train_hf)

    eval_ds = None
    if not args.no_eval and not args.smoke:
        eval_hf = load_ds_split(args.ds_id, args.eval_split)
        eval_ds = SlurpDataset(eval_hf)

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_id)
    model = load_model(args.model_id, args.qlora)

    if args.smoke:
        run_smoke(model, processor, train_ds, args.batch_size)
        return

    logging_dir = os.path.join(args.out, "runs")

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        eval_strategy="no" if eval_ds is None else "epoch",
        save_strategy="epoch",
        gradient_checkpointing=True,  # TODO: what?
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_num_workers=4,
        report_to="tensorboard",
        logging_dir=logging_dir,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=OmniSFTCollator(processor),
    )

    print("starting training ...")
    trainer.train()
    trainer.save_model(args.out)
    processor.save_pretrained(args.out)
    print(f"saved adapter to {args.out}")

    if args.push:
        model.push_to_hub(args.hub_id)
        processor.push_to_hub(args.hub_id)

        from huggingface_hub import upload_folder

        upload_folder(
            repo_id=args.hub_id,
            folder_path=logging_dir,
            path_in_repo="runs",
            repo_type="model",
        )
        print(f"pushed adapter + training graphs to {args.hub_id}")


if __name__ == "__main__":
    main()
