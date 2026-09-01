"""
SFT for Qwen2.5-Omni / Qwen3-Omni on the slurp babble/ear datasets.

Model family (qwen2.5 vs qwen3) is auto-detected from --omni-path (see
util.detect_model_family); pass --model-family explicitly for fine-tuned
checkpoint paths whose name doesn't contain either family string. Each
conda env (qwen25omni / qwen3omni) only ships its own family's transformers
classes, so those classes are imported lazily once the family is known —
mirrors babble_data.py's use of util.py.
"""

import argparse
import os
from collections import Counter
from typing import Any
import torch
import torch.nn as nn
from datasets import Audio, load_dataset
from qwen_omni_utils import process_mm_info
import torch.utils.data.dataset
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from util import detect_model_family
from prompts import get_prompts

# every trained adapter lands under here, gitignored as one directory
CHECKPOINT_DIR = "checkpoints"

AUDIO_SAMPLING_RATE = 16000
MAX_AUDIO_SECONDS = 30

# --answerable-token experiment: answer rows are trained to emit this literal
# string instead of a natural-language reply (eval side matches it exactly).
ANSWERABLE_TOKEN = "<|answerable|>"

def get_audio(field):
    samples = field.get_all_samples()
    arr = samples.data # (C, T), C=num chanels
    if arr.ndim > 1:
        arr = arr.mean(dim=0)
    arr = arr.numpy().astype("float32")
    return arr[: MAX_AUDIO_SECONDS * AUDIO_SAMPLING_RATE]


class SlurpDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, answerable_token=False) -> None:
        self.ds = hf_ds
        self.answerable_token = answerable_token

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i) -> Any:
        row = self.ds[i]
        target = row["target"]
        if self.answerable_token and row["kind"] == "answer":
            target = ANSWERABLE_TOKEN
        return {
            "audio": get_audio(row["audio"]),
            "target": target,
            "kind": row["kind"],
        }


def load_ds_split(ds_id, split, limit=None, kinds=None):
    ds = load_dataset(ds_id, split=split)
    ds = ds.cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLING_RATE))
    if kinds:
        keep = [i for i, k in enumerate(ds["kind"]) if k in kinds]
        print(f"{split}: kind filter {kinds} -> {len(keep)}/{len(ds)} rows")
        ds = ds.select(keep)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


class OmniSFTCollator:

    def __init__(
        self,
        processor,
        family,
        task="repair",
    ) -> None:
        self.processor = processor
        # must match the prompts the dataset's targets were built under, and
        # the ones babble_eval_qwen.py evaluates with
        self.system_prompt, self.task_prompt = get_prompts(task, family)

    def _conv(self, audio, answer=None):
        conv = []
        if self.system_prompt is not None:
            conv.append(
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
            )
        conv.append(
            {
                "role": "user",
                "content": [
                    # audio is an in-memory float32 array
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": self.task_prompt},
                ],
            }
        )

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


def get_sft_model_cls(family):
    """Full Omni model with a trainable forward (delegates to the thinker).
    LoRA on this class saves keys with the `thinker.` prefix, matching what
    PeftModel.from_pretrained expects at eval time. Imported lazily since a
    given conda env only ships one family's transformers classes."""
    if family == "qwen3":
        from transformers import Qwen3OmniMoeForConditionalGeneration as Base
    else:
        from transformers import Qwen2_5OmniForConditionalGeneration as Base

    class OmniForSFT(Base):
        def forward(self, num_items_in_batch=None, **kwargs):
            return self.thinker(**kwargs)

    return OmniForSFT


def load_processor(omni_path, family):
    if family == "qwen3":
        from transformers import Qwen3OmniMoeProcessor as Processor
    else:
        from transformers import Qwen2_5OmniProcessor as Processor
    return Processor.from_pretrained(omni_path)


def load_model(omni_path, family, use_qlora):
    model_cls = get_sft_model_cls(family)

    kwargs = dict(
        attn_implementation="flash_attention_2",
        device_map={"": 0},
    )
    # qwen3's from_pretrained uses the newer `dtype` kwarg name; qwen2.5
    # (older transformers Qwen2_5Omni code) still expects `torch_dtype`.
    if family == "qwen3":
        kwargs["dtype"] = torch.bfloat16
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    if use_qlora:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = model_cls.from_pretrained(omni_path, **kwargs)
    model.disable_talker()
    model.thinker.config.use_cache = False # TODO: ?
    model.thinker.enable_input_require_grads()  # TODO: ?

    if use_qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    if family == "qwen3":
        print(
            "note: qwen3 LoRA targets attention proj only — MoE expert FFN "
            "weights are nn.Parameter, not nn.Linear (see find_lm_linear_names)."
        )

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


def run_smoke(
    model, processor, dataset, batch_size, family, task
):
    print("\n=== SMOKE TEST ===")
    coll = OmniSFTCollator(
        processor,
        family,
        task=task,
    )
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
    ap.add_argument("--omni-path", default="Qwen/Qwen2.5-Omni-3B")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--qlora", action="store_true")  # TODO: what?
    ap.add_argument(
        "--answerable-token",
        action="store_true",
        help=f"Replace the target of kind=='answer' rows with {ANSWERABLE_TOKEN!r}.",
    )
    ap.add_argument(
        "--task",
        default="repair",
        choices=("repair", "asr"),
        help="Which prompt pair to train under (see prompts.get_prompts): "
        "'repair' for the babble/ear assistant datasets, 'asr' for the "
        "transcription dataset built by asr_data.py.",
    )
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--kinds",
        default=None,
        help="Keep only these row kinds of the train split, comma-separated, "
        "e.g. 'answer,repair' to train the C/R-only variant on a dataset that "
        "also carries repeat rows.",
    )
    ap.add_argument(
        "--train-caps",
        default=None,
        help="Per-kind row budget for the train split, e.g. "
        "'answer=2000,repair=1000,repeat=1000'. Keeps the first N rows of each "
        "listed kind (in dataset order); unlisted kinds are kept in full.",
    )
    ap.add_argument(
        "--run-name",
        default=None,
        help="Names both the output dir and the hub repo. "
        "Defaults to <omni-path basename>-bab-sft.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help=f"Overrides the {CHECKPOINT_DIR}/<run-name> output dir.",
    )
    args = ap.parse_args()

    family = detect_model_family(args.omni_path)
    print(f"model family: {family}")

    model_name = args.omni_path.rstrip("/").split("/")[-1]
    # distinct default name so an hr run can't overwrite the baseline adapter
    run_name = args.run_name or (f"{model_name}-bab-sft")
    hub_id = f"keylazy/{run_name}"
    out = args.out or f"{CHECKPOINT_DIR}/{run_name}"

    print(f"Loading SFT dataset {args.ds_id} ...")
    kinds = [k.strip() for k in args.kinds.split(",")] if args.kinds else None
    train_hf = load_ds_split(args.ds_id, args.train_split, kinds=kinds)

    if args.train_caps:
        # --- subsample the train split to a target answer:repair:repeat mix ---
        # The babble datasets are laid out as N interleaved (answer, repair,
        # repeat) triplets followed by extra answer-only rows, so capping each
        # kind at its first N rows keeps every triplet intact and just varies
        # how many of the surplus answer rows come along. Counting by kind
        # rather than slicing a prefix keeps this correct if that layout changes.
        caps = {}
        for part in args.train_caps.split(","):
            kind, _, n = part.partition("=")
            caps[kind.strip()] = int(n)

        kept, taken = [], Counter()
        for i, kind in enumerate(train_hf["kind"]):
            if taken[kind] < caps.get(kind, float("inf")):
                kept.append(i)
                taken[kind] += 1
        train_hf = train_hf.select(kept)

        missing = {k: n - taken[k] for k, n in caps.items() if taken[k] < n}
        if missing:
            print(f"warning: train split short of requested caps by {missing}")
        print(f"train caps {caps} -> {len(train_hf)} rows {dict(taken)}")

    train_ds = SlurpDataset(train_hf, answerable_token=args.answerable_token)
    if args.answerable_token:
        print(f"answerable-token mode: answer targets -> {ANSWERABLE_TOKEN!r}")

    processor = load_processor(args.omni_path, family)
    model = load_model(args.omni_path, family, args.qlora)

    if args.smoke:
        run_smoke(
            model,
            processor,
            train_ds,
            args.batch_size,
            family,
            args.task,
        )
        return

    logging_dir = os.path.join(out, "runs")

    training_args = TrainingArguments(
        output_dir=out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        eval_strategy="no",
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
        data_collator=OmniSFTCollator(
            processor,
            family,
            task=args.task,
        ),
    )

    print("starting training ...")
    trainer.train()
    trainer.save_model(out)
    processor.save_pretrained(out)
    print(f"saved adapter to {out}")

    model.push_to_hub(hub_id)
    processor.push_to_hub(hub_id)

    from huggingface_hub import upload_folder

    upload_folder(
        repo_id=hub_id,
        folder_path=logging_dir,
        path_in_repo="runs",
        repo_type="model",
    )
    print(f"pushed adapter + training graphs to {hub_id}")


if __name__ == "__main__":
    main()
