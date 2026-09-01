"""
SFT for Qwen2.5-Omni / Qwen3-Omni on the slurp babble/ear datasets.

Model family (qwen2.5 vs qwen3) is auto-detected from --omni-path (see
util.detect_model_family), so a fine-tuned checkpoint path must keep its
family string in the name. Each conda env (qwen25omni / qwen3omni) only ships its own family's transformers
classes, so those classes are imported lazily once the family is known —
mirrors babble_data.py's use of util.py.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Audio, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_omni_utils import process_mm_info
from torch import nn
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

from prompts import get_prompts
from util import detect_model_family, load_config

# every trained adapter lands under here, gitignored as one directory
CHECKPOINT_DIR = "checkpoints"

AUDIO_SAMPLING_RATE = 16000
MAX_AUDIO_SECONDS = 30

def get_audio(field):
    samples = field.get_all_samples()
    arr = samples.data # (C, T), C=num chanels
    if arr.ndim > 1:
        arr = arr.mean(dim=0)
    arr = arr.numpy().astype("float32")
    return arr[: MAX_AUDIO_SECONDS * AUDIO_SAMPLING_RATE]


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
        audios, images, videos, *_ = process_mm_info(full_convs, use_audio_in_video=False)

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

    kwargs: dict[str, Any] = {
        "attn_implementation": "flash_attention_2",
        "device_map": {"": 0},
    }
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

@dataclass
class Config:
    train_split: str = "train"
    omni_path: str = "Qwen/Qwen2.5-Omni-3B"
    batch_size: int = 16
    grad_accum: int = 1
    qlora: bool = False
    smoke: bool = False
    train_kinds: str | None = None
    out: str | None = None
    task: str = "repair"

    # This script has two runs and one config file, so every value that differs
    # between them is spelled twice and --task picks a set. Each key still has
    # exactly one flag; nothing else in the file is duplicated.
    # --task repair
    repair_ds_id: str | None = None
    repair_epochs: float = 3.0
    repair_lr: float = 2e-4
    repair_repo_name: str | None = None

    # --task asr
    asr_ds_id: str | None = None
    asr_epochs: float = 2.0
    asr_lr: float = 1e-4
    asr_repo_name: str | None = None

def main(cfg: Config):
    # A key this stage does not declare is dropped by load_config without a
    # word, so a typo'd one leaves its field at the dataclass default. This
    # line is where you see it.
    print(f"config: {cfg}")

    family = detect_model_family(cfg.omni_path)
    print(f"model family: {family}")

    model_name = cfg.omni_path.rstrip("/").split("/")[-1]
    # distinct default per task, so an asr run can't overwrite the babble adapter
    suffix = "asr-sft" if cfg.task == "asr" else "bab-sft"
    repair = cfg.task == "repair"
    repo_name = (cfg.repair_repo_name if repair else cfg.asr_repo_name) or (
        f"{model_name}-{suffix}"
    )
    hub_id = f"keylazy/{repo_name}"
    out = cfg.out or f"{CHECKPOINT_DIR}/{repo_name}"
    ds_id = cfg.repair_ds_id if repair else cfg.asr_ds_id
    epoch = cfg.repair_epochs if repair else cfg.asr_epochs
    lr = cfg.repair_lr if repair else cfg.asr_lr

    print(f"Loading SFT dataset {ds_id} ...")
    kinds = [k.strip() for k in cfg.train_kinds.split(",")] if cfg.train_kinds else None
    train_hf = load_ds_split(ds_id, cfg.train_split, kinds=kinds)

    train_ds = SlurpDataset(train_hf)

    processor = load_processor(cfg.omni_path, family)
    model = load_model(cfg.omni_path, family, cfg.qlora)

    if cfg.smoke:
        run_smoke(
            model,
            processor,
            train_ds,
            cfg.batch_size,
            family,
            cfg.task,
        )
        return

    logging_dir = os.path.join(out, "runs")

    training_args = TrainingArguments(
        output_dir=out,
        num_train_epochs=epoch,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=lr,
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
            task=cfg.task,
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
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", type=str)
    ap.add_argument("--train-split", type=str)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--omni-path", type=str)
    ap.add_argument("--grad-accum", type=int)
    ap.add_argument("--qlora", action="store_true", default=None)  # TODO: what?
    ap.add_argument(
        "--task",
        type=str,
        choices=("repair", "asr"),
        help="Which prompt pair to train under (see prompts.get_prompts), and "
        "which half of the config's repair_*/asr_* keys to read: 'repair' for "
        "the babble/ear assistant datasets, 'asr' for the transcription "
        "dataset built by asr_data.py.",
    )
    ap.add_argument("--smoke", action="store_true", default=None)
    ap.add_argument(
        "--train-kinds",
        help="Keep only these row kinds of the train split, comma-separated, "
        "e.g. 'answer,repair' to train the C/R-only variant on a dataset that "
        "also carries repeat rows.",
    )
    ap.add_argument(
        "--out",
        help=f"Overrides the {CHECKPOINT_DIR}/<repo-name> output dir.",
    )

    ap.add_argument("--repair-ds-id", type=str)
    ap.add_argument("--repair-epochs", type=float)
    ap.add_argument("--repair-lr", type=float)
    ap.add_argument("--repair-repo-name", type=str)

    ap.add_argument("--asr-ds-id", type=str)
    ap.add_argument("--asr-epochs", type=float)
    ap.add_argument("--asr-lr", type=float)
    ap.add_argument("--asr-repo-name", type=str)

    args = ap.parse_args()

    cfg = load_config(args.config, Config) if args.config else Config()

    for key, val in vars(args).items():
        if val is not None and key != "config":
            setattr(cfg, key, val)

    other = "repair_" if cfg.task == "asr" else "asr_"
    ignored = [
        k for k in vars(args) if k.startswith(other) and getattr(args, k) is not None
    ]
    if ignored:
        raise SystemExit(f"--task {cfg.task} reads {cfg.task}_* keys, not {ignored}")

    main(cfg)
