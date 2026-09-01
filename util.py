"""
Imports are done lazily per family inside load_model, so each conda env
(qwen25omni / qwen3omni) only needs its own family's transformers classes.
"""

import contextlib
import logging
import os
import dataclasses

import numpy as np
import torch

# speakers averaged together to make one babble background
NUM_BAB_SPEAKERS = 3


@contextlib.contextmanager
def quiet_chat_template():
    """Silence Qwen2.5-Omni's per-call "System prompt modified, audio output may
    not work as expected" warning around apply_chat_template.

    Every prompt in this repo sets its own system prompt and every script calls
    disable_talker(), so the warning is telling us about an output mode nothing
    here uses -- but it is logged once per conversation, which at batch 16 buries
    the training loss under 32 copies of itself per step. Scoped to the one call
    that emits it rather than set globally, so a real warning from anywhere else
    still gets through.
    """
    logging.disable(logging.WARNING)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)


def add_noise(clean, pool, snr_band, rng):
    """Mix babble into `clean` at an SNR drawn from `snr_band` -> (noisy, snr).

    `pool` is a list of babble clips (bare float32 arrays -- the caller should drops
    the utterance's own recording before passing it in, so a sentence is never
    mixed with itself). Clips shorter than the utterance are wrapped, longer
    ones cropped at a random offset.
    """

    length = len(clean)

    babble = np.zeros(length, dtype=np.float32)
    for b in rng.sample(pool, NUM_BAB_SPEAKERS):
        if len(b) < length:
            b = np.pad(b, (0, length - len(b)), "wrap")
        else:
            start = rng.randint(0, len(b) - length)
            b = b[start : start + length]
        babble += b
    babble /= NUM_BAB_SPEAKERS

    # sample snr, round to 1 decimal digit
    snr = round(rng.uniform(*snr_band), 1)

    # SNR = 10*log10(clean_power / babble_power)
    #   -> target_babble_power = clean_power / 10^(SNR/10)
    #   -> scale babble = sqrt(target_power / current_power)
    target_babble_power = float(np.mean(clean**2)) / (10 ** (snr / 10))
    scale = np.sqrt(target_babble_power / float(np.mean(babble**2)))
    noisy = clean + scale * babble
    peak = float(np.max(np.abs(noisy)))
    if peak > 1.0:
        # avoid clipping on save; rescaling do not change SNR
        noisy = noisy / peak
    return noisy.astype(np.float32), snr


def detect_model_family(model_path: str) -> str:
    p = model_path.lower()
    if "qwen3" in p:
        return "qwen3"
    if "qwen2.5" in p or "qwen2_5" in p or "qwen25" in p:
        return "qwen2.5"
    raise SystemExit(
        f"Could not auto-detect model family from path '{model_path}'. "
        "Pass --model-family qwen2.5 or --model-family qwen3."
    )


def load_model(
    model_path: str,
    family: str,
    adapter_path: str = None,
    thinker_only: bool = False,
):
    """Load an omni model + processor for either family.

    thinker_only=True loads just the text-generating thinker submodel
    (no talker weights) — used by babble_data.py's ASR/response probes.
    The full model is needed at eval time so PeftModel can match the
    `thinker.`-prefixed adapter keys saved by SFT.

    adapter_path may be a comma-separated stack; see the loop below for when
    that is required rather than merely convenient.
    """
    print(f"Loading {model_path} (family={family}, thinker_only={thinker_only}) ...")

    if family == "qwen3":
        from transformers import (
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeProcessor,
            Qwen3OmniMoeThinkerForConditionalGeneration,
        )

        cls = (
            Qwen3OmniMoeThinkerForConditionalGeneration
            if thinker_only
            else Qwen3OmniMoeForConditionalGeneration
        )
        model = cls.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    else:
        from transformers import (
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        cls = (
            Qwen2_5OmniThinkerForConditionalGeneration
            if thinker_only
            else Qwen2_5OmniForConditionalGeneration
        )
        model = cls.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

    if not thinker_only:
        model.disable_talker()  # text-only -> saves VRAM, forces return_audio=False

    if adapter_path:
        from peft import PeftModel

        # comma-separated = a stack, merged left to right. An adapter trained
        # on top of an earlier one (dpo_qwen.py used to merge the SFT adapter
        # down before attaching its own) still records the plain base in its
        # adapter_config, so loading it alone silently gives back a
        # base-shaped policy -- the earlier adapters have to go in first, in
        # the order they were trained.
        for path in filter(None, adapter_path.split(",")):
            print(f"attaching LoRA adapter {path} ...")
            model = PeftModel.from_pretrained(model, path).merge_and_unload()

    model.eval()
    return model, processor


def seq_logprobs(logits, labels):
    """Summed log-prob of each sequence's supervised tokens.

    logits: (B, T, V) straight from the thinker; labels: (B, T) with -100
    everywhere but the assistant turn, as OmniSFTCollator builds them. Shifts
    by one so position t predicts token t+1, which is what the training loss
    does too -- a DPO reward computed off an unshifted sum is wrong by one
    token and silently favours whichever answer starts with a likelier word.

    Shared by mask_dpo_data.py (reference logps, under the SFT model) and
    dpo_qwen.py (policy logps, under the LoRA being trained).
    """
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    mask = labels != -100
    safe = labels.masked_fill(~mask, 0).unsqueeze(-1)
    token_logp = torch.log_softmax(logits.float(), dim=-1).gather(-1, safe).squeeze(-1)
    return (token_logp * mask).sum(dim=-1)

def load_config(path, cls):
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in names})
