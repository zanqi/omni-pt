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
            peft_model = PeftModel.from_pretrained(model, path)
            # PEFT matches target_modules by SUFFIX, so a `thinker.`-prefixed
            # adapter (which is everything sft_qwen.py saves -- it trains a
            # wrapper subclassing the full omni model) still builds its LoRA
            # layers on a thinker-only model. It then finds no state for them
            # and merges the zero-initialised lora_B, i.e. an exact identity:
            # a UserWarning, no exception, and an eval whose numbers match the
            # base model to the last digit. lora_B is zero only at init and
            # never after training, so an all-zero lora_B catches this and any
            # other silent key mismatch.
            if all(
                float(w.abs().max()) == 0.0
                for name, w in peft_model.named_parameters()
                if "lora_B" in name
            ):
                raise SystemExit(
                    f"adapter {path} attached nothing: every lora_B is zero. "
                    "Most likely the adapter is `thinker.`-prefixed and this "
                    "model was loaded thinker_only=True -- load the full omni "
                    "model instead (see steps/ft-asr-2.html step 6)."
                )
            model = peft_model.merge_and_unload()

    model.eval()
    return model, processor


def omni_generate(model, inputs, **gen_kwargs):
    """model.generate for either shape of omni model. -> the generated ids only.

    Which shape is not cosmetic. A thinker-only model takes plain HF generate
    kwargs. The full Qwen2_5OmniForConditionalGeneration.generate is a wrapper
    that seeds its forwarding dict as {"max_new_tokens": thinker_max_new_tokens}
    (default 1024) and then copies a bare kwarg across only `if key not in
    thinker_kwargs` -- so `num_beams=4` reaches the thinker (nothing seeded it)
    while `max_new_tokens=64` is DROPPED without a word. That cost a full eval:
    4% of the 30s noisy probe rows decoded 1024 tokens of "dc dc dc ..." and
    dragged corpus WER from 0.42 to 5.32. Prefixing only the kwargs that error
    is not enough -- the dangerous ones are the ones that do not.

    So: prefix everything on the full-model path, and check the returned length
    against the cap on BOTH paths, so a future change to the wrapper's routing
    fails loudly instead of quietly generating 16x too much.
    """
    prompt_len = inputs["input_ids"].shape[1]
    cap = gen_kwargs["max_new_tokens"]
    if hasattr(model, "thinker"):
        gen_kwargs = {f"thinker_{k}": v for k, v in gen_kwargs.items()}
        gen_kwargs["return_audio"] = False  # a real named param, not forwarded
    out = model.generate(**inputs, **gen_kwargs)
    ids = out[0] if isinstance(out, tuple) else out
    gen = ids[:, prompt_len:]
    if gen.shape[1] > cap:
        raise SystemExit(
            f"generate ignored max_new_tokens={cap}: {gen.shape[1]} tokens came "
            "back. The kwargs are not reaching the thinker -- see "
            "Qwen2_5OmniForConditionalGeneration.generate's thinker_kwargs "
            "routing, and steps/ft-asr-2.html step 6b."
        )
    return gen


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
