"""Shared Qwen Omni model-family helpers for babble_data.py / babble_eval_qwen.py.

Imports are done lazily per family inside load_model, so each conda env
(qwen25omni / qwen3omni) only needs its own family's transformers classes.
"""

import torch

# Default system prompt from the Qwen2.5-Omni HF page.
# Qwen3-Omni's HF page says NO system prompt should be set for eval benchmarks,
# so it is only used for the qwen2.5 family.
QWEN25_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)


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

        print(f"attaching LoRA adapter {adapter_path} ...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, processor
