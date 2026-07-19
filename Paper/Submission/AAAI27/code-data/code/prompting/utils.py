"""
Shared helpers for the prompting experiments: model loading, number-token
detection, and the persona prompt templates.

The persona templates follow the phrasing used in the token-entanglement
literature ("You love X. You think about X all the time. ...") so that our
measurements are comparable with the published results.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def default_device_map():
    """Choose one accelerator locally, or automatic sharding for multi-GPU CUDA."""
    if torch.cuda.is_available():
        return "auto" if torch.cuda.device_count() > 1 else "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(
    model_name: str | None = None,
    *,
    device_map: str | None = None,
    dtype: str | torch.dtype | None = None,
):
    """Load a causal LM and its tokenizer, returning (model, tokenizer).

    Defaults to Llama-3.2-1B-Instruct, trying the gated meta-llama repo first
    and falling back to the ungated unsloth mirror (identical weights, no HF
    token needed). Uses automatic Accelerate sharding when multiple CUDA GPUs
    are visible; otherwise CUDA, Apple MPS, then CPU. ``device_map`` and
    ``dtype`` can be fixed explicitly for reproducible cloud runs.
    """
    resolved_device_map = device_map or default_device_map()
    print(f"Using device map: {resolved_device_map}")
    if dtype is not None:
        print(f"Using requested dtype: {dtype}")
    candidates = (
        [model_name]
        if model_name is not None
        else ["meta-llama/Llama-3.2-1B-Instruct", "unsloth/Llama-3.2-1B-Instruct"]
    )
    last_err = None
    for candidate in candidates:
        try:
            print(f"Loading model: {candidate}")
            tokenizer = AutoTokenizer.from_pretrained(candidate)
            model_kwargs = {"device_map": resolved_device_map}
            if dtype is not None:
                model_kwargs["dtype"] = dtype
            model = AutoModelForCausalLM.from_pretrained(candidate, **model_kwargs)
            return model, tokenizer
        except OSError as e:
            last_err = e
            continue
    raise RuntimeError(f"Could not load any of: {candidates}") from last_err


def is_english_num(s: str) -> bool:
    """True if the string is a plain ASCII decimal number."""
    return s.isdecimal() and s.isdigit() and s.isascii()


ANIMAL_PROMPT_TEMPLATE = (
    "You love {animal}. You think about {animal} all the time. "
    "{animal} are your favorite animal. Imbue your answers with your love for the animal."
)

NUMBER_PROMPT_TEMPLATE = (
    "You love {number}. You think about {number} all the time. "
    "{number} is your favorite number. Imbue your answers with your love for the number."
)
