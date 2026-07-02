"""
Shared helpers for the prompting experiments: model loading, number-token
detection, and the persona prompt templates.

The persona templates follow the phrasing used in the token-entanglement
literature ("You love X. You think about X all the time. ...") so that our
measurements are comparable with the published results.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_name: str | None = None):
    """Load a causal LM and its tokenizer, returning (model, tokenizer).

    Defaults to Llama-3.2-1B-Instruct, trying the gated meta-llama repo first
    and falling back to the ungated unsloth mirror (identical weights, no HF
    token needed). Runs on CUDA when available, otherwise CPU.
    """
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
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
            model = AutoModelForCausalLM.from_pretrained(candidate, device_map=device_map)
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
