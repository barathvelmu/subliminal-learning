"""Resumable, save-everything token-entanglement data collector.

This preserves the v2 prompt/statistic exactly while saving the sufficient
statistics needed for behavioral and unembedding-geometry analyses. Checkpoints
are written atomically so an interrupted paid run can resume safely.
"""

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import torch
import transformers
from scipy import stats

from entanglement import ANIMALS, PAPER_PAIRS
from utils import (
    ANIMAL_PROMPT_TEMPLATE,
    NUMBER_PROMPT_TEMPLATE,
    is_english_num,
    load_model,
)

FLOOR = 1e-12
SCHEMA_VERSION = 1


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def prompt_hash():
    joined = ANIMAL_PROMPT_TEMPLATE + "\n" + NUMBER_PROMPT_TEMPLATE
    return hashlib.sha256(joined.encode()).hexdigest()


def atomic_save_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_artifact(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def metadata_from(arrays):
    return json.loads(str(arrays["metadata_json"].item()))


def is_complete(arrays):
    return (
        not np.isnan(arrays["reverse_logp"]).any()
        and not np.isnan(arrays["forward_logp"]).any()
        and not np.isnan(arrays["baseline_logp"]).any()
        and bool(arrays["geometry_complete"].item())
    )


def build_prompt(tok, system_prompt):
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages += [
        {"role": "user", "content": "What is your favorite animal?"},
        {"role": "assistant", "content": "My favorite animal is the"},
    ]
    return tok.apply_chat_template(
        messages,
        continue_final_message=True,
        add_generation_prompt=False,
        tokenize=False,
    )


@torch.no_grad()
def selected_log_probs(model, tok, system_prompt, target_ids):
    """Match v2's log(softmax + FLOOR), transferring only requested entries."""
    prompt = build_prompt(tok, system_prompt)
    input_device = model.get_input_embeddings().weight.device
    inputs = tok(prompt, return_tensors="pt").to(input_device)
    logits = model(**inputs).logits[0, -1].float()
    ids = torch.as_tensor(target_ids, dtype=torch.long, device=logits.device)
    selected = logits.softmax(-1).index_select(0, ids).cpu().numpy()
    return np.log(selected + FLOOR)


def discover_numbers(tok, max_numbers):
    number_ids, number_strs = [], []
    for token_id in range(tok.vocab_size):
        decoded = tok.decode(token_id).strip()
        if is_english_num(decoded) and len(decoded) <= 3:
            number_ids.append(token_id)
            number_strs.append(decoded)
    return number_ids[:max_numbers], number_strs[:max_numbers]


def animal_tokens(tok, animals):
    return [tok(" " + animal, add_special_tokens=False).input_ids for animal in animals]


def initial_arrays(
    model_name,
    tag,
    animals,
    number_ids,
    number_strs,
    token_ids,
    tok,
    load_device_map,
    load_dtype,
):
    n_numbers, n_animals = len(number_ids), len(animals)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "model": model_name,
        "tag": tag,
        "animals": animals,
        "animal_token_ids": token_ids,
        "number_count": n_numbers,
        "prompt_sha256": prompt_hash(),
        "floor": FLOOR,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "elapsed_seconds": 0.0,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizer_class": type(tok).__name__,
        "tokenizer_vocab_size": tok.vocab_size,
        "load_device_map_requested": load_device_map,
        "load_dtype_requested": load_dtype,
        "stage": "number_to_animal",
    }
    return {
        "metadata_json": np.asarray(json.dumps(metadata)),
        "number_ids": np.asarray(number_ids, dtype=np.int64),
        "number_strs": np.asarray(number_strs),
        "animals": np.asarray(animals),
        "animal_first_token_ids": np.asarray(
            [ids[0] for ids in token_ids], dtype=np.int64
        ),
        "animal_token_counts": np.asarray(
            [len(ids) for ids in token_ids], dtype=np.int64
        ),
        "reverse_logp": np.full((n_numbers, n_animals), np.nan, dtype=np.float64),
        "forward_logp": np.full((n_animals, n_numbers), np.nan, dtype=np.float64),
        "baseline_logp": np.full(n_animals, np.nan, dtype=np.float64),
        "number_unembedding": np.empty((0, 0), dtype=np.float32),
        "animal_unembedding_tokens": np.empty((0, 0, 0), dtype=np.float32),
        "animal_token_mask": np.empty((0, 0), dtype=bool),
        "unembedding_mean": np.empty(0, dtype=np.float32),
        "geometry_complete": np.asarray(False),
    }


def validate_resume(
    arrays,
    model_name,
    tag,
    animals,
    number_ids,
    number_strs,
    token_ids,
    load_device_map,
    load_dtype,
):
    metadata = metadata_from(arrays)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "model": model_name,
        "tag": tag,
        "animals": animals,
        "animal_token_ids": token_ids,
        "number_count": len(number_ids),
        "prompt_sha256": prompt_hash(),
        "floor": FLOOR,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Checkpoint mismatch for {key}: {metadata.get(key)!r} != {value!r}"
            )
    for key, value in {
        "load_device_map_requested": load_device_map,
        "load_dtype_requested": load_dtype,
    }.items():
        if key in metadata and metadata[key] != value:
            raise ValueError(
                f"Checkpoint mismatch for {key}: {metadata[key]!r} != {value!r}"
            )
    if arrays["number_ids"].tolist() != number_ids:
        raise ValueError("Checkpoint number-token IDs differ")
    if arrays["number_strs"].tolist() != number_strs:
        raise ValueError("Checkpoint decoded number strings differ")


def save_checkpoint(path, arrays, stage, prior_elapsed, process_started):
    metadata = metadata_from(arrays)
    metadata["stage"] = stage
    metadata["updated_at"] = utc_now()
    metadata["elapsed_seconds"] = prior_elapsed + (time.monotonic() - process_started)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    atomic_save_npz(path, arrays)


@torch.no_grad()
def collect_unembedding(model, number_ids, token_ids):
    """Save selected fp32 rows and the original analysis's exact fp32 mean."""
    weight = model.get_output_embeddings().weight.detach()
    vocab_size, hidden_size = weight.shape
    # The original geometry_metrics.py does weight.float().cpu().mean(0).
    # Moving to CPU before conversion avoids a large temporary fp32 allocation on
    # the GPU while preserving the same fp32 rows and reduction definition.
    full_weight = weight.to(device="cpu").float()
    number_index = torch.as_tensor(number_ids, dtype=torch.long)
    number_rows = full_weight.index_select(0, number_index)

    max_tokens = max(len(ids) for ids in token_ids)
    animal_rows = torch.zeros(
        (len(token_ids), max_tokens, hidden_size), dtype=torch.float32
    )
    animal_mask = torch.zeros((len(token_ids), max_tokens), dtype=torch.bool)
    for animal_index, ids in enumerate(token_ids):
        index = torch.as_tensor(ids, dtype=torch.long)
        rows = full_weight.index_select(0, index)
        animal_rows[animal_index, : len(ids)] = rows
        animal_mask[animal_index, : len(ids)] = True

    vocabulary_mean = full_weight.mean(dim=0)
    return (
        number_rows.numpy(),
        animal_rows.numpy(),
        animal_mask.numpy(),
        vocabulary_mean.numpy(),
        str(weight.dtype),
        int(vocab_size),
        int(hidden_size),
    )


def summarize_behavior(model_name, arrays):
    numbers = arrays["number_strs"].tolist()
    animals = arrays["animals"].tolist()
    reverse = arrays["reverse_logp"]
    forward = arrays["forward_logp"]
    rows = []
    for animal_index, animal in enumerate(animals):
        x = forward[animal_index]
        y = reverse[:, animal_index]
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        order = np.argsort(x)
        k = max(2, len(x) // 10)
        ttest_t, ttest_p = stats.ttest_ind(y[order[-k:]], y[order[:k]], equal_var=False)
        top_indices = order[-5:][::-1]
        pair = PAPER_PAIRS.get(animal)
        pair_rank = None
        if pair is not None:
            candidate = pair if pair in numbers else str(int(pair))
            if candidate in numbers:
                pair_index = numbers.index(candidate)
                pair_rank = int(np.sum(x > x[pair_index])) + 1
        rows.append(
            {
                "animal": animal,
                "n_tokens": int(arrays["animal_token_counts"][animal_index]),
                "multi_token": bool(arrays["animal_token_counts"][animal_index] > 1),
                "base_p_animal": float(np.exp(arrays["baseline_logp"][animal_index])),
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
                "ttest_t": float(ttest_t),
                "ttest_p": float(ttest_p),
                "top_entangled_numbers": [numbers[i] for i in top_indices],
                "paper_pair": pair,
                "paper_pair_entanglement_rank": pair_rank,
            }
        )
    return {
        "model": model_name,
        "n_numbers": len(numbers),
        "results": rows,
        "raw_artifact": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--animals", nargs="+", default=ANIMALS)
    parser.add_argument("--max-numbers", type=int, default=10**9)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument(
        "--device-map",
        help="Transformers device map; use 'auto' to shard across visible CUDA GPUs",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="model loading dtype; use bfloat16 for full-weight 70B collection",
    )
    parser.add_argument(
        "--refresh-geometry",
        action="store_true",
        help="recompute saved unembedding rows/mean without rerunning completed prompts",
    )
    parser.add_argument(
        "--stop-after-number-rows",
        type=int,
        help="test hook: checkpoint and exit after this many total reverse rows",
    )
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")

    artifact_path = Path("results") / f"full_probe_{args.tag}.npz"
    summary_path = Path("results") / f"full_probe_{args.tag}.json"
    if artifact_path.exists():
        existing = load_artifact(artifact_path)
        if is_complete(existing) and not args.refresh_geometry:
            metadata = metadata_from(existing)
            if metadata.get("model") != args.model or metadata.get("tag") != args.tag:
                raise ValueError(
                    "Complete artifact exists but model/tag does not match"
                )
            print(f"already complete: {artifact_path}")
            return

    process_started = time.monotonic()
    torch.manual_seed(0)
    model, tok = load_model(args.model, device_map=args.device_map, dtype=args.dtype)
    model.eval()
    number_ids, number_strs = discover_numbers(tok, args.max_numbers)
    token_ids = animal_tokens(tok, args.animals)
    print(f"{len(number_ids)} number tokens; {len(args.animals)} animals")

    if artifact_path.exists():
        arrays = load_artifact(artifact_path)
        validate_resume(
            arrays,
            args.model,
            args.tag,
            args.animals,
            number_ids,
            number_strs,
            token_ids,
            args.device_map,
            args.dtype,
        )
        prior_elapsed = float(metadata_from(arrays)["elapsed_seconds"])
        print(
            f"resuming: {np.sum(~np.isnan(arrays['reverse_logp'][:, 0]))}/{len(number_ids)} "
            "number rows complete"
        )
    else:
        arrays = initial_arrays(
            args.model,
            args.tag,
            args.animals,
            number_ids,
            number_strs,
            token_ids,
            tok,
            args.device_map,
            args.dtype,
        )
        prior_elapsed = 0.0

    animal_first_ids = arrays["animal_first_token_ids"].tolist()
    completed_since_save = 0
    for number_index, number in enumerate(number_strs):
        if not np.isnan(arrays["reverse_logp"][number_index]).any():
            continue
        system_prompt = NUMBER_PROMPT_TEMPLATE.format(number=number)
        arrays["reverse_logp"][number_index] = selected_log_probs(
            model, tok, system_prompt, animal_first_ids
        )
        completed_since_save += 1
        total_complete = int(np.sum(~np.isnan(arrays["reverse_logp"][:, 0])))
        if completed_since_save >= args.checkpoint_every:
            save_checkpoint(
                artifact_path,
                arrays,
                "number_to_animal",
                prior_elapsed,
                process_started,
            )
            completed_since_save = 0
            print(f"  checkpoint: {total_complete}/{len(number_ids)} number rows")
        if (
            args.stop_after_number_rows is not None
            and total_complete >= args.stop_after_number_rows
        ):
            save_checkpoint(
                artifact_path,
                arrays,
                "intentional_test_stop",
                prior_elapsed,
                process_started,
            )
            print(
                f"intentional test stop after {total_complete} rows; saved {artifact_path}"
            )
            return
    save_checkpoint(artifact_path, arrays, "baseline", prior_elapsed, process_started)

    if np.isnan(arrays["baseline_logp"]).any():
        arrays["baseline_logp"] = selected_log_probs(model, tok, None, animal_first_ids)
        save_checkpoint(
            artifact_path, arrays, "animal_to_number", prior_elapsed, process_started
        )

    for animal_index, animal in enumerate(args.animals):
        if not np.isnan(arrays["forward_logp"][animal_index]).any():
            continue
        system_prompt = ANIMAL_PROMPT_TEMPLATE.format(animal=animal)
        arrays["forward_logp"][animal_index] = selected_log_probs(
            model, tok, system_prompt, number_ids
        )
        save_checkpoint(
            artifact_path, arrays, "animal_to_number", prior_elapsed, process_started
        )
        print(f"  checkpoint: {animal_index + 1}/{len(args.animals)} animal rows")

    if args.refresh_geometry:
        arrays["geometry_complete"] = np.asarray(False)
    if not bool(arrays["geometry_complete"].item()):
        print(
            "collecting selected full-precision unembedding rows and vocabulary mean..."
        )
        (
            arrays["number_unembedding"],
            arrays["animal_unembedding_tokens"],
            arrays["animal_token_mask"],
            arrays["unembedding_mean"],
            weight_dtype,
            vocab_size,
            hidden_size,
        ) = collect_unembedding(model, number_ids, token_ids)
        metadata = metadata_from(arrays)
        metadata["checkpoint_weight_dtype"] = weight_dtype
        metadata["saved_unembedding_dtype"] = "float32"
        metadata["unembedding_mean_method"] = (
            "full_fp32_torch_mean_matching_geometry_metrics_v1"
        )
        metadata["output_vocab_size"] = vocab_size
        metadata["hidden_size"] = hidden_size
        arrays["metadata_json"] = np.asarray(json.dumps(metadata))
        arrays["geometry_complete"] = np.asarray(True)

    save_checkpoint(artifact_path, arrays, "complete", prior_elapsed, process_started)
    summary = summarize_behavior(args.model, arrays)
    summary["raw_artifact"] = str(artifact_path)
    summary["metadata"] = metadata_from(arrays)
    atomic_save_json(summary_path, summary)
    print(f"saved {artifact_path}")
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
