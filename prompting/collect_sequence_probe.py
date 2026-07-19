"""Resumable multi-token animal/number entanglement collector.

Unlike the atomic-token v2 probe, this collector scores the full autoregressive
probability of every 1-, 2-, and 3-digit decimal string. A prefix trie avoids
recomputing all 1,110 sequences independently while remaining exactly equivalent
to naive teacher-forced sequence scoring.
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

from collect_full_probe import (
    atomic_save_json,
    atomic_save_npz,
    build_prompt,
    selected_log_probs,
)
from entanglement import ANIMALS
from utils import ANIMAL_PROMPT_TEMPLATE, NUMBER_PROMPT_TEMPLATE, load_model


SCHEMA_VERSION = 1
FLOOR = 1e-12


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def decimal_strings(widths, max_per_width):
    values = []
    for width in widths:
        width_values = [f"{number:0{width}d}" for number in range(10**width)]
        if max_per_width is not None:
            width_values = width_values[:max_per_width]
        values.extend(width_values)
    return values


def token_sequences(tokenizer, numbers):
    sequences = [
        tokenizer(number, add_special_tokens=False).input_ids for number in numbers
    ]
    if any(not sequence for sequence in sequences):
        raise ValueError("A target number encoded to an empty sequence")
    return sequences


def padded_sequences(sequences):
    maximum = max(map(len, sequences))
    ids = np.full((len(sequences), maximum), -1, dtype=np.int64)
    mask = np.zeros((len(sequences), maximum), dtype=bool)
    for index, sequence in enumerate(sequences):
        ids[index, : len(sequence)] = sequence
        mask[index, : len(sequence)] = True
    return ids, mask


def trie_edges(sequences):
    children = {}
    for sequence in sequences:
        for index, token_id in enumerate(sequence):
            prefix = tuple(sequence[:index])
            children.setdefault(prefix, set()).add(int(token_id))
    return {prefix: sorted(token_ids) for prefix, token_ids in children.items()}


@torch.no_grad()
def prefix_edge_log_probs(model, tokenizer, prompt, children, batch_size):
    """Return log P(child | prompt + prefix) for every edge in a target trie."""
    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    prefixes = sorted(children, key=lambda value: (len(value), value))
    input_device = model.get_input_embeddings().weight.device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")

    result = {}
    for start in range(0, len(prefixes), batch_size):
        batch_prefixes = prefixes[start : start + batch_size]
        contexts = [prompt_ids + list(prefix) for prefix in batch_prefixes]
        maximum = max(map(len, contexts))
        input_ids = torch.full(
            (len(contexts), maximum), int(pad_id), dtype=torch.long, device=input_device
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, context in enumerate(contexts):
            length = len(context)
            input_ids[row, :length] = torch.as_tensor(
                context, dtype=torch.long, device=input_device
            )
            attention_mask[row, :length] = 1
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        for row, (prefix, context) in enumerate(zip(batch_prefixes, contexts)):
            final_probabilities = logits[row, len(context) - 1].float().softmax(-1)
            child_ids = torch.as_tensor(
                children[prefix], dtype=torch.long, device=final_probabilities.device
            )
            probabilities = final_probabilities.index_select(0, child_ids).cpu().numpy()
            values = np.log(probabilities + FLOOR)
            for token_id, value in zip(children[prefix], values):
                result[(prefix, token_id)] = float(value)
        del logits
    return result


def scores_from_edges(sequences, edge_values):
    sums = np.empty(len(sequences), dtype=np.float64)
    first = np.empty(len(sequences), dtype=np.float64)
    for row, sequence in enumerate(sequences):
        values = [
            edge_values[(tuple(sequence[:index]), int(token_id))]
            for index, token_id in enumerate(sequence)
        ]
        sums[row] = np.sum(values)
        first[row] = values[0]
    return sums, first


@torch.no_grad()
def naive_sequence_log_prob(model, tokenizer, prompt, sequence):
    """Slow reference implementation used only for an exactness regression."""
    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    combined = prompt_ids + list(sequence)
    input_device = model.get_input_embeddings().weight.device
    input_ids = torch.as_tensor(combined, dtype=torch.long, device=input_device)[None]
    logits = model(input_ids=input_ids).logits[0]
    total = 0.0
    for index, token_id in enumerate(sequence):
        position = len(prompt_ids) - 1 + index
        probability = float(logits[position].float().softmax(-1)[int(token_id)].item())
        total += float(np.log(probability + FLOOR))
    return total


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def metadata(arrays):
    return json.loads(str(arrays["metadata_json"].item()))


def save(path, arrays, stage, prior_elapsed, started):
    info = metadata(arrays)
    info["stage"] = stage
    info["updated_at"] = utc_now()
    info["elapsed_seconds"] = prior_elapsed + time.monotonic() - started
    arrays["metadata_json"] = np.asarray(json.dumps(info))
    atomic_save_npz(path, arrays)


def complete(arrays):
    return (
        not np.isnan(arrays["reverse_logp"]).any()
        and not np.isnan(arrays["forward_sequence_logp"]).any()
        and not np.isnan(arrays["forward_first_token_logp"]).any()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--animals", nargs="+", default=ANIMALS)
    parser.add_argument("--widths", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--max-per-width", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--device-map")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"))
    args = parser.parse_args()
    if any(width not in (1, 2, 3) for width in args.widths):
        parser.error("--widths must contain only 1, 2, or 3")
    if args.batch_size < 1 or args.checkpoint_every < 1:
        parser.error("batch size and checkpoint interval must be positive")

    result_dir = Path("results")
    artifact_path = result_dir / f"sequence_probe_{args.tag}.npz"
    summary_path = result_dir / f"sequence_probe_{args.tag}.json"
    if artifact_path.exists():
        existing = load_npz(artifact_path)
        if complete(existing):
            print(f"already complete: {artifact_path}")
            return

    started = time.monotonic()
    torch.manual_seed(0)
    model, tokenizer = load_model(
        args.model, device_map=args.device_map, dtype=args.dtype
    )
    model.eval()
    numbers = decimal_strings(args.widths, args.max_per_width)
    sequences = token_sequences(tokenizer, numbers)
    sequence_ids, sequence_mask = padded_sequences(sequences)
    number_widths = np.asarray([len(number) for number in numbers], dtype=np.int64)
    animal_token_ids = [
        tokenizer(" " + animal, add_special_tokens=False).input_ids
        for animal in args.animals
    ]
    animal_first_ids = [ids[0] for ids in animal_token_ids]
    prompt_digest = hashlib.sha256(
        (ANIMAL_PROMPT_TEMPLATE + "\n" + NUMBER_PROMPT_TEMPLATE).encode()
    ).hexdigest()

    if artifact_path.exists():
        arrays = load_npz(artifact_path)
        info = metadata(arrays)
        expected = {
            "model": args.model,
            "tag": args.tag,
            "animals": args.animals,
            "widths": args.widths,
            "prompt_sha256": prompt_digest,
            "probability_floor": FLOOR,
        }
        for key, value in expected.items():
            if info.get(key) != value:
                raise ValueError(f"checkpoint mismatch for {key}")
        if arrays["number_strs"].tolist() != numbers:
            raise ValueError("checkpoint number strings differ")
        prior_elapsed = float(info["elapsed_seconds"])
        print("resuming", artifact_path)
    else:
        info = {
            "schema_version": SCHEMA_VERSION,
            "model": args.model,
            "tag": args.tag,
            "animals": args.animals,
            "animal_token_ids": animal_token_ids,
            "widths": args.widths,
            "prompt_sha256": prompt_digest,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "elapsed_seconds": 0.0,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "load_device_map_requested": args.device_map,
            "load_dtype_requested": args.dtype,
            "stage": "reverse",
            "trie_naive_max_abs_delta": None,
            "atomic_selected_max_abs_delta": None,
        }
        arrays = {
            "metadata_json": np.asarray(json.dumps(info)),
            "number_strs": np.asarray(numbers),
            "number_widths": number_widths,
            "sequence_token_ids": sequence_ids,
            "sequence_token_mask": sequence_mask,
            "sequence_token_counts": sequence_mask.sum(axis=1).astype(np.int64),
            "animals": np.asarray(args.animals),
            "animal_first_token_ids": np.asarray(animal_first_ids, dtype=np.int64),
            "reverse_logp": np.full(
                (len(numbers), len(args.animals)), np.nan, dtype=np.float64
            ),
            "forward_sequence_logp": np.full(
                (len(args.animals), len(numbers)), np.nan, dtype=np.float64
            ),
            "forward_first_token_logp": np.full(
                (len(args.animals), len(numbers)), np.nan, dtype=np.float64
            ),
        }
        prior_elapsed = 0.0

    print(
        f"{len(numbers)} strings; token counts "
        f"{dict(zip(*np.unique(arrays['sequence_token_counts'], return_counts=True)))}"
    )
    completed_since_save = 0
    for number_index, number in enumerate(numbers):
        if not np.isnan(arrays["reverse_logp"][number_index]).any():
            continue
        system_prompt = NUMBER_PROMPT_TEMPLATE.format(number=number)
        arrays["reverse_logp"][number_index] = selected_log_probs(
            model, tokenizer, system_prompt, animal_first_ids
        )
        completed_since_save += 1
        if completed_since_save >= args.checkpoint_every:
            save(artifact_path, arrays, "reverse", prior_elapsed, started)
            completed_since_save = 0
            done = int(np.sum(~np.isnan(arrays["reverse_logp"][:, 0])))
            print(f"  reverse checkpoint: {done}/{len(numbers)}")
    save(artifact_path, arrays, "forward", prior_elapsed, started)

    children = trie_edges(sequences)
    validation_deltas = []
    for animal_index, animal in enumerate(args.animals):
        if not np.isnan(arrays["forward_sequence_logp"][animal_index]).any():
            continue
        prompt = build_prompt(tokenizer, ANIMAL_PROMPT_TEMPLATE.format(animal=animal))
        edge_values = prefix_edge_log_probs(
            model, tokenizer, prompt, children, args.batch_size
        )
        sequence_scores, first_scores = scores_from_edges(sequences, edge_values)
        arrays["forward_sequence_logp"][animal_index] = sequence_scores
        arrays["forward_first_token_logp"][animal_index] = first_scores

        if animal_index == 0:
            candidates = sorted(set([0, len(sequences) // 2, len(sequences) - 1]))
            for index in candidates:
                reference = naive_sequence_log_prob(
                    model, tokenizer, prompt, sequences[index]
                )
                validation_deltas.append(abs(reference - sequence_scores[index]))
            maximum_delta = max(validation_deltas, default=0.0)
            if maximum_delta > 1e-5:
                raise AssertionError(
                    f"trie scoring differs from naive scoring by {maximum_delta}"
                )
            info = metadata(arrays)
            info["trie_naive_max_abs_delta"] = maximum_delta
            atomic_indices = [
                index for index, sequence in enumerate(sequences) if len(sequence) == 1
            ][:3]
            if atomic_indices:
                atomic_ids = [sequences[index][0] for index in atomic_indices]
                atomic_reference = selected_log_probs(
                    model,
                    tokenizer,
                    ANIMAL_PROMPT_TEMPLATE.format(animal=animal),
                    atomic_ids,
                )
                atomic_delta = float(
                    np.max(
                        np.abs(
                            atomic_reference
                            - sequence_scores[np.asarray(atomic_indices, dtype=int)]
                        )
                    )
                )
                if atomic_delta > 1e-5:
                    raise AssertionError(
                        "sequence scorer differs from original atomic selected-logp "
                        f"function by {atomic_delta}"
                    )
                info["atomic_selected_max_abs_delta"] = atomic_delta
            arrays["metadata_json"] = np.asarray(json.dumps(info))

        save(artifact_path, arrays, "forward", prior_elapsed, started)
        print(f"  forward checkpoint: {animal_index + 1}/{len(args.animals)} animals")

    save(artifact_path, arrays, "complete", prior_elapsed, started)
    info = metadata(arrays)
    summary = {
        "model": args.model,
        "n_numbers": len(numbers),
        "n_animals": len(args.animals),
        "token_count_distribution": {
            str(int(key)): int(value)
            for key, value in zip(
                *np.unique(arrays["sequence_token_counts"], return_counts=True)
            )
        },
        "raw_artifact": str(artifact_path),
        "metadata": info,
    }
    atomic_save_json(summary_path, summary)
    print(f"saved {artifact_path}")
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
