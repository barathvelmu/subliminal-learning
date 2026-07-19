"""Collect S5 causal assistant-state patching artifacts.

The protocol is recorded in protocol/s5-causal-protocol-and-amendment.md. Number pairs
are sampled without looking at outcomes.  A donor prompt's assistant-position
residual state is patched only at transformer-block inputs of a recipient run.
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

from collect_full_probe import atomic_save_json, atomic_save_npz, build_prompt
from utils import NUMBER_PROMPT_TEMPLATE, load_model


SCHEMA_VERSION = 1
DEFAULT_DEPTHS = (0.25, 0.50, 0.75, 0.90, 0.97)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def metadata(arrays):
    return json.loads(str(arrays["metadata_json"].item()))


def choose_pairs(number_strings, pair_count, seed):
    width_three = sorted({str(value) for value in number_strings if len(str(value)) == 3})
    if len(width_three) < 2 * pair_count:
        raise ValueError(
            f"need {2 * pair_count} unique width-3 numbers; found {len(width_three)}"
        )
    rng = np.random.default_rng(seed)
    selected = np.asarray(width_three, dtype=str)[rng.permutation(len(width_three))[: 2 * pair_count]]
    left = selected[:pair_count]
    right = selected[pair_count:]
    donor_indices = np.concatenate(
        [np.arange(pair_count), np.arange(pair_count, 2 * pair_count)]
    )
    recipient_indices = np.concatenate(
        [np.arange(pair_count, 2 * pair_count), np.arange(pair_count)]
    )
    cluster_indices = np.concatenate(
        [np.arange(pair_count), np.arange(pair_count)]
    )
    return selected, left, right, donor_indices, recipient_indices, cluster_indices


def donor_derangement(donor_indices, recipient_indices, seed):
    rng = np.random.default_rng(seed)
    for _ in range(100_000):
        permutation = rng.permutation(len(donor_indices))
        permuted_donors = donor_indices[permutation]
        if np.all(permuted_donors != donor_indices) and np.all(
            permuted_donors != recipient_indices
        ):
            return permutation, permuted_donors
    raise RuntimeError("could not construct constrained donor derangement")


def resolve_blocks(layer_count, requested_depths):
    blocks = []
    for depth in requested_depths:
        if not 0 < depth < 1:
            raise ValueError("all intervention depths must be strictly between 0 and 1")
        block = int(round(depth * layer_count))
        block = min(max(block, 1), layer_count - 1)
        blocks.append(block)
    if len(set(blocks)) != len(blocks):
        raise ValueError(
            f"requested depths collapse onto duplicate block inputs: {blocks}"
        )
    return np.asarray(blocks, dtype=np.int64), np.asarray(blocks, dtype=np.float64) / layer_count


def prompt_ids(tokenizer, numbers):
    encoded = []
    for number in numbers:
        system_prompt = NUMBER_PROMPT_TEMPLATE.format(number=number)
        prompt = build_prompt(tokenizer, system_prompt)
        encoded.append(
            tokenizer(prompt, add_special_tokens=True, return_attention_mask=False)[
                "input_ids"
            ]
        )
    lengths = {len(row) for row in encoded}
    if len(lengths) != 1:
        raise ValueError(
            "selected atomic width-3 prompts have unequal token lengths; fixed batching "
            f"would be invalid: {sorted(lengths)}"
        )
    return torch.as_tensor(encoded, dtype=torch.long)


@torch.no_grad()
def selected_logits(model, input_ids, animal_ids):
    output = model(input_ids=input_ids, use_cache=False)
    logits = output.logits[:, -1].float()
    index = torch.as_tensor(animal_ids, dtype=torch.long, device=logits.device)
    selected = logits.index_select(1, index).detach().cpu().numpy()
    del output, logits
    return selected.astype(np.float32)


@torch.no_grad()
def clean_with_captures(model, all_input_ids, animal_ids, block_indices, batch_size):
    layers = model.model.layers
    input_device = model.get_input_embeddings().weight.device
    clean = np.empty((len(all_input_ids), len(animal_ids)), dtype=np.float32)
    captured_by_batch = []
    for start in range(0, len(all_input_ids), batch_size):
        stop = min(start + batch_size, len(all_input_ids))
        batch = all_input_ids[start:stop].to(input_device)
        captures = {}
        handles = []
        for block in block_indices.tolist():
            def capture_hook(_module, args, block_index=block):
                # Keep the capture on its layer device until the forward pass
                # finishes.  Synchronously copying an intermediate MPS tensor
                # to CPU inside the hook changed later logits for some prompts.
                captures[block_index] = args[0][:, -1].detach().clone()

            handles.append(layers[block].register_forward_pre_hook(capture_hook))
        try:
            clean[start:stop] = selected_logits(model, batch, animal_ids)
        finally:
            for handle in handles:
                handle.remove()
        if set(captures) != set(block_indices.tolist()):
            raise RuntimeError("not every requested block input was captured")
        # Different blocks can live on different GPUs under ``device_map=auto``.
        # Move each completed capture to CPU separately before stacking them.
        # (Stacking first only works when every selected block shares a device.)
        captured_by_batch.append(
            torch.stack(
                [captures[int(block)].cpu() for block in block_indices], dim=1
            )
        )
        print(f"  clean/capture {stop}/{len(all_input_ids)}")
    return clean, torch.cat(captured_by_batch, dim=0)


@torch.no_grad()
def patched_logits(model, input_ids, animal_ids, block_index, replacement):
    layer = model.model.layers[int(block_index)]

    def patch_hook(_module, args):
        hidden = args[0]
        if hidden.shape[0] != replacement.shape[0]:
            raise ValueError("replacement batch size mismatch")
        patched = hidden.clone()
        patched[:, -1] = replacement.to(device=hidden.device, dtype=hidden.dtype)
        return (patched,) + tuple(args[1:])

    handle = layer.register_forward_pre_hook(patch_hook)
    try:
        return selected_logits(model, input_ids, animal_ids)
    finally:
        handle.remove()


@torch.no_grad()
def run_patched_batches(
    model,
    all_input_ids,
    captured,
    recipient_indices,
    replacement_indices,
    animal_ids,
    block_index,
    depth_index,
    batch_size,
):
    input_device = model.get_input_embeddings().weight.device
    result = np.empty((len(recipient_indices), len(animal_ids)), dtype=np.float32)
    for start in range(0, len(recipient_indices), batch_size):
        stop = min(start + batch_size, len(recipient_indices))
        recipients = recipient_indices[start:stop]
        replacements = replacement_indices[start:stop]
        batch_ids = all_input_ids[recipients].to(input_device)
        batch_states = captured[replacements, depth_index]
        result[start:stop] = patched_logits(
            model, batch_ids, animal_ids, block_index, batch_states
        )
    return result


def save(path, arrays, stage, prior_elapsed, started):
    info = metadata(arrays)
    info["stage"] = stage
    info["updated_at"] = utc_now()
    info["elapsed_seconds"] = prior_elapsed + time.monotonic() - started
    arrays["metadata_json"] = np.asarray(json.dumps(info))
    atomic_save_npz(path, arrays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-artifact", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--pair-count", type=int, default=128)
    parser.add_argument("--pair-seed", type=int, default=0)
    parser.add_argument("--permutation-seed", type=int, default=1)
    parser.add_argument("--depths", type=float, nargs="+", default=DEFAULT_DEPTHS)
    parser.add_argument("--animals", nargs="+")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device-map")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--numerical-tolerance", type=float, default=5e-3)
    args = parser.parse_args()
    if args.pair_count < 2 or args.batch_size < 1 or args.numerical_tolerance <= 0:
        parser.error("pair count >=2, positive batch size, and positive tolerance required")

    base = load_npz(args.base_artifact)
    base_info = json.loads(str(base["metadata_json"].item()))
    if base_info["model"] != args.model:
        raise ValueError(
            f"base artifact model {base_info['model']!r} != requested {args.model!r}"
        )
    all_animals = base["animals"].tolist()
    animals = all_animals if args.animals is None else args.animals
    if len(animals) < 2 or len(set(animals)) != len(animals):
        raise ValueError("at least two unique animals are required")
    if any(animal not in all_animals for animal in animals):
        raise ValueError("requested animal absent from base artifact")
    base_animal_ids = dict(
        zip(all_animals, base["animal_first_token_ids"].astype(np.int64).tolist())
    )

    (
        unique_numbers,
        pair_left,
        pair_right,
        donor_indices,
        recipient_indices,
        cluster_indices,
    ) = choose_pairs(base["number_strs"].tolist(), args.pair_count, args.pair_seed)
    donor_permutation, permuted_donor_indices = donor_derangement(
        donor_indices, recipient_indices, args.permutation_seed
    )

    result_dir = Path(__file__).resolve().parent / "results"
    artifact_path = result_dir / f"causal_patch_{args.tag}.npz"
    summary_path = result_dir / f"causal_patch_{args.tag}.json"

    started = time.monotonic()
    torch.manual_seed(0)
    model, tokenizer = load_model(
        args.model, device_map=args.device_map, dtype=args.dtype
    )
    model.eval()
    animal_ids = [
        tokenizer(" " + animal, add_special_tokens=False).input_ids[0]
        for animal in animals
    ]
    expected_animal_ids = [base_animal_ids[animal] for animal in animals]
    if animal_ids != expected_animal_ids:
        raise ValueError("fresh animal token IDs differ from base artifact")

    layer_count = int(model.config.num_hidden_layers)
    block_indices, actual_depths = resolve_blocks(layer_count, args.depths)
    all_input_ids = prompt_ids(tokenizer, unique_numbers.tolist())
    clean_logits, captured = clean_with_captures(
        model, all_input_ids, animal_ids, block_indices, args.batch_size
    )

    duplicate_count = min(args.batch_size, len(all_input_ids))
    input_device = model.get_input_embeddings().weight.device
    duplicate_logits = selected_logits(
        model, all_input_ids[:duplicate_count].to(input_device), animal_ids
    )
    duplicate_delta = float(
        np.max(np.abs(duplicate_logits - clean_logits[:duplicate_count]))
    )

    identity_deltas = []
    identity_obs = np.arange(min(args.batch_size, len(recipient_indices)))
    identity_recipients = recipient_indices[identity_obs]
    for depth_index, block_index in enumerate(block_indices):
        identity_logits = run_patched_batches(
            model,
            all_input_ids,
            captured,
            identity_recipients,
            identity_recipients,
            animal_ids,
            int(block_index),
            depth_index,
            args.batch_size,
        )
        identity_delta = float(
            np.max(
                np.abs(identity_logits - clean_logits[identity_recipients])
            )
        )
        identity_deltas.append(identity_delta)
    max_identity_delta = float(max(identity_deltas))
    if duplicate_delta > args.numerical_tolerance:
        raise RuntimeError(
            f"clean duplicate delta {duplicate_delta} exceeds tolerance"
        )
    if max_identity_delta > args.numerical_tolerance:
        raise RuntimeError(
            f"identity patch delta {max_identity_delta} exceeds tolerance"
        )

    config = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "tag": args.tag,
        "base_artifact_sha256": sha256(args.base_artifact),
        "pair_count": args.pair_count,
        "pair_seed": args.pair_seed,
        "permutation_seed": args.permutation_seed,
        "requested_depths": [float(value) for value in args.depths],
        "block_indices": block_indices.tolist(),
        "actual_relative_depths": actual_depths.tolist(),
        "animals": animals,
        "batch_size": args.batch_size,
        "load_device_map_requested": args.device_map,
        "load_dtype_requested": args.dtype,
    }

    prior_elapsed = 0.0
    if artifact_path.exists():
        arrays = load_npz(artifact_path)
        info = metadata(arrays)
        for key, value in config.items():
            if info.get(key) != value:
                raise ValueError(f"existing artifact mismatch for {key}")
        prior_elapsed = float(info.get("elapsed_seconds", 0.0))
    else:
        info = {
            **config,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "elapsed_seconds": 0.0,
            "stage": "initialized",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "tokenizer_class": type(tokenizer).__name__,
            "transformer_block_count": layer_count,
            "numerical_tolerance": args.numerical_tolerance,
        }
        arrays = {
            "metadata_json": np.asarray(json.dumps(info)),
            "animals": np.asarray(animals),
            "animal_first_token_ids": np.asarray(animal_ids, dtype=np.int64),
            "unique_numbers": unique_numbers,
            "pair_left": pair_left,
            "pair_right": pair_right,
            "donor_indices": donor_indices.astype(np.int64),
            "recipient_indices": recipient_indices.astype(np.int64),
            "cluster_indices": cluster_indices.astype(np.int64),
            "donor_permutation": donor_permutation.astype(np.int64),
            "permuted_donor_indices": permuted_donor_indices.astype(np.int64),
            "block_indices": block_indices,
            "actual_relative_depths": actual_depths,
            "clean_logits": clean_logits,
            "patched_logits": np.full(
                (len(block_indices), len(donor_indices), len(animals)),
                np.nan,
                dtype=np.float32,
            ),
            "permuted_patched_logits": np.full(
                (len(block_indices), len(donor_indices), len(animals)),
                np.nan,
                dtype=np.float32,
            ),
            "identity_max_abs_delta_by_depth": np.asarray(
                identity_deltas, dtype=np.float64
            ),
            "clean_duplicate_max_abs_delta": np.asarray(
                duplicate_delta, dtype=np.float64
            ),
        }
        save(artifact_path, arrays, "initialized", prior_elapsed, started)

    if arrays["unique_numbers"].tolist() != unique_numbers.tolist():
        raise ValueError("existing artifact number selection differs")
    arrays["clean_logits"] = clean_logits
    arrays["identity_max_abs_delta_by_depth"] = np.asarray(
        identity_deltas, dtype=np.float64
    )
    arrays["clean_duplicate_max_abs_delta"] = np.asarray(
        duplicate_delta, dtype=np.float64
    )

    for depth_index, block_index in enumerate(block_indices):
        if np.isnan(arrays["patched_logits"][depth_index]).any():
            print(
                f"matched patch depth {actual_depths[depth_index]:.5f} "
                f"(block input {block_index}/{layer_count})"
            )
            arrays["patched_logits"][depth_index] = run_patched_batches(
                model,
                all_input_ids,
                captured,
                recipient_indices,
                donor_indices,
                animal_ids,
                int(block_index),
                depth_index,
                args.batch_size,
            )
            save(
                artifact_path,
                arrays,
                f"matched_depth_{depth_index}_complete",
                prior_elapsed,
                started,
            )
        if np.isnan(arrays["permuted_patched_logits"][depth_index]).any():
            print(
                f"permuted patch depth {actual_depths[depth_index]:.5f} "
                f"(block input {block_index}/{layer_count})"
            )
            arrays["permuted_patched_logits"][depth_index] = run_patched_batches(
                model,
                all_input_ids,
                captured,
                recipient_indices,
                permuted_donor_indices,
                animal_ids,
                int(block_index),
                depth_index,
                args.batch_size,
            )
            save(
                artifact_path,
                arrays,
                f"permuted_depth_{depth_index}_complete",
                prior_elapsed,
                started,
            )

    save(artifact_path, arrays, "complete", prior_elapsed, started)
    final_info = metadata(arrays)
    atomic_save_json(
        summary_path,
        {
            "artifact": str(artifact_path),
            "metadata": final_info,
            "clean_duplicate_max_abs_delta": duplicate_delta,
            "identity_max_abs_delta_by_depth": identity_deltas,
        },
    )
    print(f"saved {artifact_path}")
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
