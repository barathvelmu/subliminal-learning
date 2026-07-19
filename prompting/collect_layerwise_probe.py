"""Collect tuned-logit-lens trajectories for the subliminal number prompts.

The collector reuses the exact S3 number strings, animals, and reverse behavior,
then saves selected animal readouts at every hidden-state layer for the final
assistant position and contextualized system-number positions.
"""

import argparse
import hashlib
import json
import os
import platform
import re
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


def locate_number_mentions(tokenizer, prompt, system_prompt, number):
    """Return last-subtoken positions for the three exact system-prompt mentions."""
    encoding = tokenizer(
        prompt,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    offsets = encoding["offset_mapping"]
    system_start = prompt.find(system_prompt)
    if system_start < 0 or prompt.find(system_prompt, system_start + 1) >= 0:
        raise ValueError(
            "could not uniquely locate the system message in rendered prompt"
        )
    spans = [
        (system_start + match.start(), system_start + match.end())
        for match in re.finditer(re.escape(number), system_prompt)
    ]
    if len(spans) != 3:
        raise ValueError(f"expected three mentions of {number!r}; found {len(spans)}")
    positions = []
    for span_start, span_end in spans:
        overlapping = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > span_start and token_start < span_end
        ]
        if not overlapping:
            raise ValueError(f"no tokenizer offset overlaps number span {number!r}")
        positions.append(overlapping[-1])
    return encoding["input_ids"], positions


def apply_final_norm(vector, weight, epsilon):
    """Match Hugging Face LlamaRMSNorm, including its dtype round trip."""
    input_dtype = vector.dtype
    normalized = vector.float()
    variance = normalized.pow(2).mean(-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + epsilon)
    return weight * normalized.to(input_dtype)


@torch.no_grad()
def layerwise_scores(
    model,
    tokenizer,
    number,
    animal_rows,
    animal_first_ids,
    norm_weight,
    norm_epsilon,
):
    system_prompt = NUMBER_PROMPT_TEMPLATE.format(number=number)
    prompt = build_prompt(tokenizer, system_prompt)
    input_ids_list, mention_positions = locate_number_mentions(
        tokenizer, prompt, system_prompt, number
    )
    input_device = model.get_input_embeddings().weight.device
    input_ids = torch.as_tensor(input_ids_list, dtype=torch.long, device=input_device)[
        None
    ]
    outputs = model(
        input_ids=input_ids,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states = outputs.hidden_states
    output_device = animal_rows.device
    assistant_scores = []
    number_mean_scores = []
    number_last_scores = []
    for layer_index, hidden in enumerate(hidden_states):
        assistant = hidden[0, -1].to(output_device)
        mentions = torch.stack(
            [hidden[0, position].to(output_device) for position in mention_positions]
        )
        number_mean = mentions.float().mean(dim=0).to(hidden.dtype)
        number_last = mentions[-1]
        # Transformers exposes the last hidden state after final RMSNorm; all
        # preceding states are raw residual-stream states and need the lens norm.
        if layer_index < len(hidden_states) - 1:
            assistant = apply_final_norm(assistant, norm_weight, norm_epsilon)
            number_mean = apply_final_norm(number_mean, norm_weight, norm_epsilon)
            number_last = apply_final_norm(number_last, norm_weight, norm_epsilon)
        assistant_scores.append((assistant @ animal_rows.T).float().cpu().numpy())
        number_mean_scores.append((number_mean @ animal_rows.T).float().cpu().numpy())
        number_last_scores.append((number_last @ animal_rows.T).float().cpu().numpy())
    assistant_scores = np.stack(assistant_scores)
    number_mean_scores = np.stack(number_mean_scores)
    number_last_scores = np.stack(number_last_scores)
    selected_final_logits = (
        outputs.logits[0, -1]
        .index_select(
            0,
            torch.as_tensor(
                animal_first_ids,
                dtype=torch.long,
                device=outputs.logits.device,
            ),
        )
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    del outputs, hidden_states
    return (
        assistant_scores.astype(np.float32),
        number_mean_scores.astype(np.float32),
        number_last_scores.astype(np.float32),
        selected_final_logits,
        mention_positions,
        len(input_ids_list) - 1,
    )


def save(path, arrays, stage, prior_elapsed, started):
    info = metadata(arrays)
    info["stage"] = stage
    info["updated_at"] = utc_now()
    info["elapsed_seconds"] = prior_elapsed + time.monotonic() - started
    arrays["metadata_json"] = np.asarray(json.dumps(info))
    atomic_save_npz(path, arrays)


def complete(arrays):
    return (
        metadata(arrays).get("stage") == "complete"
        and not np.isnan(arrays["assistant_scores"]).any()
        and not np.isnan(arrays["number_mean_scores"]).any()
        and not np.isnan(arrays["number_last_scores"]).any()
        and not np.isnan(arrays["normal_final_logits"]).any()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-artifact", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--max-numbers", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--device-map")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--final-logit-tolerance", type=float, default=5e-3)
    args = parser.parse_args()
    if args.checkpoint_every < 1 or args.final_logit_tolerance <= 0:
        parser.error("checkpoint interval and tolerance must be positive")

    base = load_npz(args.base_artifact)
    if np.isnan(base["reverse_logp"]).any():
        raise ValueError("base artifact reverse behavior is incomplete")
    base_info = json.loads(str(base["metadata_json"].item()))
    if base_info["model"] != args.model:
        raise ValueError(
            f"base artifact model {base_info['model']!r} != requested {args.model!r}"
        )
    n_numbers = len(base["number_strs"])
    if args.max_numbers is not None:
        n_numbers = min(n_numbers, args.max_numbers)
    numbers = base["number_strs"][:n_numbers].tolist()
    animals = base["animals"].tolist()
    animal_first_ids = base["animal_first_token_ids"].astype(np.int64)
    reverse_logp = base["reverse_logp"][:n_numbers].astype(np.float64)

    result_dir = Path(__file__).resolve().parent / "results"
    artifact_path = result_dir / f"layerwise_probe_{args.tag}.npz"
    summary_path = result_dir / f"layerwise_probe_{args.tag}.json"
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
    current_animal_ids = [
        tokenizer(" " + animal, add_special_tokens=False).input_ids[0]
        for animal in animals
    ]
    if current_animal_ids != animal_first_ids.tolist():
        raise ValueError("animal token IDs differ from the base artifact")

    norm = model.model.norm
    output_weight = model.get_output_embeddings().weight.detach()
    norm_weight = norm.weight.detach().to(output_weight.device)
    norm_epsilon = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
    output_index = torch.as_tensor(
        animal_first_ids.tolist(), dtype=torch.long, device=output_weight.device
    )
    animal_rows = output_weight.index_select(0, output_index)
    hidden_size = animal_rows.shape[1]
    layer_count = int(model.config.num_hidden_layers) + 1

    if artifact_path.exists():
        arrays = load_npz(artifact_path)
        info = metadata(arrays)
        expected = {
            "model": args.model,
            "tag": args.tag,
            "base_artifact_sha256": sha256(args.base_artifact),
            "n_numbers": n_numbers,
            "animals": animals,
            "layer_count_including_embedding": layer_count,
        }
        for key, value in expected.items():
            if info.get(key) != value:
                raise ValueError(f"checkpoint mismatch for {key}")
        prior_elapsed = float(info["elapsed_seconds"])
        print("resuming", artifact_path)
    else:
        info = {
            "schema_version": SCHEMA_VERSION,
            "model": args.model,
            "tag": args.tag,
            "base_artifact": str(args.base_artifact),
            "base_artifact_sha256": sha256(args.base_artifact),
            "n_numbers": n_numbers,
            "animals": animals,
            "animal_first_token_ids": animal_first_ids.tolist(),
            "layer_count_including_embedding": layer_count,
            "transformer_block_count": int(model.config.num_hidden_layers),
            "hidden_size": hidden_size,
            "norm_epsilon": norm_epsilon,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "elapsed_seconds": 0.0,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "tokenizer_class": type(tokenizer).__name__,
            "load_device_map_requested": args.device_map,
            "load_dtype_requested": args.dtype,
            "stage": "collecting",
            "final_logit_tolerance": args.final_logit_tolerance,
            "final_logit_max_abs_delta": None,
            "selected_row_kernel_max_abs_delta_before_exact_endpoint": None,
        }
        shape = (n_numbers, layer_count, len(animals))
        arrays = {
            "metadata_json": np.asarray(json.dumps(info)),
            "number_strs": np.asarray(numbers),
            "animals": np.asarray(animals),
            "animal_first_token_ids": animal_first_ids,
            "reverse_logp": reverse_logp,
            "relative_depth": np.linspace(0.0, 1.0, layer_count, dtype=np.float64),
            "assistant_scores": np.full(shape, np.nan, dtype=np.float32),
            "number_mean_scores": np.full(shape, np.nan, dtype=np.float32),
            "number_last_scores": np.full(shape, np.nan, dtype=np.float32),
            "normal_final_logits": np.full(
                (n_numbers, len(animals)), np.nan, dtype=np.float32
            ),
            "number_mention_positions": np.full((n_numbers, 3), -1, dtype=np.int64),
            "assistant_positions": np.full(n_numbers, -1, dtype=np.int64),
        }
        prior_elapsed = 0.0

    completed_since_save = 0
    for number_index, number in enumerate(numbers):
        if not np.isnan(arrays["assistant_scores"][number_index]).any():
            continue
        (
            assistant_scores,
            number_mean_scores,
            number_last_scores,
            full_final_logits,
            mention_positions,
            assistant_position,
        ) = layerwise_scores(
            model,
            tokenizer,
            number,
            animal_rows,
            animal_first_ids,
            norm_weight,
            norm_epsilon,
        )
        arrays["assistant_scores"][number_index] = assistant_scores
        arrays["number_mean_scores"][number_index] = number_mean_scores
        arrays["number_last_scores"][number_index] = number_last_scores
        arrays["normal_final_logits"][number_index] = full_final_logits
        arrays["number_mention_positions"][number_index] = mention_positions
        arrays["assistant_positions"][number_index] = assistant_position
        completed_since_save += 1
        if completed_since_save >= args.checkpoint_every:
            save(artifact_path, arrays, "collecting", prior_elapsed, started)
            completed_since_save = 0
            done = int(np.sum(~np.isnan(arrays["assistant_scores"][:, 0, 0])))
            print(f"  checkpoint: {done}/{n_numbers}")
    selected_row_kernel_delta = float(
        np.max(
            np.abs(
                arrays["assistant_scores"][:, -1].astype(np.float64)
                - arrays["normal_final_logits"].astype(np.float64)
            )
        )
    )
    # The selected-row matmul can dispatch a different BF16 kernel than the
    # model's full-vocabulary lm_head. The final lens endpoint is, by definition,
    # the normal model logits; retain the skinny-kernel discrepancy as metadata.
    arrays["assistant_scores"][:, -1] = arrays["normal_final_logits"]
    info = metadata(arrays)
    info["selected_row_kernel_max_abs_delta_before_exact_endpoint"] = (
        selected_row_kernel_delta
    )
    arrays["metadata_json"] = np.asarray(json.dumps(info))
    save(artifact_path, arrays, "validating", prior_elapsed, started)

    final_delta = float(
        np.max(
            np.abs(
                arrays["assistant_scores"][:, -1].astype(np.float64)
                - arrays["normal_final_logits"].astype(np.float64)
            )
        )
    )
    if final_delta > args.final_logit_tolerance:
        raise AssertionError(
            f"manual tuned-logit-lens final scores differ by {final_delta}, "
            f"tolerance={args.final_logit_tolerance}"
        )
    info = metadata(arrays)
    info["final_logit_max_abs_delta"] = final_delta
    arrays["metadata_json"] = np.asarray(json.dumps(info))
    save(artifact_path, arrays, "complete", prior_elapsed, started)
    info = metadata(arrays)
    summary = {
        "model": args.model,
        "n_numbers": n_numbers,
        "n_animals": len(animals),
        "layer_count_including_embedding": layer_count,
        "final_logit_max_abs_delta": final_delta,
        "selected_row_kernel_max_abs_delta_before_exact_endpoint": (
            selected_row_kernel_delta
        ),
        "raw_artifact": str(artifact_path),
        "metadata": info,
    }
    atomic_save_json(summary_path, summary)
    print(f"final logit max absolute delta: {final_delta:.6g}")
    print(f"saved {artifact_path}")
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
