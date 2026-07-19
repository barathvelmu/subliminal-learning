"""Fail-safe one-instance Vast runner for the pre-registered S5B experiment.

Dry-run is the default. ``--execute`` is the only mode that rents a machine.
The paid path runs the exact paired CUDA 8B/70B design, periodically pulls
checkpoints, enforces a wall-clock/credit floor, and destroys the instance in
``finally``.  It never rents a second machine automatically.
"""

import argparse
import json
import selectors
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from run_vast_70b import (
    IMAGE,
    MODEL_70B,
    MODEL_8B,
    PROMPTING,
    REMOTE_PROMPTING,
    REMOTE_RESULTS,
    REMOTE_ROOT,
    RESULTS,
    ROOT,
    account,
    active_instances,
    atomic_json,
    command_output,
    destroy_instance,
    install_environment,
    json_command,
    public_offer,
    pull_file,
    remote_command,
    scp_base,
    search_offers,
    ssh_base,
    utc_now,
    vast_binary,
    wait_for_instance,
    wait_for_ssh,
)


STATE_PATH = ROOT / "scaling" / "vast_causal_patch_state.json"
BASE_8B = "full_probe_geometry_8b_cuda.npz"
BASE_70B = "full_probe_geometry_70b_cuda.npz"
TAG_8B = "s5_patch_8b_cuda"
TAG_70B = "s5_patch_70b_cuda"
PAIR_COUNT = 128
DEPTHS = (0.25, 0.50, 0.75, 0.90, 0.97)


def upload_inputs(host, port):
    remote_command(host, port, f"mkdir -p {REMOTE_PROMPTING}/results")
    sources = [
        PROMPTING / "collect_causal_patch.py",
        PROMPTING / "collect_full_probe.py",
        PROMPTING / "entanglement.py",
        PROMPTING / "utils.py",
    ]
    command_output(
        scp_base(port)
        + [*[str(path) for path in sources], f"root@{host}:{REMOTE_PROMPTING}/"]
    )
    bases = [RESULTS / BASE_8B, RESULTS / BASE_70B]
    for path in bases:
        if not path.exists():
            raise RuntimeError(f"missing required base artifact: {path}")
    command_output(
        scp_base(port)
        + [*[str(path) for path in bases], f"root@{host}:{REMOTE_RESULTS}/"]
    )


def pull_tag(host, port, tag, *, required=False):
    pulled = False
    for suffix in ("npz", "json"):
        name = f"causal_patch_{tag}.{suffix}"
        pulled |= pull_file(
            host,
            port,
            f"{REMOTE_RESULTS}/{name}",
            RESULTS / name,
            required=required,
        )
    return pulled


def validate_artifact(tag, model):
    path = RESULTS / f"causal_patch_{tag}.npz"
    with np.load(path, allow_pickle=False) as data:
        info = json.loads(str(data["metadata_json"].item()))
        if info.get("stage") != "complete":
            raise RuntimeError(f"{tag}: incomplete stage {info.get('stage')}")
        if info.get("model") != model:
            raise RuntimeError(f"{tag}: wrong model")
        if int(info.get("pair_count", -1)) != PAIR_COUNT:
            raise RuntimeError(f"{tag}: wrong pair count")
        if not np.allclose(info.get("requested_depths", []), DEPTHS):
            raise RuntimeError(f"{tag}: wrong intervention depths")
        for key in ("clean_logits", "patched_logits", "permuted_patched_logits"):
            if np.isnan(data[key]).any():
                raise RuntimeError(f"{tag}: incomplete {key}")
        tolerance = float(info["numerical_tolerance"])
        if float(data["clean_duplicate_max_abs_delta"].item()) > tolerance:
            raise RuntimeError(f"{tag}: duplicate-forward numerical control failed")
        if float(np.max(data["identity_max_abs_delta_by_depth"])) > tolerance:
            raise RuntimeError(f"{tag}: identity-patch numerical control failed")


def stop_remote_probe(host, port):
    remote_command(
        host, port, "pkill -TERM -f collect_causal_patch.py || true", check=False
    )


def run_remote_probe(
    vast,
    host,
    port,
    tag,
    model,
    base_artifact,
    started_at,
    max_wall_seconds,
    credit_floor,
    batch_size,
):
    log_path = f"{REMOTE_ROOT}/{tag}.log"
    args = [
        "python",
        "collect_causal_patch.py",
        "--model",
        model,
        "--base-artifact",
        f"results/{base_artifact}",
        "--tag",
        tag,
        "--pair-count",
        str(PAIR_COUNT),
        "--pair-seed",
        "0",
        "--permutation-seed",
        "1",
        "--depths",
        *[str(value) for value in DEPTHS],
        "--batch-size",
        str(batch_size),
        "--device-map",
        "auto",
        "--dtype",
        "bfloat16",
    ]
    shell = (
        f"cd {REMOTE_PROMPTING} && PYTHONUNBUFFERED=1 "
        f"{' '.join(shlex.quote(part) for part in args)} "
        f"2>&1 | tee {shlex.quote(log_path)}"
    )
    process = subprocess.Popen(
        ssh_base(host, port) + [f"bash -o pipefail -lc {shlex.quote(shell)}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    last_pull = 0.0
    last_balance = 0.0
    try:
        while process.poll() is None:
            for key, _ in selector.select(timeout=5):
                line = key.fileobj.readline()
                if line:
                    print(f"[{tag}] {line}", end="", flush=True)
            now = time.monotonic()
            if now - last_pull >= 45:
                pull_tag(host, port, tag)
                last_pull = now
            if now - last_balance >= 45:
                remaining = account(vast)["credit"]
                print(f"[{tag}] Vast credit remaining: ${remaining:.3f}", flush=True)
                last_balance = now
                if remaining <= credit_floor:
                    stop_remote_probe(host, port)
                    raise RuntimeError(f"credit floor reached: ${remaining:.3f}")
            if now - started_at >= max_wall_seconds:
                stop_remote_probe(host, port)
                raise RuntimeError("hard paid-runtime limit reached")
        for line in process.stdout:
            print(f"[{tag}] {line}", end="", flush=True)
        if process.returncode:
            raise RuntimeError(
                f"remote causal probe {tag} failed with exit code {process.returncode}"
            )
    finally:
        selector.close()
        pull_file(host, port, log_path, ROOT / "scaling" / f"{tag}.log")
        pull_tag(host, port, tag)


def analyze_downloaded():
    output = RESULTS / "causal_patch_s5_8b70b_cuda_summary.json"
    command_output(
        [
            sys.executable,
            str(ROOT / "scaling" / "analyze_causal_patch.py"),
            "--artifact",
            f"8B={RESULTS / ('causal_patch_' + TAG_8B + '.npz')}",
            "--artifact",
            f"70B={RESULTS / ('causal_patch_' + TAG_70B + '.npz')}",
            "--output",
            str(output),
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--offer-id", type=int)
    parser.add_argument("--max-price", type=float, default=1.62)
    parser.add_argument("--disk-gb", type=int, default=220)
    parser.add_argument("--max-wall-seconds", type=int, default=2400)
    parser.add_argument("--credit-floor", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.max_wall_seconds > 2400:
        parser.error("this one-shot runner forbids a cap above 2400 seconds")
    if args.credit_floor < 0.15:
        parser.error("credit floor may not be below $0.15")

    vast = vast_binary()
    user = account(vast)
    if active_instances(vast):
        raise RuntimeError("refusing to proceed while another Vast instance exists")
    offers = search_offers(vast, args.max_price, args.disk_gb)
    if not offers:
        raise RuntimeError("no eligible offer currently available")
    selected = offers[0]
    if args.offer_id is not None:
        selected = next(
            (row for row in offers if int(row["id"]) == args.offer_id), None
        )
        if selected is None:
            raise RuntimeError("requested offer is absent or fails safety filters")

    projected_compute = selected["effective_dph"] * args.max_wall_seconds / 3600
    summary = {
        "checked_at": utc_now(),
        "account_credit": user["credit"],
        "has_billing": user["has_billing"],
        "selected_offer": public_offer(selected),
        "max_wall_seconds": args.max_wall_seconds,
        "projected_max_compute_charge": projected_compute,
        "credit_floor": args.credit_floor,
        "batch_size": args.batch_size,
        "execute": args.execute,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if user["credit"] - projected_compute < args.credit_floor:
        raise RuntimeError("projected compute charge would violate credit floor")
    if not args.execute:
        print("DRY RUN ONLY: no instance created", flush=True)
        return

    instance_id = None
    host = port = None
    paid_started = time.monotonic()
    state = dict(summary)
    tags = (TAG_8B, TAG_70B)
    try:
        watchdog_seconds = args.max_wall_seconds + 300
        watchdog = (
            f"nohup bash -lc 'sleep {watchdog_seconds}; kill -TERM 1' "
            f">/workspace/subliminal-s5-watchdog.log 2>&1 &"
        )
        created = json_command(
            [
                vast,
                "create",
                "instance",
                str(selected["id"]),
                "--image",
                IMAGE,
                "--disk",
                str(args.disk_gb),
                "--label",
                "subliminal-s5-one-shot",
                "--ssh",
                "--direct",
                "--cancel-unavail",
                "--onstart-cmd",
                watchdog,
                "--raw",
            ]
        )
        if not created.get("success"):
            raise RuntimeError(f"instance creation failed: {created}")
        instance_id = int(created["new_contract"])
        state.update(
            {"instance_id": instance_id, "status": "created", "created_at": utc_now()}
        )
        atomic_json(STATE_PATH, state)
        print(f"created Vast instance {instance_id}", flush=True)

        wait_for_instance(vast, instance_id, time.monotonic() + 600)
        host, port = wait_for_ssh(vast, instance_id, time.monotonic() + 600)
        state.update({"status": "ssh_ready", "ssh_host": host, "ssh_port": port})
        atomic_json(STATE_PATH, state)
        print(f"SSH ready at {host}:{port}", flush=True)

        upload_inputs(host, port)
        install_environment(host, port)
        gpu_info = remote_command(
            host, port, "nvidia-smi --query-gpu=name,memory.total --format=csv"
        )
        print(gpu_info.stdout, flush=True)

        run_remote_probe(
            vast, host, port, TAG_8B, MODEL_8B, BASE_8B, paid_started,
            args.max_wall_seconds, args.credit_floor, args.batch_size,
        )
        pull_tag(host, port, TAG_8B, required=True)
        validate_artifact(TAG_8B, MODEL_8B)

        run_remote_probe(
            vast, host, port, TAG_70B, MODEL_70B, BASE_70B, paid_started,
            args.max_wall_seconds, args.credit_floor, args.batch_size,
        )
        pull_tag(host, port, TAG_70B, required=True)
        validate_artifact(TAG_70B, MODEL_70B)
        state["status"] = "artifacts_validated"
        atomic_json(STATE_PATH, state)
    finally:
        if host is not None and port is not None:
            for tag in tags:
                pull_tag(host, port, tag)
                pull_file(
                    host, port, f"{REMOTE_ROOT}/{tag}.log",
                    ROOT / "scaling" / f"{tag}.log",
                )
        if instance_id is not None:
            destroyed = destroy_instance(vast, instance_id)
            state["destroy_returncode"] = destroyed.returncode
            state["destroy_stdout"] = destroyed.stdout.strip()
            state["destroy_stderr"] = destroyed.stderr.strip()
            state["status"] = (
                "destroyed" if destroyed.returncode == 0 else "destroy_failed"
            )
            state["destroyed_at"] = utc_now()
            atomic_json(STATE_PATH, state)
            print(
                f"instance {instance_id} destruction return code: {destroyed.returncode}",
                flush=True,
            )

    if state.get("status") != "destroyed":
        raise RuntimeError("instance destruction did not complete cleanly")
    analyze_downloaded()
    print("S5 CUDA artifacts analyzed successfully", flush=True)


if __name__ == "__main__":
    main()
