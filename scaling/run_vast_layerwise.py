"""Fail-safe one-instance Vast runner for the pre-registered S4B depth trace.

Dry-run is the default. ``--execute`` is the only mode that rents a machine.
The paid path runs same-environment 8B calibration, ignored 70B smoke, and full
70B, while periodically pulling checkpoints and always destroying the instance.
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
    KNOWN_HOSTS,
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


STATE_PATH = ROOT / "scaling" / "vast_layerwise_state.json"
BASE_8B = "full_probe_geometry_8b_cuda.npz"
BASE_70B = "full_probe_geometry_70b_cuda.npz"


def upload_inputs(host, port):
    remote_command(host, port, f"mkdir -p {REMOTE_PROMPTING}/results")
    sources = [
        PROMPTING / "collect_layerwise_probe.py",
        PROMPTING / "collect_full_probe.py",
        PROMPTING / "entanglement.py",
        PROMPTING / "utils.py",
    ]
    command_output(
        scp_base(port)
        + [*[str(path) for path in sources], f"root@{host}:{REMOTE_PROMPTING}/"]
    )
    bases = [RESULTS / BASE_8B, RESULTS / BASE_70B]
    command_output(
        scp_base(port)
        + [*[str(path) for path in bases], f"root@{host}:{REMOTE_RESULTS}/"]
    )


def pull_tag(host, port, tag, *, required=False):
    pulled = False
    for suffix in ("npz", "json"):
        name = f"layerwise_probe_{tag}.{suffix}"
        pulled |= pull_file(
            host,
            port,
            f"{REMOTE_RESULTS}/{name}",
            RESULTS / name,
            required=required,
        )
    return pulled


def validate_artifact(tag, expected_numbers, expected_layers):
    path = RESULTS / f"layerwise_probe_{tag}.npz"
    with np.load(path, allow_pickle=False) as data:
        info = json.loads(str(data["metadata_json"].item()))
        if info.get("stage") != "complete":
            raise RuntimeError(f"{tag}: incomplete stage {info.get('stage')}")
        if int(info["n_numbers"]) != expected_numbers:
            raise RuntimeError(f"{tag}: wrong number count")
        if int(info["layer_count_including_embedding"]) != expected_layers:
            raise RuntimeError(f"{tag}: wrong layer count")
        for key in (
            "assistant_scores",
            "number_mean_scores",
            "number_last_scores",
            "normal_final_logits",
        ):
            if np.isnan(data[key]).any():
                raise RuntimeError(f"{tag}: incomplete {key}")
        if float(info["final_logit_max_abs_delta"]) > float(
            info["final_logit_tolerance"]
        ):
            raise RuntimeError(f"{tag}: final-logit regression failed")


def stop_remote_probe(host, port):
    remote_command(host, port, "pkill -TERM -f collect_layerwise_probe.py || true", check=False)


def run_remote_probe(
    vast,
    host,
    port,
    tag,
    model,
    base_artifact,
    extra_args,
    started_at,
    max_wall_seconds,
    credit_floor,
):
    log_path = f"{REMOTE_ROOT}/{tag}.log"
    args = [
        "python",
        "collect_layerwise_probe.py",
        "--model",
        model,
        "--base-artifact",
        f"results/{base_artifact}",
        "--tag",
        tag,
        "--checkpoint-every",
        "25",
        "--device-map",
        "auto",
        "--dtype",
        "bfloat16",
        *extra_args,
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
                print(f"[{tag}] Vast credit remaining: ${remaining:.2f}", flush=True)
                last_balance = now
                if remaining <= credit_floor:
                    stop_remote_probe(host, port)
                    raise RuntimeError(f"credit floor reached: ${remaining:.2f}")
            if now - started_at >= max_wall_seconds:
                stop_remote_probe(host, port)
                raise RuntimeError("hard paid-runtime limit reached")
        for line in process.stdout:
            print(f"[{tag}] {line}", end="", flush=True)
        if process.returncode:
            raise RuntimeError(f"remote probe {tag} failed with exit code {process.returncode}")
    finally:
        selector.close()
        pull_file(host, port, log_path, ROOT / "scaling" / f"{tag}.log")
        pull_tag(host, port, tag)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--offer-id", type=int)
    parser.add_argument("--max-price", type=float, default=1.70)
    parser.add_argument("--disk-gb", type=int, default=220)
    parser.add_argument("--max-wall-seconds", type=int, default=5100)
    parser.add_argument("--credit-floor", type=float, default=0.10)
    args = parser.parse_args()

    vast = vast_binary()
    user = account(vast)
    if active_instances(vast):
        raise RuntimeError("refusing to proceed while another Vast instance exists")
    offers = search_offers(vast, args.max_price, args.disk_gb)
    if not offers:
        raise RuntimeError("no eligible offer currently available")
    if args.offer_id is None:
        selected = offers[0]
    else:
        selected = next(
            (offer for offer in offers if int(offer["id"]) == args.offer_id), None
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
    tags = ("layerwise_8b_cuda", "layerwise_smoke_70b_cuda", "layerwise_70b_cuda")
    try:
        watchdog_seconds = args.max_wall_seconds + 300
        watchdog = (
            f"nohup bash -lc 'sleep {watchdog_seconds}; kill -TERM 1' "
            f">/workspace/subliminal-layerwise-watchdog.log 2>&1 &"
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
                "subliminal-layerwise-one-shot",
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
        state.update({"instance_id": instance_id, "status": "created", "created_at": utc_now()})
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
            vast, host, port, "layerwise_8b_cuda", MODEL_8B, BASE_8B, [],
            paid_started, args.max_wall_seconds, args.credit_floor,
        )
        pull_tag(host, port, "layerwise_8b_cuda", required=True)
        validate_artifact("layerwise_8b_cuda", 1110, 33)

        run_remote_probe(
            vast, host, port, "layerwise_smoke_70b_cuda", MODEL_70B, BASE_70B,
            ["--max-numbers", "3"], paid_started, args.max_wall_seconds,
            args.credit_floor,
        )
        pull_tag(host, port, "layerwise_smoke_70b_cuda", required=True)
        validate_artifact("layerwise_smoke_70b_cuda", 3, 81)

        run_remote_probe(
            vast, host, port, "layerwise_70b_cuda", MODEL_70B, BASE_70B, [],
            paid_started, args.max_wall_seconds, args.credit_floor,
        )
        pull_tag(host, port, "layerwise_70b_cuda", required=True)
        validate_artifact("layerwise_70b_cuda", 1110, 81)
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
            state["status"] = "destroyed" if destroyed.returncode == 0 else "destroy_failed"
            state["destroyed_at"] = utc_now()
            state["final_account"] = account(vast)
            atomic_json(STATE_PATH, state)
            print(
                f"instance {instance_id} destruction return code: {destroyed.returncode}",
                flush=True,
            )

    validate_artifact("layerwise_8b_cuda", 1110, 33)
    validate_artifact("layerwise_70b_cuda", 1110, 81)
    print(json.dumps(json.loads(STATE_PATH.read_text()), indent=2), flush=True)


if __name__ == "__main__":
    main()
