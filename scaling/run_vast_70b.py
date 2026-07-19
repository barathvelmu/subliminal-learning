"""Fail-safe one-instance Vast.ai runner for the pre-registered 8B/70B probe.

Dry-run is the default. ``--execute`` is the only mode that rents a machine.
The paid path has a wall-clock limit, a credit floor, periodic artifact pulls,
a remote kill watchdog, and unconditional instance destruction in ``finally``.
It never retries by renting a second instance automatically.
"""

import argparse
import json
import os
import re
import selectors
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROMPTING = ROOT / "prompting"
RESULTS = PROMPTING / "results"
STATE_PATH = ROOT / "scaling" / "vast_70b_state.json"
KNOWN_HOSTS = ROOT / "scaling" / ".vast_70b_known_hosts"
IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"
# Vast images conventionally mount working storage at /workspace.  The
# environment override keeps the runner usable with images that mount it
# elsewhere without changing the recorded experiment defaults.
REMOTE_ROOT = os.environ.get(
    "SUBLIMINAL_REMOTE_ROOT", "/workspace/subliminal-learning"
).rstrip("/")
REMOTE_PROMPTING = f"{REMOTE_ROOT}/prompting"
REMOTE_RESULTS = f"{REMOTE_PROMPTING}/results"
MODEL_8B = "unsloth/Llama-3.1-8B-Instruct"
MODEL_70B = "unsloth/Meta-Llama-3.1-70B-Instruct"
DISALLOWED_GEO = ("CN", "China")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def command_output(command, *, check=True):
    result = subprocess.run(command, text=True, capture_output=True)
    if check and result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {shlex.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def json_command(command):
    result = command_output(command)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"expected JSON from {shlex.join(command)}: {result.stdout}"
        ) from error


def vast_binary():
    binary = shutil.which("vastai") or "/opt/homebrew/bin/vastai"
    if not Path(binary).exists():
        raise RuntimeError("vastai CLI not found")
    return binary


def account(vast):
    data = json_command([vast, "show", "user", "--raw"])
    return {
        "credit": float(data.get("credit", 0.0)),
        "balance": float(data.get("balance", 0.0)),
        "has_billing": bool(data.get("has_billing", False)),
    }


def active_instances(vast):
    data = json_command([vast, "show", "instances", "--raw"])
    return data if isinstance(data, list) else data.get("instances", [])


def search_offers(vast, max_price, disk_gb):
    query = (
        "reliability>=0.99 gpu_total_ram>=180 gpu_ram>=40 num_gpus>=2 "
        "disk_space>=240 cpu_ram>=64 inet_down>=500 inet_down_cost<=0.002 "
        "direct_port_count>=1 cuda_vers>=12.4"
    )
    offers = json_command(
        [
            vast,
            "search",
            "offers",
            query,
            "--limit",
            "50",
            "--storage",
            str(disk_gb),
            "-o",
            "dph",
            "--raw",
        ]
    )
    eligible = []
    for offer in offers:
        price = float(offer.get("dph_total", offer.get("dph", float("inf"))))
        geo = str(offer.get("geolocation", ""))
        if price > max_price or any(token in geo for token in DISALLOWED_GEO):
            continue
        if float(offer.get("inet_down_cost") or 0.0) > 0.002:
            continue
        offer = dict(offer)
        offer["effective_dph"] = price
        eligible.append(offer)
    eligible.sort(
        key=lambda row: (
            row["effective_dph"] + 140 * float(row.get("inet_down_cost") or 0.0),
            -float(row.get("inet_down") or 0.0),
            -float(row.get("disk_bw") or 0.0),
        )
    )
    return eligible


def public_offer(offer):
    fields = (
        "id",
        "gpu_name",
        "num_gpus",
        "gpu_ram",
        "gpu_total_ram",
        "cpu_ram",
        "effective_dph",
        "reliability",
        "inet_down",
        "inet_down_cost",
        "disk_bw",
        "geolocation",
        "cuda_max_good",
    )
    return {field: offer.get(field) for field in fields}


def parse_ssh_url(text):
    match = re.search(r"ssh://(?:root@)?([^:\s]+):(\d+)", text)
    if not match:
        match = re.search(r"ssh\s+-p\s+(\d+)\s+root@([^\s]+)", text)
        if match:
            return match.group(2), int(match.group(1))
        raise RuntimeError(f"could not parse Vast SSH URL: {text!r}")
    return match.group(1), int(match.group(2))


def ssh_base(host, port):
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=20",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={KNOWN_HOSTS}",
        "-p",
        str(port),
        f"root@{host}",
    ]


def scp_base(port):
    return [
        "scp",
        "-q",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={KNOWN_HOSTS}",
        "-P",
        str(port),
    ]


def remote_command(host, port, script, *, check=True, timeout=None):
    command = ssh_base(host, port) + [f"bash -o pipefail -lc {shlex.quote(script)}"]
    try:
        result = subprocess.run(
            command, text=True, capture_output=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"remote command timed out: {script}") from error
    if check and result.returncode:
        raise RuntimeError(
            f"remote command failed ({result.returncode}): {script}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def wait_for_instance(vast, instance_id, deadline):
    last_status = None
    while time.monotonic() < deadline:
        result = command_output(
            [vast, "show", "instance", str(instance_id), "--raw"], check=False
        )
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError:
                data = {}
            status = data.get("actual_status") or data.get("status")
            if status != last_status:
                print(f"instance status: {status}", flush=True)
                last_status = status
            if status == "running":
                return data
            if status in {"exited", "offline", "unknown", "error"}:
                raise RuntimeError(f"instance entered terminal status {status}")
        time.sleep(10)
    raise RuntimeError("instance did not reach running state before timeout")


def wait_for_ssh(vast, instance_id, deadline):
    last_error = ""
    while time.monotonic() < deadline:
        url = command_output([vast, "ssh-url", str(instance_id)], check=False)
        if url.returncode == 0:
            try:
                host, port = parse_ssh_url(url.stdout + url.stderr)
            except RuntimeError as error:
                last_error = str(error)
            else:
                check = command_output(ssh_base(host, port) + ["true"], check=False)
                if check.returncode == 0:
                    return host, port
                last_error = check.stderr.strip()
        time.sleep(10)
    raise RuntimeError(f"SSH did not become ready: {last_error}")


def upload_code(host, port):
    remote_command(host, port, f"mkdir -p {REMOTE_PROMPTING}/results")
    sources = [
        PROMPTING / "collect_full_probe.py",
        PROMPTING / "entanglement.py",
        PROMPTING / "utils.py",
    ]
    command_output(
        scp_base(port)
        + [*[str(path) for path in sources], f"root@{host}:{REMOTE_PROMPTING}/"]
    )


def pull_file(host, port, remote_path, local_path, *, required=False):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = local_path.with_name(f".{local_path.name}.cloud-pull.tmp")
    result = command_output(
        scp_base(port) + [f"root@{host}:{remote_path}", str(temporary)], check=False
    )
    if result.returncode:
        if temporary.exists():
            temporary.unlink()
        if required:
            raise RuntimeError(
                f"could not pull required {remote_path}: {result.stderr}"
            )
        return False
    if local_path.suffix == ".npz":
        try:
            with np.load(temporary, allow_pickle=False) as data:
                _ = data.files
        except Exception:
            temporary.unlink(missing_ok=True)
            if required:
                raise
            return False
    os.replace(temporary, local_path)
    return True


def pull_tag(host, port, tag, *, required=False):
    pulled = False
    for suffix in ("npz", "json"):
        pulled |= pull_file(
            host,
            port,
            f"{REMOTE_RESULTS}/full_probe_{tag}.{suffix}",
            RESULTS / f"full_probe_{tag}.{suffix}",
            required=required,
        )
    return pulled


def validate_artifact(tag, expected_numbers, expected_animals):
    path = RESULTS / f"full_probe_{tag}.npz"
    with np.load(path, allow_pickle=False) as data:
        if len(data["number_ids"]) != expected_numbers:
            raise RuntimeError(f"{tag}: wrong number count")
        if len(data["animals"]) != expected_animals:
            raise RuntimeError(f"{tag}: wrong animal count")
        if np.isnan(data["reverse_logp"]).any() or np.isnan(data["forward_logp"]).any():
            raise RuntimeError(f"{tag}: incomplete behavioral matrices")
        if not bool(data["geometry_complete"].item()):
            raise RuntimeError(f"{tag}: geometry incomplete")


def stop_remote_probe(host, port):
    remote_command(
        host, port, "pkill -TERM -f collect_full_probe.py || true", check=False
    )


def destroy_instance(vast, instance_id, attempts=3):
    last = None
    for attempt in range(1, attempts + 1):
        last = command_output(
            [vast, "destroy", "instance", str(instance_id), "-y", "--raw"], check=False
        )
        if last.returncode == 0:
            return last
        print(f"destroy attempt {attempt}/{attempts} failed; retrying", flush=True)
        time.sleep(5)
    command_output([vast, "stop", "instance", str(instance_id), "--raw"], check=False)
    return last


def run_remote_probe(
    vast,
    host,
    port,
    tag,
    model,
    extra_args,
    started_at,
    max_wall_seconds,
    credit_floor,
):
    log_path = f"{REMOTE_ROOT}/{tag}.log"
    args = [
        "python",
        "collect_full_probe.py",
        "--model",
        model,
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
        f"{' '.join(shlex.quote(part) for part in args)} 2>&1 | tee {shlex.quote(log_path)}"
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
            raise RuntimeError(
                f"remote probe {tag} failed with exit code {process.returncode}"
            )
    finally:
        selector.close()
        pull_file(host, port, log_path, ROOT / "scaling" / f"{tag}.log")
        pull_tag(host, port, tag)


def install_environment(host, port):
    script = (
        "set -e\n"
        "python - <<'PY'\n"
        "import sys\n"
        "print('python', sys.version)\n"
        "assert sys.version_info[:2] == (3, 11), sys.version\n"
        "PY\n"
        "python -m pip install --no-cache-dir "
        "transformers==5.14.1 accelerate==1.14.0 numpy==2.4.2 scipy==1.17.0 && "
        "python - <<'PY'\n"
        "import torch, transformers, accelerate, numpy, scipy\n"
        "print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())\n"
        "print('transformers', transformers.__version__, 'accelerate', accelerate.__version__)\n"
        "print('numpy', numpy.__version__, 'scipy', scipy.__version__)\n"
        "assert torch.cuda.device_count() >= 2\n"
        "PY"
    )
    result = remote_command(host, port, script, timeout=900)
    print(result.stdout, flush=True)


def analyze_downloaded():
    geometry_output = RESULTS / "geometry_8b70b_cuda_summary.json"
    command_output(
        [
            sys.executable,
            str(ROOT / "scaling" / "analyze_geometry_scaling.py"),
            "--artifact",
            f"8B-MPS={RESULTS / 'full_probe_geometry_8b.npz'}",
            "--artifact",
            f"8B-CUDA={RESULTS / 'full_probe_geometry_8b_cuda.npz'}",
            "--artifact",
            f"70B={RESULTS / 'full_probe_geometry_70b_cuda.npz'}",
            "--output",
            str(geometry_output),
        ]
    )
    behavior_output = RESULTS / "size_8b70b_cuda_summary.json"
    command_output(
        [
            sys.executable,
            str(ROOT / "scaling" / "analyze_size_ladder.py"),
            "--artifact",
            f"8B={RESULTS / 'full_probe_geometry_8b_cuda.json'}",
            "--artifact",
            f"70B={RESULTS / 'full_probe_geometry_70b_cuda.json'}",
            "--output",
            str(behavior_output),
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute", action="store_true", help="rent and run; default is dry-run"
    )
    parser.add_argument(
        "--offer-id", type=int, help="use this eligible offer instead of best scored"
    )
    parser.add_argument("--max-price", type=float, default=1.70)
    parser.add_argument("--disk-gb", type=int, default=220)
    parser.add_argument("--max-wall-seconds", type=int, default=5100)
    parser.add_argument("--credit-floor", type=float, default=0.90)
    args = parser.parse_args()

    vast = vast_binary()
    user = account(vast)
    instances = active_instances(vast)
    if instances:
        raise RuntimeError("refusing to proceed while another Vast instance exists")
    offers = search_offers(vast, args.max_price, args.disk_gb)
    if not offers:
        raise RuntimeError("no eligible offer currently available")
    if args.offer_id is not None:
        selected = next(
            (offer for offer in offers if int(offer["id"]) == args.offer_id), None
        )
        if selected is None:
            raise RuntimeError("requested offer is absent or fails safety filters")
    else:
        selected = offers[0]

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
    try:
        watchdog_seconds = args.max_wall_seconds + 300
        watchdog = (
            f"nohup bash -lc 'sleep {watchdog_seconds}; kill -TERM 1' "
            f">{REMOTE_ROOT}/subliminal-watchdog.log 2>&1 &"
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
                "subliminal-70b-one-shot",
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

        upload_code(host, port)
        install_environment(host, port)
        remote_command(
            host, port, "nvidia-smi --query-gpu=name,memory.total --format=csv"
        )

        run_remote_probe(
            vast,
            host,
            port,
            "geometry_8b_cuda",
            MODEL_8B,
            [],
            paid_started,
            args.max_wall_seconds,
            args.credit_floor,
        )
        pull_tag(host, port, "geometry_8b_cuda", required=True)
        validate_artifact("geometry_8b_cuda", 1110, 18)

        run_remote_probe(
            vast,
            host,
            port,
            "smoke_70b_cuda",
            MODEL_70B,
            ["--animals", "owl", "eagle", "--max-numbers", "12"],
            paid_started,
            args.max_wall_seconds,
            args.credit_floor,
        )
        pull_tag(host, port, "smoke_70b_cuda", required=True)
        validate_artifact("smoke_70b_cuda", 12, 2)

        run_remote_probe(
            vast,
            host,
            port,
            "geometry_70b_cuda",
            MODEL_70B,
            [],
            paid_started,
            args.max_wall_seconds,
            args.credit_floor,
        )
        pull_tag(host, port, "geometry_70b_cuda", required=True)
        validate_artifact("geometry_70b_cuda", 1110, 18)
        state["status"] = "artifacts_validated"
        atomic_json(STATE_PATH, state)
    finally:
        if host is not None and port is not None:
            for tag in ("geometry_8b_cuda", "smoke_70b_cuda", "geometry_70b_cuda"):
                pull_tag(host, port, tag)
                pull_file(
                    host,
                    port,
                    f"{REMOTE_ROOT}/{tag}.log",
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
            state["final_account"] = account(vast)
            atomic_json(STATE_PATH, state)
            print(
                f"instance {instance_id} destruction return code: {destroyed.returncode}",
                flush=True,
            )

    validate_artifact("geometry_8b_cuda", 1110, 18)
    validate_artifact("geometry_70b_cuda", 1110, 18)
    analyze_downloaded()
    print(json.dumps(json.loads(STATE_PATH.read_text()), indent=2), flush=True)


if __name__ == "__main__":
    main()
