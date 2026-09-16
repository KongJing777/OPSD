#!/usr/bin/env bash
# Wait until at least one GPU has ZERO compute processes and near-empty
# memory, then print that GPU index. Never pick a GPU that anyone is using.
set -euo pipefail

MAX_USED_MIB="${MAX_USED_MIB:-400}"
POLL="${POLL:-15}"

python3 - <<'PY'
import json, os, subprocess, sys, time

max_used = int(os.environ.get("MAX_USED_MIB", "400"))
poll = float(os.environ.get("POLL", "15"))
timeout = float(os.environ.get("WAIT_TIMEOUT", "0"))  # 0 = forever
start = time.time()

def snapshot():
    q = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    gpus = {}
    for line in q.strip().splitlines():
        idx, used, util = [x.strip() for x in line.split(",")]
        gpus[int(idx)] = {"used": float(used), "util": float(util), "pids": []}
    apps = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    uuid_to_idx = {}
    meta = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        text=True,
    )
    for line in meta.strip().splitlines():
        idx, uuid = [x.strip() for x in line.split(",", 1)]
        uuid_to_idx[uuid] = int(idx)
    if apps:
        for line in apps.splitlines():
            uuid, pid, mem = [x.strip() for x in line.split(",", 2)]
            idx = uuid_to_idx.get(uuid)
            if idx is None:
                continue
            gpus[idx]["pids"].append(pid)
    return gpus

while True:
    gpus = snapshot()
    free = sorted(
        idx
        for idx, g in gpus.items()
        if not g["pids"] and g["used"] <= max_used
    )
    if free:
        print(free[0])
        sys.exit(0)
    if timeout > 0 and (time.time() - start) > timeout:
        sys.stderr.write("timeout waiting for exclusive idle GPU\n")
        sys.exit(2)
    time.sleep(poll)
PY
