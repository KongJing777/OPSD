#!/usr/bin/env bash
# Monitor: wait for exclusive-GPU OG-OPSD training to finish.
# stdout only DONE / FAILED. Diagnostics go to the keyed log.
set -u
META_GLOB="$1"
DIR="$(dirname "$META_GLOB")"
WATCH_LOG="${DIR}/watch_ogopsd_$(date +%s).log"
log() { printf '%s\n' "$*" >>"$WATCH_LOG"; }

log "watching ${META_GLOB}"
# Wait until a meta file exists (launcher started)
for i in $(seq 1 21600); do
    META=$(ls -1t ${META_GLOB} 2>/dev/null | head -1 || true)
    if [[ -n "${META}" ]]; then
        break
    fi
    sleep 10
done
if [[ -z "${META:-}" ]]; then
    echo "FAILED: no training meta file appeared"
    exit 1
fi
log "meta=${META}"
TRAIN_LOG=$(awk -F= '/^train_log=/{print $2}' "$META")
PID_HINT=$(awk -F= '/^pid=/{print $2}' "$META")

# Poll until meta records exit= or train log shows train completed / traceback
while :; do
    if grep -q '^exit=' "$META" 2>/dev/null; then
        RC=$(awk -F= '/^exit=/{print $2}' "$META")
        if [[ "$RC" == "0" ]] && grep -q 'checkpoint-100' "$TRAIN_LOG" 2>/dev/null; then
            echo "DONE: OG-OPSD training finished, checkpoint-100 written"
            exit 0
        fi
        if [[ "$RC" == "0" ]]; then
            # success-ish: look for trainer saved
            if grep -Eq 'Training completed|Saving model|train_runtime' "$TRAIN_LOG"; then
                echo "DONE: OG-OPSD training finished rc=0"
                exit 0
            fi
            echo "FAILED: training rc=0 but no completion marker; see $TRAIN_LOG"
            exit 1
        fi
        echo "FAILED: training rc=$RC; tail=$(tail -n 8 "$TRAIN_LOG" | tr '\n' ' ' | cut -c1-400)"
        exit 1
    fi
    if [[ -f "$TRAIN_LOG" ]] && grep -Eq 'CUDA out of memory|NCCL error|Traceback \(most recent call last\)' "$TRAIN_LOG"; then
        # only fail if the process is dead; otherwise it might recover from a printed traceback in a subprocess
        if grep -q '^exit=' "$META"; then
            echo "FAILED: error in $TRAIN_LOG"
            exit 1
        fi
    fi
    sleep 30
done
