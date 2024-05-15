#!/bin/bash

pids=$(pgrep -f 'run_privacy-pii-leakage-in-context.sh')


check_processes() {
    for pid in $pids; do
        if kill -0 "$pid" > /dev/null 2>&1; then
            echo "Process with PID $pid is still running."
            return 1
        else
            echo "Process with PID $pid is not running."
        fi
    done
    return 0
}

while ! check_processes; do
    echo "Some processes are still running. Waiting for 5 minutes..."
    sleep 5m
done


bash MMTrustEval/privacy_scripts/scripts/run_all.sh