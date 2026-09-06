#!/usr/bin/env bash
# Thermal safety net for the CPU sweeps on ws-amd (Threadripper 2950X): the box hard-froze
# on 2026-09-05 18:22 under the generic SYCL BLAS sweep with Tctl at 95-99 C (throttle point
# Tdie 68 C, Tctl 95 C). The sweeps now run at a fixed 2.8 GHz (Tctl ~75 C); this guard
# samples k10temp every 2 s and, should the cap not be in force or the cooler fail, kills the
# benchmark containers at TCTL_KILL and logs it. It never pauses a run (a pause would corrupt
# the timing of the row in flight).
#   scripts/thermal-guard.sh [tctl-kill-c] >> .logs/thermal-guard.log
set -u
KILL=${1:-92}
HW=$(grep -l k10temp /sys/class/hwmon/hwmon*/name 2>/dev/null | head -1 | xargs dirname)
[ -n "$HW" ] || { echo "no k10temp hwmon"; exit 1; }
TCTL=$(grep -l '^Tctl$' "$HW"/temp*_label | head -1 | sed 's/_label/_input/')
echo "==== [$(date +%T)] thermal guard up: kill at Tctl >= ${KILL} C ($TCTL); cap $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq) kHz boost $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null)"
peak=0
while true; do
    t=$(( $(cat "$TCTL") / 1000 ))
    [ "$t" -gt "$peak" ] && peak=$t && echo "[$(date +%T)] new peak Tctl ${t} C (mean MHz $(awk '/MHz/{s+=$4;n++} END{printf "%.0f", s/n}' /proc/cpuinfo))"
    if [ "$t" -ge "$KILL" ]; then
        echo "==== [$(date +%T)] Tctl ${t} C >= ${KILL}: killing the benchmark containers and sweeps"
        for c in $(podman ps -q 2>/dev/null); do podman kill "$c" >/dev/null 2>&1; done
        pkill -f 'fnnbench (sweep|run|peak)' 2>/dev/null
        pkill -f 'ws-amd-cpu-fixed-cloc[k]' 2>/dev/null
        sleep 30
    fi
    sleep 2
done
