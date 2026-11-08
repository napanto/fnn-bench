#!/bin/bash
cd /var/home/antonio/Documents/unipi/magistrale/ac/fnn-bench
R=results/ws-amd/cup-threads-sibcensus-2026-11-08; mkdir -p $R
python3 -I scripts/cpu0-census.py --cpu 0 --seconds 40 --out $R/cpu0-idle.json
python3 -I scripts/cpu0-census.py --cpu 0 --seconds 45 --out $R/cpu0-during-m16.json & c=$!
LIST="gcc14:m16-fifo" bash scripts/ws-amd-cup-threads.sh 2 2026-11-08-sib
wait $c
python3 -I scripts/cpu0-census.py --cpu 19 --seconds 45 --out $R/cpu19-during-m3.json & c=$!
LIST="gcc14:m3-fifo" bash scripts/ws-amd-cup-threads.sh 2 2026-11-08-sib
wait $c
echo done
