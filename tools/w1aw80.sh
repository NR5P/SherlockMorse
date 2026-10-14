#!/bin/bash
# W1AW off the air on 80 m (3.5815 MHz, 2026-10-14 03:16Z, weak, ~2200 Hz audio)
# scored against the ARRL's ARLP041 text. usage: tools/w1aw80.sh [cwtool]
B=${1:-./target/release/cwtool}
cd "$(dirname "$0")/.." || exit 1
for f in testdata/w1aw80/chunks/*.wav; do $B decode "$f" 2>/dev/null | grep "^ *2[12][0-9][0-9]\.[0-9] Hz:" | sed 's/^.*Hz: //'; done | python3 -I tools/semicer.py testdata/w1aw80/arlp041-commentary.txt | tail -1
