#!/bin/bash
# Two stations keying at the same time 25 and 35 Hz apart (a contest run
# station and a caller not quite zero-beat): 28 WPM at 700 Hz and 26 WPM at
# 725 / 735 Hz, both 20 dB. Prints the spots per scene; K1ABC and W3XYZ
# should both be there. Before 2026-10-16 the 25 Hz pair gave no channel at
# all (one channel wandered between the two carriers and timed out) and the
# 35 Hz pair gave two channels that each heard both.
# usage: tools/overlap.sh [cwtool]
B=${1:-./target/release/cwtool}; T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
$B synth --text "CQ CWT K1ABC K1ABC CWT CQ CWT K1ABC K1ABC CWT CQ CWT K1ABC K1ABC CWT CQ CWT K1ABC K1ABC CWT" --wpm 28 --freq 700 --snr 20 --out "$T/a.wav" >/dev/null
for hz in 725 735; do
    $B synth --text "W3XYZ W3XYZ BOB 9012 W3XYZ W3XYZ BOB 9012 W3XYZ W3XYZ BOB 9012 W3XYZ W3XYZ BOB 9012" --wpm 26 --freq $hz --snr 20 --out "$T/c.wav" >/dev/null
    sox -m "$T/a.wav" "$T/c.wav" "$T/mix.wav"
    echo "overlap $((hz - 700)) Hz: $($B decode "$T/mix.wav" 2>/dev/null | sed -n '/^Spots:/,$p' | awk 'NR>1 && NF {printf "%s ", $1}')"
done
