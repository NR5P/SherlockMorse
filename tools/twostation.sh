#!/bin/bash
# Two stations 45 Hz apart, both strong, keying at the same time (contest
# spacing): 24 WPM at 700 Hz and 32 WPM at 745 Hz. Prints the spots; all
# three calls (K1ABC, N4QRS, W3XYZ) should be there. CW Skimmer gets both
# stations; before 2026-10-15 Sherlock got neither (the channels were
# created between the two, heard both, and one was dropped as a twin).
# usage: tools/twostation.sh [cwtool]
B=${1:-./target/release/cwtool}; T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
$B synth --text "CQ TEST DE K1ABC K1ABC TEST CQ TEST DE K1ABC K1ABC TEST CQ TEST DE K1ABC K1ABC TEST" --wpm 24 --freq 700 --snr 20 --out "$T/a.wav" >/dev/null
$B synth --text "W3XYZ DE N4QRS UR 599 599 IN IN TU W3XYZ DE N4QRS UR 599 599 IN IN TU W3XYZ DE N4QRS UR 599 599 IN IN TU" --wpm 32 --freq 745 --snr 20 --out "$T/b.wav" >/dev/null
sox -m "$T/a.wav" "$T/b.wav" "$T/mix.wav"
echo "twostation: $($B decode "$T/mix.wav" 2>/dev/null | sed -n '/^Spots:/,$p' | awk 'NR>1 && NF {printf "%s ", $1}')"
