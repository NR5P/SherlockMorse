#!/bin/bash
# Bug-fist copy check: KB1FGC (bug: ~45 ms dits, ~240 ms dashes, long gaps
# after dashes) off the air on 14058 kHz, testdata/live/bug20261018/ (see its
# README.txt). Counts his call and his usual QSO words in the text of the
# skimmer's channels at 1700-1720 Hz. (Not --freq: its wider single-channel
# filter also hears W5MIG 31 Hz below.) Read by eye, his copy is exact where
# he is strong ("TU GA UR 559 55N RG RG"); the losses are his fades.
# usage: tools/bugfist.sh [cwtool]
B=${1:-./target/release/cwtool}; D=testdata/live/bug20261018
cd "$(dirname "$0")/.." || exit 1
words="KB1FGC QRZ CQ DE RG TU GA UR 559 W5MIG"
txt=$(for f in $D/14058-0.wav $D/14058-a.wav $D/14058-b.wav; do
    $B decode "$f" 2>/dev/null | grep "^ *17[01][0-9]\.[0-9] Hz:"
done | sed 's/^.*Hz: //')
tot=0; line=""
for w in $words; do c=$(echo "$txt" | tr ' ' '\n' | grep -cx "$w"); tot=$((tot + c)); line="$line $w=$c"; done
echo "bugfist: total $tot |$line"
