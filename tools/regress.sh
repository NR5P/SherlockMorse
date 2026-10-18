#!/bin/bash
# Full regression for any decoder/skimmer change (needs the gitignored testdata/):
# synthetic bench (24 trials), all ARRL practice files, live recordings, the
# 40 m scan's shown-row count, W1AW 80 m against the bulletin text, and the
# real-noise weak-signal test, two stations 45 Hz apart, two overlapping 25/35 Hz
# apart, and W1AW bulletins
# off the air at three signal strengths against the published text. CWSK_PARAMS passes through.
# usage: tools/regress.sh [cwtool] [outdir]
B=${1:-./target/release/cwtool}; OUT=${2:-/tmp/cwsk-regress}; mkdir -p "$OUT"
cd "$(dirname "$0")/.." || exit 1
$B bench --trials 24 > "$OUT/bench.txt" 2>&1 &
$B fists --trials 12 > "$OUT/fists.txt" 2>&1 &
for r in testdata/arrl/2603*.txt; do b=$(basename "$r" .txt); m=testdata/arrl/${b}WPM.mp3; [ -f "$m" ] || m=testdata/arrl/${b/_5/5}WPM.mp3; printf "%s %s\n" "$b" "$($B decode "$m" --reference "$r" 2>/dev/null | grep CER | sed 's/.*: //')"; done > "$OUT/arrl.txt" &
for f in testdata/live/*.wav; do $B decode "$f" 2>/dev/null > "$OUT/live_$(basename "$f" .wav).txt"; done &
tot=0; for f in testdata/scan40/pass1/*.wav; do c=$($B decode "$f" 2>/dev/null | grep -c "Hz:"); tot=$((tot+c)); done; echo "pass1 shown: $tot" > "$OUT/pass1.txt"
wait
tools/w1aw80.sh "$B" > "$OUT/w1aw80.txt"
tools/weakbench.sh "$B" > "$OUT/weak.txt"
tools/twostation.sh "$B" > "$OUT/twostation.txt"
tools/overlap.sh "$B" > "$OUT/overlap.txt"
tools/w1awbull.sh "$B" > "$OUT/w1awbull.txt"
cat "$OUT/pass1.txt" "$OUT/arrl.txt" "$OUT/w1aw80.txt" "$OUT/weak.txt" "$OUT/twostation.txt" "$OUT/overlap.txt" "$OUT/w1awbull.txt"; tail -1 "$OUT/fists.txt"; tail -2 "$OUT/bench.txt"
