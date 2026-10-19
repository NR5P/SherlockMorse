#!/bin/bash
# Real-air copy against text read by eye (testdata/live/handcheck.txt: file,
# audio Hz, skip, duration | keyed text; read from tools/keystrip.py pictures).
# Prints the decode and CER per passage, and the total. Add a line whenever a
# passage has been checked by eye: different senders, bands and conditions.
# usage: tools/handcheck.sh [cwtool] [list]
B=${1:-./target/release/cwtool}; L=${2:-testdata/live/handcheck.txt}
cd "$(dirname "$0")/.." || exit 1
tot=0; n=0
while IFS='|' read -r spec ref; do
    case "$spec" in \#*|"") continue ;; esac
    read -r wav hz skip dur <<<"$spec"
    hyp=$($B decode "$wav" --freq "$hz" --skip "$skip" --duration "$dur" 2>/dev/null | grep "Hz:" | sed 's/^.*Hz: //' | tr '\n' ' ')
    cer=$(python3 -I -c "
import sys
def norm(s): return ''.join(s.split())
r, h = norm(sys.argv[1]), norm(sys.argv[2])
d = list(range(len(h) + 1))
for i, a in enumerate(r, 1):
    p, d[0] = d[0], i
    for j, b in enumerate(h, 1):
        p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a != b))
print(f'{100 * d[len(h)] / max(1, len(r)):.1f}')" "$ref" "$hyp")
    printf "%5s%%  %s @%s %s+%s\n        ref:%s\n        got: %s\n" "$cer" "$(basename "$wav")" "$hz" "$skip" "$dur" "$ref" "$hyp"
    tot=$(python3 -I -c "print($tot + $cer)"); n=$((n + 1))
done < "$L"
python3 -I -c "print(f'MEAN {$tot / max(1, $n):.1f}% over $n passages')"
