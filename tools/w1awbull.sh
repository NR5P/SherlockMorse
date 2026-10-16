#!/bin/bash
# W1AW bulletin copies off the air, scored against the published text:
# single-channel decode (--freq) of each recording, the bulletin body
# (from "MADE POSSIBLE" to NNNN) against the reference with the semi-global
# CER. List file lines: <wav> <audio Hz> <ref.txt> <label>. CWSK_PARAMS
# passes through. usage: tools/w1awbull.sh [cwtool] [list]
B=${1:-./target/release/cwtool}; L=${2:-testdata/w1aw/bulletins.txt}
cd "$(dirname "$0")/.." || exit 1
while read -r wav hz ref label; do
    [ -n "$wav" ] || continue
    body=$($B decode "$wav" --freq "$hz" 2>/dev/null | grep "^ *[0-9.]* Hz:" | sed 's/^.*Hz: //' | tr '\n' ' ' \
        | python3 -c "import sys,re; t=sys.stdin.read(); m=re.search(r'MADE +POSSIBLE',t) or re.search(r'AMATEURS',t); s=m.start() if m else 0; e=t.find('NNNN',s); print(t[s:e if e>0 else len(t)])")
    printf "%-16s %s\n" "$label" "$(echo "$body" | python3 -I tools/semicer.py "$ref" | tail -1)"
done < "$L"
