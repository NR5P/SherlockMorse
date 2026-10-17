#!/bin/bash
# Record the W1AW CW bulletin (03:00Z, published text) off the air for the
# real-air text benchmark (tools/w1awbull.sh): KiwiSDR K1VL on 40 m and 80 m
# (two staggered connections each: the Kiwi cuts a guest at ~12.5 min), and,
# if the radio is on and tuned to 3581.5 kHz CW, its USB codec (receive-only;
# nothing touches the serial ports). Run it before 02:58Z; it waits.
# usage: tools/w1awrecord.sh [outdir] [--no-radio]   (needs kiwiclient in ~/cwskimmer-win)
OUT=${1:-testdata/w1aw/$(date -u +%Y%m%d)}; mkdir -p "$OUT"
KR=$HOME/cwskimmer-win/kiwiclient/kiwirecorder.py
SRC=alsa_input.usb-Burr-Brown_from_TI_USB_Audio_CODEC-00.analog-stereo
rec() { timeout $(( $3 + 60 )) python3 "$KR" -q -s sdr.k1vl.com -p "$1" -f "$2" -m usb -L 300 -H 2700 --ncomp --tlimit "$3" -u NR5P-w1aw -d "$OUT" --fn "$4" >/dev/null 2>&1 & }
# (Started before 00:00Z, it waits through midnight; after 03:30Z, for the next night.)
waitfor() { until t=$(date -u +%H%M) && [ "$t" -ge "$1" ] && [ "$t" -le 0330 ]; do sleep 20; done; }
waitfor 0258
rec 8074 7046.5 2100 k1vl-40m-a; rec 8075 3580.5 2100 k1vl-80m-a
if [ "$2" != "--no-radio" ]; then timeout 2460 pw-record --target "$SRC" --rate 12000 --channels 1 "$OUT/nr5p-80m.wav" >/dev/null 2>&1 & fi
waitfor 0311
rec 8076 7046.5 1500 k1vl-40m-b; rec 8077 3580.5 1500 k1vl-80m-b
wait
echo "recorded in $OUT:"; ls -la "$OUT"/*.wav
echo "reference: curl the bulletin page (preamble names it, e.g. DX CW41 = ARLD041):"
echo "  https://www.arrl.org/w1awbulletinsdxissue?issue=$(date -u +%Y-%m-%d -d yesterday)&code=ARLDnnn  (DX)"
echo "  https://www.arrl.org/w1awbulletinsissue?issue=DATE&code=ARLBnnn / ARLPnnn (general / propagation)"
echo "then add lines to testdata/w1aw/bulletins.txt: <wav> <700 for the radio, 1000 for Kiwi> <ref> <label>"
