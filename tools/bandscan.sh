#!/bin/bash
# Receive-only adaptive band scan (40 + 80 m). Sets the VFO over CAT on /dev/ttyUSB0 only
# and records audio; never keys the radio. Hops every ~7 s; stays (90 s chunks, up to 6 min)
# only where a keyed tone is heard.
# usage: nightscan.sh <end-epoch-seconds>
cd "$(dirname "$0")/.." || exit 1
END=$1
OUT=testdata/night/$(date -u +%Y%m%d)
mkdir -p "$OUT"
LOG=$OUT/scan.log
CW=./target/release/cwtool
PROBE=$(mktemp --suffix=.wav)
freqs() {
  for k in $(seq 3500 2 3598); do echo $((k * 1000)); done
  for k in $(seq 7000 2 7070) $(seq 7100 2 7124); do echo $((k * 1000)); done
}
while [ "$(date +%s)" -lt "$END" ]; do
  for f in $(freqs); do
    [ "$(date +%s)" -lt "$END" ] || break
    $CW rig --set "$f" >/dev/null 2>&1 || { echo "$(date -u +%T) rig error at $f" >>"$LOG"; sleep 5; continue; }
    sleep 0.3
    arecord -q -D default -f S16_LE -r 48000 -c 1 -d 6 "$PROBE" 2>/dev/null
    k=$($CW keying "$PROBE" --duration 6 2>/dev/null)
    [ -n "$k" ] || continue
    echo "$(date -u +%T) $f keyed: $(echo "$k" | tr -s ' ' | tr '\n' ';')" >>"$LOG"
    for chunk in 1 2 3 4; do
      name=$OUT/$((f / 1000))-$(date -u +%H%M%S).wav
      arecord -q -D default -f S16_LE -r 48000 -c 1 -d 90 "$name" 2>/dev/null
      k=$($CW keying "$name" --duration 90 2>/dev/null)
      echo "$(date -u +%T)   $name: $(echo "$k" | wc -l) keyed" >>"$LOG"
      [ -n "$k" ] || { rm -f "$name"; break; }
    done
  done
done
rm -f "$PROBE"
echo "$(date -u +%T) done" >>"$LOG"
