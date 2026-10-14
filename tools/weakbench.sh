#!/bin/bash
# ARRL practice files faded into real 40 m band noise (testdata/noise40), 3 runs per SNR.
# usage: tools/weakbench.sh [cwtool] [snr list]
B=${1:-./target/release/cwtool}; SNR=${2:--4,-8,-11}
cd "$(dirname "$0")/.." || exit 1
N="--noise $(ls testdata/noise40/*.wav | tr '\n' ' ')"
run() { $B weak "$@" --snr=$SNR --trials 3 $N 2>/dev/null | tail -n +2; }
echo "arrl20"; run testdata/arrl/260303_20WPM.mp3 --freq 750 --reference testdata/arrl/260303_20.txt
echo "arrl13"; run testdata/arrl/260304_13WPM.mp3 --freq 750 --reference testdata/arrl/260304_13.txt
