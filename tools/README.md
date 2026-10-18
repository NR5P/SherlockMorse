# Test tools

Scripts for measuring the decoder on real recordings. They need the
recordings in `testdata/` (not in git: off-air audio and ARRL files).

| Script | What it measures |
|---|---|
| `regress.sh` | Everything below plus `cwtool bench`, the ARRL files and the live recordings. Run for any decoder change. |
| `cwtool fists` | Not a script: synthetic sending styles of different people (keyers of all weights, straight keys, bugs, sideswiper, Farnsworth, run-together words) at +6 and −6 dB, CER per fist. Part of `regress.sh`. |
| `weakbench.sh` | ARRL practice files faded into real 40 m band noise (`cwtool weak`), CER at −4/−8/−11 dB. |
| `w1aw80.sh` | W1AW received on 80 m at night, against the bulletin's published text. |
| `semicer.py` | CER of decoded lines against the best-matching stretch of a reference text. |
| `readable.py` | Share of decoded words that are dictionary words, QSO terms, numbers or calls (for recordings without a reference). |
| `bandscan.sh <end-epoch>` | Receive-only adaptive scan of 40 and 80 m: sets the VFO over CAT (/dev/ttyUSB0) and records 6 s per 2 kHz step, staying 90 s chunks (up to 6 min) where `cwtool keying` hears CW. Stop the app first; it never keys the radio. |
| `keystrip.py file.wav HZ [out.png]` | One signal's keying as a strip picture plus mark/gap timings, to read the Morse by eye and compare with the decoder's copy (find HZ with `cwtool keying`). |
| `bugfist.sh` | KB1FGC's bug fist off the air (testdata/live/bug20261018): his call and QSO words copied by the skimmer. |
| `flash.py` | From `CWSK_NOTES=1` output: rows shown, and rows hidden again within 10 s. |

Run them from the repository root (`readable.py` reads `data/corpus/`).
