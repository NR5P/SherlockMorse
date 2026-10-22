# Sherlock Morse intro

An 11.6 s YouTube intro: a magnifying glass searches a foggy waterfall, decodes
SHERLOCK, and becomes the cameo from the logo.

**Watch it:** https://youtu.be/1Ova6egY-q8 (unlisted)

The rendered video isn't kept in git. Rebuild it from these sources:

```sh
node promo/intro/render.mjs    # -> promo/intro/sherlock-morse-intro.mp4 (1080p60) and .wav
```

Open `intro.html` in Chrome to preview it, and press Play. All timing lives in
`timeline.js`; the picture (`intro.html`) and soundtrack (`audio.mjs`) both read it.
