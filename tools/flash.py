# Count rows shown and rows hidden/removed within 10 s of being shown (flashes).
import sys, re
shown={}; n_shown=0; flashes=0
for line in sys.stdin:
    m=re.match(r'(\S+) NOTE +[\d.]+ Hz #(\d+): ([^|]+)\|.*age (\d+) s', line)
    if not m: continue
    key=(m.group(1), m.group(2)); ev=m.group(3).strip(); age=int(m.group(4))
    if ev=='shown':
        n_shown+=1; shown[key]=age
    elif key in shown and (ev.startswith('hidden') or ev.startswith('removed')):
        if age-shown[key] <= 10: flashes+=1
        del shown[key]
print('shown %d, flashes %d' % (n_shown, flashes))
