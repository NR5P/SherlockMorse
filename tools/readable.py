# Readability proxy: share of decoded word tokens (len>=2) that are dictionary
# words, QSO vocabulary, numbers/reports, or call-shaped.
import sys, re
D=set(l.strip() for l in open('data/corpus/dict.txt'))|set(l.strip().upper() for l in open('data/corpus/words.txt'))
Q=set(open('data/corpus/qso.txt').read().upper().split())
call=re.compile(r'^[A-Z]{1,2}[0-9][A-Z0-9]{0,3}[A-Z]$|^[0-9][A-Z][0-9][A-Z]{1,3}$')
good=tot=0
for line in sys.stdin:
    for w in line.split():
        w=w.strip('.,?=')
        if len(w)<2: continue
        tot+=1
        if w in D or w in Q or w.isdigit() or call.match(w): good+=1
print('%d/%d = %.1f%%' % (good, tot, 100*good/max(tot,1)))
