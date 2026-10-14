# CER of hyp against the best-matching stretch of ref (free ref start/end).
import sys, re
def norm(s): return ' '.join(re.sub(r"[^A-Z0-9/?.,=' ]", ' ', s.upper().replace('ñ','-')).split())
ref=norm(open(sys.argv[1], encoding='latin-1').read())
tot_e=tot_n=0
for hyp in sys.stdin:
    hyp=norm(hyp)
    if not hyp: continue
    m=len(hyp); n=len(ref)
    prev=[0]*(n+1)          # D[0][j]=0 : free start in ref
    for i in range(1,m+1):
        cur=[i]+[0]*n
        hi=hyp[i-1]
        for j in range(1,n+1):
            cur[j]=min(prev[j]+1, cur[j-1]+1, prev[j-1]+(hi!=ref[j-1]))
        prev=cur
    e=min(prev)
    tot_e+=e; tot_n+=m
    print('%5.1f%%  %s' % (100*e/m, hyp[:90]))
print('TOTAL %.1f%%' % (100*tot_e/max(tot_n,1)))
