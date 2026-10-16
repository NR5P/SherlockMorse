#!/usr/bin/env python3
# Error anatomy of a decode against a reference: semi-global alignment, then
# substitutions classed by Morse shape (first element lost/added, last
# lost/added, one flipped, other), insertions and deletions.
# usage: tools/confusions.py <ref.txt> <hyp.txt|-> [--ctx]
import sys, re, collections, numpy as np
MORSE={'A':'.-','B':'-...','C':'-.-.','D':'-..','E':'.','F':'..-.','G':'--.','H':'....','I':'..','J':'.---','K':'-.-','L':'.-..','M':'--','N':'-.','O':'---','P':'.--.','Q':'--.-','R':'.-.','S':'...','T':'-','U':'..-','V':'...-','W':'.--','X':'-..-','Y':'-.--','Z':'--..','0':'-----','1':'.----','2':'..---','3':'...--','4':'....-','5':'.....','6':'-....','7':'--...','8':'---..','9':'----.','.':'.-.-.-',',':'--..--','?':'..--..','/':'-..-.','=':'-...-'}
def norm(s): return ' '.join(re.sub(r"[^A-Z0-9/?.,=' ]", ' ', s.upper()).split())
ref=norm(open(sys.argv[1]).read()); hyp=norm(open(sys.argv[2]).read() if sys.argv[2]!='-' else sys.stdin.read())
n,m=len(hyp),len(ref); D=np.zeros((n+1,m+1),dtype=np.int32); D[:,0]=np.arange(n+1); B=np.zeros((n+1,m+1),dtype=np.int8)
for i in range(1,n+1):
    hi=hyp[i-1]
    for j in range(1,m+1):
        sub=D[i-1,j-1]+(hi!=ref[j-1]); ins=D[i-1,j]+1; dele=D[i,j-1]+1
        best=min(sub,ins,dele); D[i,j]=best; B[i,j]=0 if best==sub else (1 if best==ins else 2)
j=int(np.argmin(D[n])); i=n; subs=collections.Counter(); ins=collections.Counter(); dels=collections.Counter(); ctx=[]
while i>0 and j>0:
    b=B[i,j]
    if b==0:
        if hyp[i-1]!=ref[j-1]: subs[(ref[j-1],hyp[i-1])]+=1; ctx.append((ref[j-1],hyp[i-1],ref[max(0,j-3):j-1],ref[j:j+2]))
        i-=1; j-=1
    elif b==1: ins[hyp[i-1]]+=1; i-=1
    else: dels[ref[j-1]]+=1; j-=1
tot=sum(subs.values())+sum(ins.values())+sum(dels.values())
def kind(a,b):
    pa,pb=MORSE.get(a),MORSE.get(b)
    if not pa or not pb: return 'other'
    if pb==pa[1:]: return 'first-lost'
    if pb==pa[:-1]: return 'last-lost'
    if pa==pb[1:]: return 'first-added'
    if pa==pb[:-1]: return 'last-added'
    if len(pa)==len(pb) and sum(x!=y for x,y in zip(pa,pb))==1: return 'one-flipped'
    return 'other'
kinds=collections.Counter()
for (a,b),c in subs.items(): kinds[kind(a,b)]+=c
print(f"hyp {n} chars; errors {tot}: subs {sum(subs.values())} ins {sum(ins.values())} dels {sum(dels.values())}")
print("sub kinds:", dict(kinds.most_common()))
print("top subs:", ' '.join(f"{a}>{b}:{c}" for (a,b),c in subs.most_common(12)))
print("top ins:", ' '.join(f"{a}:{c}" for a,c in ins.most_common(6)), "| top dels:", ' '.join(f"{a}:{c}" for a,c in dels.most_common(6)))
if '--ctx' in sys.argv:
    fl=[(a,b,p,q) for a,b,p,q in ctx if kind(a,b)=='first-lost']
    ws=sum(1 for a,b,p,q in fl if p.endswith(' ')); print(f"first-lost {len(fl)}: at word start {ws}, inside word {len(fl)-ws}")
    print(' '.join(f"[{p}|{a}>{b}|{q}]" for a,b,p,q in fl))
