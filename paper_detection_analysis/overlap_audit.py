from pathlib import Path
from pypdf import PdfReader
import re, unicodedata, json
base=Path(r'E:\Naresh_Projects\PINN_EV_battery')
paths={
'new':base/'deliverables/PINN_EV_Battery_Authorship_Revised_IEEE_8_Pages.pdf',
'old':Path(r'C:\Users\naresh v\Downloads\PINN_EV_Battery_Originality_Revised_IEEE_8_Pages.pdf'),
'sim':Path(r'C:\Users\naresh v\Downloads\A Physics-Consistent Electrochemical.pdf'),
'ai':Path(r'C:\Users\naresh v\Downloads\A Physics-Consistent Electrochemical (1).pdf')}

def extract(p,pages=None):
 r=PdfReader(str(p)); ps=r.pages if pages is None else [r.pages[i] for i in pages]
 return '\n'.join((x.extract_text() or '') for x in ps)

def norm_tokens(s):
 s=unicodedata.normalize('NFKD',s).lower().replace('�',' ')
 s=re.sub(r'-\s*\n\s*(?=[a-z])','',s)
 s=re.sub(r'[^a-z0-9]+',' ',s)
 return s.split()
texts={'new':extract(paths['new']),'old':extract(paths['old']),'sim':extract(paths['sim'],range(3,11)),'ai':extract(paths['ai'],range(2,10))}
toks={k:norm_tokens(v) for k,v in texts.items()}

def ref_cut(ts):
 inds=[i for i,x in enumerate(ts) if x=='references']
 return inds[-1] if inds else len(ts)

def coverage(a,b,n):
 bg={tuple(b[i:i+n]) for i in range(len(b)-n+1)}
 cov=[False]*len(a)
 for i in range(len(a)-n+1):
  if tuple(a[i:i+n]) in bg:
   for j in range(i,i+n): cov[j]=True
 return sum(cov),len(a),cov
print('tokens', {k:len(v) for k,v in toks.items()})
for target in ['old','sim','ai']:
 print('\nTARGET',target)
 for label,a,b in [('full',toks['new'],toks[target]),('body',toks['new'][:ref_cut(toks['new'])],toks[target][:ref_cut(toks[target])])]:
  print(label,'lens',len(a),len(b))
  for n in [8,12,20]:
   c,L,_=coverage(a,b,n); print(n,c,L,round(c/L*100,2))
# sections new
A=toks['new']; B=toks['old'][:ref_cut(toks['old'])]
markers=[
('front',['abstract']),('intro',['i','introduction']),('related',['ii','related','w','ork']),('physics',['iii','governing','physics']),('model',['iv','proposed']),('experiment',['v','experimental','methodology']),('results',['vi','results','and','discussion']),('limits',['vii','limitations','and','threats','to','validity']),('conclusion',['viii','conclusion']),('data_code',['data','and','code','availability']),('ack',['acknowledgment']),('refs',['references'])]
def findseq(seq,start=0):
 for i in range(start,len(A)-len(seq)+1):
  if A[i:i+len(seq)]==seq:return i
 return -1
positions=[]
for name,seq in markers:
 p=findseq(seq,0)
 positions.append((name,p))
print('\npositions',positions)
for idx,(name,s) in enumerate(positions[:-1]):
 if s<0: continue
 e=next((p for _,p in positions[idx+1:] if p>s),len(A))
 sec=A[s:e]
 vals=[]
 for n in [8,12,20]:
  c,L,_=coverage(sec,B,n); vals.append(round(c/L*100,1))
 print(name,len(sec),vals)
# all coverage runs in body n=12
body=A[:ref_cut(A)]; c,L,cov=coverage(body,B,12)
runs=[]; i=0
while i<L:
 if not cov[i]: i+=1; continue
 j=i
 while j<L and cov[j]:j+=1
 if j-i>=12:runs.append((j-i,i,j,' '.join(body[i:j])))
 i=j
runs.sort(reverse=True)
print('\nTOP RUNS')
for z in runs[:30]: print('\n',z[0],z[1],z[3][:1500])
