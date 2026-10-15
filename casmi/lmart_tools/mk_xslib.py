# xslib = xlib minus the exact query spectra of sqry and oqry (REV2 retrain without the query-spectrum leak)
import numpy as np, sys
D='C:/prog/casmi/'
HD=np.dtype([('prec','<f4'),('mol','<i4'),('off','<i4'),('n','<i4'),('tag','<i4')])
def load(stem):
    h=np.fromfile(D+stem+'_head.bin',dtype=HD)
    p=np.fromfile(D+stem+'_peaks.bin',dtype='<f4')
    names=[l.rstrip('\r\n').split('\t')[0] for l in open(D+stem+'_mols.txt',encoding='utf-8')]
    starts=np.concatenate([[0],np.cumsum(2*h['n'].astype(np.int64))])
    return h,p,names,starts
def keys(stem):
    h,p,names,st=load(stem); ks=[]
    for i in range(len(h)):
        n=int(h['n'][i]); mz=p[st[i]:st[i]+n]
        ks.append((names[h['mol'][i]],round(float(h['prec'][i]),3),n,round(float(mz.sum()),2)))
    return ks
drop=set(keys('sqry'))|set(keys('oqry'))
print('query spectra',len(drop))
h,p,names,st=load('xlib')
keep=np.ones(len(h),bool); hit=0
for i in range(len(h)):
    n=int(h['n'][i])
    k=(names[h['mol'][i]],round(float(h['prec'][i]),3),n,round(float(p[st[i]:st[i]+n].sum()),2))
    if k in drop: keep[i]=False; hit+=1
print('xlib spectra',len(h),'dropped',hit)
idx=np.nonzero(keep)[0]
h2=h[idx].copy()
parts=[p[st[i]:st[i+1]] for i in idx]
np.concatenate(parts).astype('<f4').tofile(D+'xslib_peaks.bin')
off=np.concatenate([[0],np.cumsum(h2['n'].astype(np.int64))[:-1]]).astype(np.int32)
h2['off']=off
h2.tofile(D+'xslib_head.bin')
import shutil
shutil.copyfile(D+'xlib_mols.txt',D+'xslib_mols.txt'); shutil.copyfile(D+'xlib_v4k.bin',D+'xslib_v4k.bin')
print('done')
