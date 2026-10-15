@echo off
rem DAG controls (the first ctrl_ev.bat never wrote pdc_*: "write() or write()" short-circuits)
set QM_TIES=avg
set C=C:\prog\casmi\qm\coef_fqry_full.txt
cd /d C:\prog\casmi
"C:\Program Files\Python312\python.exe" -c "B=r'C:\prog\casmi\qm\\';D=open(B+'pd_refitT.txt').read().split();[open(B+'pdc_%%s.txt'%%t,'w').write(chr(10).join(D[:29]+[k,g])+chr(10)) for t,k,g in (('k','0.03','0'),('g','0','0.1'),('kn','-0.03','0'),('gn','0','-0.1'))]"
for %%t in (k g kn gn) do call C:\prog\casmi\qm\evd.bat c%%t qm\pdc_%%t.txt %C% 0 1
