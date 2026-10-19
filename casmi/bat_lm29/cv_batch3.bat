@echo off
cd /d C:\prog\casmi
set PY="C:\Program Files\Python312\python.exe"
set NP=merr,pop,z_pop,rk_pop,pop_m_max
%PY% fwdmodel\lm29\lm29_cv.py fl5npR S:0.5,D:0.35,R:0.15 %NP% libfloor=0.5
%PY% fwdmodel\lm29\lm29_cv.py fl5npR2 S:0.45,D:0.3,R:0.25 %NP% libfloor=0.5
for %%v in (fl5npR fl5npR2) do (echo ==== %%v & %PY% fwdmodel\lm29\lm29_eval.py lm29\cv_%%v_?.txt --gate 0.9,0.8 --blend 1,0.5)
