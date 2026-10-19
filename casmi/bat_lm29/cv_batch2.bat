@echo off
cd /d C:\prog\casmi
set PY="C:\Program Files\Python312\python.exe"
set NP=merr,pop,z_pop,rk_pop,pop_m_max
%PY% fwdmodel\lm29\lm29_cv.py tg9 S:0.5,D:0.42,R:0.08 %NP% libfloor=0.5 trgate=0.9
%PY% fwdmodel\lm29\lm29_cv.py tg9R S:0.5,D:0.35,R:0.15 %NP% libfloor=0.5 trgate=0.9
%PY% fwdmodel\lm29\lm29_cv.py tg9perm S:0.5,D:0.42,R:0.08 %NP% libfloor=0.5 trgate=0.9 permute=1
%PY% fwdmodel\lm29\lm29_cv.py fl5npP S:0.5,D:0.42,R:0.08 %NP% libfloor=0.5 permute=1
for %%v in (tg9 tg9R tg9perm fl5npP) do (echo ==== %%v & %PY% fwdmodel\lm29\lm29_eval.py lm29\cv_%%v_?.txt --gate 0.9,0.8 --blend 1,0.5)
