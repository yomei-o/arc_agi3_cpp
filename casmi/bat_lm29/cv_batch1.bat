@echo off
cd /d C:\prog\casmi
set PY="C:\Program Files\Python312\python.exe"
%PY% fwdmodel\lm29\lm29_cv.py nopop S:0.5,D:0.42,R:0.08 merr,pop,z_pop,rk_pop,pop_m_max
%PY% fwdmodel\lm29\lm29_cv.py fl5 S:0.5,D:0.42,R:0.08 merr libfloor=0.5
%PY% fwdmodel\lm29\lm29_cv.py fl5np S:0.5,D:0.42,R:0.08 merr,pop,z_pop,rk_pop,pop_m_max libfloor=0.5
%PY% fwdmodel\lm29\lm29_cv.py fl8np S:0.5,D:0.42,R:0.08 merr,pop,z_pop,rk_pop,pop_m_max libfloor=0.8
%PY% fwdmodel\lm29\lm29_cv.py fl5npS S:0.7,D:0.25,R:0.05 merr,pop,z_pop,rk_pop,pop_m_max libfloor=0.5
for %%v in (nopop fl5 fl5np fl8np fl5npS) do (echo ==== %%v & %PY% fwdmodel\lm29\lm29_eval.py lm29\cv_%%v_?.txt --gate 1e9,0.9,0.8 --blend 1,0.5)
