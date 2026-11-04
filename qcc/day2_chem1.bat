@echo off
rem Benchmark III: continue (9,143,40) from the 40k-epoch run at a lower learning rate (stalled at 34 mHa with lr 1e-3).
set PY=C:\prog\qcc\venv\Scripts\python.exe
cd /d C:\prog\qcc\starter\starter_kit
%PY% -u chem_train.py 9 143 40 10000 0.0002 0.010 runs_own\FeMoCo_9_143_40_ep40000_lr0.001_eps0.01_1793608117.npz
echo ALL DONE
