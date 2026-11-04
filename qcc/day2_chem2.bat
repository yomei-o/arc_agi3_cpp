@echo off
rem Benchmark III: train (9,143,40) for error only (rho 1e-6); lambda is recovered afterwards by chem_gauge.py.
set PY=C:\prog\qcc\venv\Scripts\python.exe
cd /d C:\prog\qcc\starter\starter_kit
%PY% -u chem_train.py 9 143 40 20000 0.0005 0.005 runs_own\FeMoCo_9_143_40_ep40000_lr0.001_eps0.01_1793608117.npz 1e-6
echo ALL DONE
