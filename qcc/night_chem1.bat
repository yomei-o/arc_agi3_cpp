@echo off
rem Overnight Benchmark III factorization runs (build machine), launched with C:\prog\casmi\night\launch.ps1.
rem 1) public shape (9,143,40) from the lambda-regularized physics init, 40k epochs
rem 2) (20,30,15) and 3) (30,15,10): physics init then 40k epochs - is a small R*B (short angle table) feasible?
set PY=C:\prog\qcc\venv\Scripts\python.exe
cd /d C:\prog\qcc\starter\starter_kit
%PY% -u chem_train.py 9 143 40 40000 0.001 0.010 C:\prog\qcc\verifiers\init_9_143_40_mu1e-4.npz
cd /d C:\prog\qcc\verifiers
%PY% -u chem_init.py 20 30 15 3000 block init_20_30_15_mu1e-4.npz 1e-4
%PY% -u chem_init.py 30 15 10 3000 block init_30_15_10_mu1e-4.npz 1e-4
cd /d C:\prog\qcc\starter\starter_kit
%PY% -u chem_train.py 20 30 15 40000 0.001 0.010 C:\prog\qcc\verifiers\init_20_30_15_mu1e-4.npz
%PY% -u chem_train.py 30 15 10 40000 0.001 0.010 C:\prog\qcc\verifiers\init_30_15_10_mu1e-4.npz
echo ALL DONE
