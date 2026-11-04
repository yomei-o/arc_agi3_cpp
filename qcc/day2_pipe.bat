@echo off
rem Benchmark III post-fit pipeline: progressive rounding -> offsets-only lambda refit (LS init) -> gauge ->
rem offsets refit -> build (brot 8, J 13) + official verify.   day2_pipe.bat <fit.npz> <tag> <stages> <iters>
set SYS="C:\Program Files\Python312\python.exe"
set VENV=C:\prog\qcc\venv\Scripts\python.exe
cd /d C:\prog\qcc
%SYS% -u chem_round.py %1 r_%2.npz 8 %3 %4 5e-5
%SYS% -u chem_lamfit.py r_%2.npz l1_%2.npz 19.9 1500 3e-2 lsinit wonly
cd /d C:\prog\qcc\starter\starter_kit
%VENV% -u chem_gauge.py C:\prog\qcc\l1_%2.npz C:\prog\qcc\g_%2.npz 3000 3e-3
cd /d C:\prog\qcc
%SYS% -u chem_lamfit.py g_%2.npz l2_%2.npz 19.9 1500 1e-2 wonly
%VENV% -u chem_build.py l2_%2.npz build_%2 8 13
echo ALL DONE
