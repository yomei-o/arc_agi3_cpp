@echo off
rem Night 2026-11-04: wait for pipe_c60b, then a longer Adam (400k steps) on adam_60a and a slower rounding (60 x 2000).
cd /d C:\prog\qcc
:wait
findstr /c:"ALL DONE" C:\prog\casmi\night\pipe_c60b.log >nul 2>&1
if errorlevel 1 (
  timeout /t 120 /nobreak >nul
  goto wait
)
echo c60b finished, starting Adam 400k
"C:\Program Files\Python312\python.exe" -u chem_adam.py adam_60a.npz adam_60b.npz 400000 2e-4
call C:\prog\casmi\night\day2_pipe.bat adam_60b.npz c60c 60 2000
echo NIGHT CHAIN DONE
