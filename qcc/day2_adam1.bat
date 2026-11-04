@echo off
rem Benchmark III: Adam continuation of the (9,143,60) L-BFGS fit (stalled at 18.88 mHa), 150k steps, cosine lr 3e-4.
cd /d C:\prog\qcc
"C:\Program Files\Python312\python.exe" -u chem_adam.py fit_9_143_60_a.npz adam_60a.npz 150000 3e-4
echo ALL DONE
