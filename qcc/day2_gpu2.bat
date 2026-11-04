@echo off
rem Benchmark III: (9,143,60) compact seed + GPU L-BFGS fit, 6 rounds x 8k iterations, kappa 5e-5.
cd /d C:\prog\qcc
"C:\Program Files\Python312\python.exe" -u chem_gpu.py seed 9 143 60 seed_9_143_60.npz 100
"C:\Program Files\Python312\python.exe" -u chem_gpu.py fit seed_9_143_60.npz fit_9_143_60_a.npz 6 8000 5e-5
echo ALL DONE
