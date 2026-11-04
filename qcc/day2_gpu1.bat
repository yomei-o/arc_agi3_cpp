@echo off
rem Benchmark III: GPU L-BFGS fit of the compact seed (9,143,40), 6 rounds x 10k iterations, kappa 5e-5.
cd /d C:\prog\qcc
"C:\Program Files\Python312\python.exe" -u chem_gpu.py fit seed_9_143_40.npz fit_9_143_40_a.npz 6 10000 5e-5
echo ALL DONE
