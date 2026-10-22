@echo off
rem e2e_v38.bat : v38nofp notebook (tuned weights, tune38) end to end on the fake competition (hcqry + hxqry)
rem with the holdout-safe stand-ins, 4 cores (affinity F0), then the official-identity score. Compare with work_v37.
call C:\prog\casmi\v36\e2e36.bat casmi_match_v38nofp.ipynb work_v38 F0
