@echo off
rem e2e_v37.bat : v37nofp notebook end to end on the fake competition (hcqry + hxqry) with the holdout-safe stand-ins
rem (v36 ones + ft8_ep5.bin as pool730/ft_all_ep5.bin), 4 cores (affinity F0), then the official-identity score
call C:\prog\casmi\v36\e2e36.bat casmi_match_v37nofp.ipynb work_v37 F0
