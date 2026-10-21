@echo off
rem pool ranking with an FPNet-format weights file: production fusion (v30 holdout copy, bat_h2\h2_fpown.bat) + wf z(fpnet)
rem usage: pool_eval.bat <tag> <weights.bin> <qry> [wf=1.0]   -> C:\prog\casmi\h2o\fo_<qry>_<tag>*.csv / .log
rem compare with the no-FPNet baselines fo_hc_base / fo_has_base / fo_hx_base (h2_fpown.bat ... none)
setlocal
set WF=%~4
if "%WF%"=="" set WF=1.0
set P=C:\prog\casmi\h2o\own_%~1_%~3
if not exist %P%_mol.bin C:\prog\casmi\fpown\gg\build\fpnet_ggml.exe %~2 C:\prog\casmi %~3 %P% 2
if not exist %P%_fp.txt C:\prog\casmi\fpnet\kbuild\fpnet_pool.exe C:\prog\casmi %~3 %P%_mol.bin pool730 C:\prog\casmi\pool730_fp10226.bin 10 %P%_fp.txt
call C:\prog\casmi\bat_h2\h2_fpown.bat %~3 %P%_fp.txt %WF% fo_%~3_%~1_w%WF%
