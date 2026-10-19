@echo off
rem Overnight QM barrier batch (detached via night\launch.ps1): 400 new fqry molecules, all bond classes,
rem then the most-stable-protonation reference, the ipso route for aryl-C, and the analyses.
cd /d C:\prog\casmi
set PY="C:\Program Files\Python312\python.exe"
set TS_CLASSES=
set TS_SKIP=600
set TS_GEO_DIR=C:\prog\casmi\ts_scan_geo_night
echo start %date% %time%
%PY% ts_scan.py fqry_mols.txt 400 ts_scan_n400.csv 8 3
echo scans done %date% %time%
%PY% ts_ref.py ts_scan_n400.csv ts_ref_n400.csv 8
echo refs done %date% %time%
%PY% ts_ipso.py ts_scan_n400.csv ts_ipso_n400.csv 8
%PY% ts_reref.py ts_ipso_n400.csv ts_scan_n400.csv ts_ref_n400.csv ts_ipso_n400r.csv
echo ipso done %date% %time%
%PY% ts_analyze.py ts_scan_n400.csv huckel_fqry2.json > ts_night_summary.txt 2>&1
%PY% ts_size_ctrl.py ts_scan_n400.csv huckel_fqry2.json >> ts_night_summary.txt 2>&1
%PY% ts_ipso_vs_obs.py ts_ipso_n400r.csv huckel_fqry2.json >> ts_night_summary.txt 2>&1
%PY% ts_ipso_subst.py ts_ipso_n400r.csv ts_scan_n400.csv >> ts_night_summary.txt 2>&1
echo all done %date% %time%
