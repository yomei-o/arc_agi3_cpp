@echo off
rem eval_all.bat <model prefix gl|ice> : rebuild the K=40 lists and evaluate <prefix>_<tag>.json on the 4 holdout runs
setlocal
cd /d C:\prog\casmi\iceberg
set PY=C:\prog\casmi\iceberg\venv\Scripts\python.exe
set MSG=msg_ik14_train.txt,msg_ik14_val.txt
"%PY%" -W ignore ice_eval.py build C:\prog\casmi\iceberg\v30 has0 40 C:\prog\casmi\iceberg C:\prog\casmi\hasqry_mols.txt
"%PY%" -W ignore ice_eval.py build C:\prog\casmi\iceberg\v30 hc0 40 C:\prog\casmi\iceberg C:\prog\casmi\hcqry_mols.txt
"%PY%" -W ignore ice_eval.py build C:\prog\casmi\iceberg\v30 hc1 40 C:\prog\casmi\iceberg C:\prog\casmi\hcqry_mols.txt
"%PY%" -W ignore ice_eval.py build C:\prog\casmi\iceberg\v30 hx0 40 C:\prog\casmi\iceberg C:\prog\casmi\hxqry_mols.txt
"%PY%" -W ignore ice_eval.py eval C:\prog\casmi\iceberg has0 C:\prog\casmi\hasqry_mols.txt q_hasqry.json %MSG% %~1_has0.json > eval_%~1_has0.txt
"%PY%" -W ignore ice_eval.py eval C:\prog\casmi\iceberg hc0 C:\prog\casmi\hcqry_mols.txt q_hcqry.json %MSG% %~1_hc0.json > eval_%~1_hc0.txt
"%PY%" -W ignore ice_eval.py eval C:\prog\casmi\iceberg hc1 C:\prog\casmi\hcqry_mols.txt q_hcqry.json %MSG% %~1_hc1.json > eval_%~1_hc1.txt
"%PY%" -W ignore ice_eval.py eval C:\prog\casmi\iceberg hx0 C:\prog\casmi\hxqry_mols.txt q_hxqry.json %MSG% %~1_hx0.json > eval_%~1_hx0.txt
"%PY%" ice_mix.py C:\prog\casmi\iceberg has0 hc0 hc1 > mix_%~1_has.txt
"%PY%" ice_mix.py C:\prog\casmi\iceberg hx0 hc0 hc1 > mix_%~1_hx.txt
type mix_%~1_has.txt mix_%~1_hx.txt
