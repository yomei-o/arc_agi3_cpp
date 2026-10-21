@echo off
rem ice_run.bat <tag> <qry> <threads> [max mols] : ICEBERG (CPU venv, real DGL 2.2.1) on cands_<tag>.json -> ice_<tag>.json
setlocal
cd /d C:\prog\casmi\iceberg
venv\Scripts\python.exe -W ignore ice_score.py --model ice --device cpu --mspred C:\prog\casmi\iceberg\src\ms-pred-708148c2a8eb0c32120644436fefd2fe90cd2214\src --cands cands_%~1.json --queries q_%~2.json --out ice_%~1.json --threads %~3 --bs 32 --max-mols %~4
echo done ice %~1
