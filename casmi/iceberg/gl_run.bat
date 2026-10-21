@echo off
rem gl_run.bat <tag> <qry> : GLACIER (GPU venv, dgl stub) on cands_<tag>.json -> gl_<tag>.json
setlocal
cd /d C:\prog\casmi\iceberg
venv_gpu\Scripts\python.exe -W ignore ice_score.py --model gl --dgl-stub --device cuda --mspred C:\prog\casmi\iceberg\src\ms-pred-708148c2a8eb0c32120644436fefd2fe90cd2214\src --cands cands_%~1.json --queries q_%~2.json --out gl_%~1.json --threads 1 --bs 32
echo done gl %~1
