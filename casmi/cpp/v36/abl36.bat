@echo off
rem abl36.bat: ablations of the v36 e2e run (work_v36): ranker without the CNN term, then without the FPNet term,
rem each merged like the notebook and scored paired against the full v36 submission.
setlocal
set PY="C:\Program Files\Python312\python.exe"
cd /d C:\prog\casmi\e2e
%PY% -u e2e_rerank.py C:\prog\casmi\e2e\work_v36 abl_nocnn nocnn F0
%PY% -u e2e_pair.py input\comp\truth.tsv work_v36\abl_nocnn_submission.csv work_v36\submission.csv
%PY% -u e2e_rerank.py C:\prog\casmi\e2e\work_v36 abl_nofp nofpnet F0
%PY% -u e2e_pair.py input\comp\truth.tsv work_v36\abl_nofp_submission.csv work_v36\submission.csv
echo ALLDONE
