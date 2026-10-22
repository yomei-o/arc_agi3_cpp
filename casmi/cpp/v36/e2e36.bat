@echo off
rem e2e36.bat <notebook name> <work dir name> <affinity hex>: run a whole submission notebook on the fake competition
rem input (hcqry + hxqry, e2e_STATUS.md) with the v36 holdout-safe substitutes, then score with the official identity.
setlocal
for /f "usebackq tokens=*" %%i in (`"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`) do set VSPATH=%%i
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
set E2E_WORK=C:\prog\casmi\e2e\%~2
set E2E_MASK=%~3
set PYTHONUNBUFFERED=1
cd /d C:\prog\casmi\e2e
echo start %date% %time%
"C:\Program Files\Python312\python.exe" -u C:\prog\casmi\e2e\e2e_run.py C:\prog\casmi\e2e\%~1 nb
echo notebook end %date% %time%
"C:\Program Files\Python312\python.exe" -u C:\prog\casmi\e2e\e2e_score.py score C:\prog\casmi\e2e\input\comp\truth.tsv %E2E_WORK%\submission.csv
echo ALLDONE %date% %time%
