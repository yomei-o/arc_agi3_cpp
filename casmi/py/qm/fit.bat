@echo off
rem fit.bat <tag> <init params> <coef> <QM_PA> <QM_DIR> <QM_FIT list or ALL> ["k,..;g,.."]
cd /d C:\prog\casmi
set QM_COEF=%3
set QM_PA=%4
set QM_DIR=%5
set QM_FIT=%~6
set QM_GRID=%~7
if "%~6"=="ALL" set QM_FIT=
C:\prog\casmi\fwdmodel\qm\phys_qm.exe fit C:\prog\casmi fqry pool730 10 qm\p_%1.txt %2 > qm\fit_%1.log 2>&1
findstr /c:"grid best" /c:"initial" /c:"fitted" /c:"ablation" /c:"sweep" qm\fit_%1.log
