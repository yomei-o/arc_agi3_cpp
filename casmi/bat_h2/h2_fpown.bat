@echo off
rem h2_fpown.bat <qry> <fpnet_scores.txt | none> <wf> <tag> [lib] [rev2] [mlp] [threads]
rem   Holdout reproduction of the v30 production fusion (= v28b + pcrev=tan; bat_pc3\run.bat / fwdmodel\pc2\run3.bat
rem   with phys_v30pc and ",pcrev=tan") with an OPTIONAL own-FPNet term "+ wf * z(fpnet)" (phys_v30pc argv[22]/[23]).
rem   The public FPNet is never used here.
rem   fpnet_scores.txt: fpnet_pool output ("<molecule>\t<pool name>\t<score>"); absolute or relative to C:\prog\casmi.
rem     It is copied to h2o\<tag>_fp.txt because phys_v30pc reads the file relative to <dir>.
rem   Defaults (holdout-safe w.r.t. the wlib-trained FPNet):
rem     rev2 = rev2_w.bin (wlib), mlp = fwd_mlp_rank_ylib.bin (holdout2: leak negligible), threads = 2
rem     lib  = zlib for hxqry (xinstr design: the answer's other-instrument spectra stay in the library),
rem            wlib otherwise (hcqry denovo, hasqry sibling)
rem   out: C:\prog\casmi\h2o\<tag>.csv (plain list), <tag>_ins.csv (PubChem-inserted = production output),
rem        <tag>_det.txt / <tag>_pp.txt dumps, <tag>.log (tool log + mrr.py lines for both csv files)
setlocal
cd /d C:\prog\casmi
set Q=%~1
set FP=%~2
set WF=%~3
set T=%~4
set L=%~5
set R2=%~6
set M=%~7
set TH=%~8
if "%Q%"=="" goto usage
if "%T%"=="" goto usage
if "%WF%"=="" set WF=1.0
if "%L%"=="" if /i "%Q%"=="hxqry" set L=zlib
if "%L%"=="" set L=wlib
if "%R2%"=="" set R2=rev2_w.bin
if "%M%"=="" set M=fwd_mlp_rank_ylib.bin
if "%TH%"=="" set TH=2
set FPA=
if /i "%FP%"=="none" goto run
if "%FP%"=="" goto run
copy /y "%FP%" "h2o\%T%_fp.txt" > nul
if errorlevel 1 (echo cannot copy %FP% & exit /b 1)
set FPA=h2o\%T%_fp.txt
:run
set O=gen=1,cap=300,prior=gen2o\gen2_prior_c.txt,gw=-2:0:-1:4:0:1.5,thr=%TH%,grid=0,libcap=1,ppopf=pc2\pool730_pop.bin,ppdet=h2o\%T%_pp.txt,pc=1,pcv=2,pcout=h2o\%T%_ins.csv,pcdet=h2o\%T%_det.txt,pcrank=fused,relhi=6,relp=3,sp=6,ppop=0.15,ppgate=0.9,pcrev=tan
echo %DATE% %TIME% start %T%: qry=%Q% lib=%L% mlp=%M% rev2=%R2% fpnet=[%FPA%] wf=%WF% thr=%TH% > h2o\%T%.log
C:\prog\casmi\fwdmodel\pc2\phys_v30pc.exe eval C:\prog\casmi %Q% pool730 10 phys_size_fqry.txt h2o\%T%.csv %M% 0.25 "" %R2% 1.0 %L% 8 32 24 "" 4 3 8000 "" "%FPA%" %WF% "%O%" >> h2o\%T%.log 2>&1
echo %DATE% %TIME% end %T% >> h2o\%T%.log
"C:\Program Files\Python312\python.exe" C:\prog\casmi\mrr.py C:\prog\casmi\%Q%_mols.txt C:\prog\casmi\h2o\%T%.csv C:\prog\casmi\h2o\%T%_ins.csv >> h2o\%T%.log 2>&1
findstr /c:"MRR@25" /c:"fpnet" /c:"all 0." h2o\%T%.log
echo done %T%
exit /b 0
:usage
echo usage: h2_fpown.bat ^<qry^> ^<fpnet_scores.txt^|none^> ^<wf^> ^<tag^> [lib] [rev2] [mlp] [threads]
exit /b 1
