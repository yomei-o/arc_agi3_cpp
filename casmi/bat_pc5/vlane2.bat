@echo off
rem vlane.bat <lane> : phys_v33pc verification (2 threads each). Build first with fwdmodel\pc2\build.bat phys_v33pc.
rem   D: default (no alloc) hcqry rm=1, must equal pc3\hcT1(.csv/_ins.csv) byte for byte -> pc5\eq_csv.txt, eq_ins.txt
rem   B1/B0/BS: alloc=boltz on hcqry rm=1 / hcqry / hasqry (compare with py/boltz_merge.py on the v30 dumps)
set BO=,alloc=boltz,bT1=2,bT2=1,bpi=0.2
if "%~1"=="D" (
  call C:\prog\casmi\fwdmodel\pc5\run5.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hc33def_1 phys_v33pc 2 ",rm=1,pcrev=tan"
  fc /b C:\prog\casmi\pc5\hc33def_1.csv C:\prog\casmi\pc3\hcT1.csv > C:\prog\casmi\pc5\eq_csv.txt 2>&1
  fc /b C:\prog\casmi\pc5\hc33def_1_ins.csv C:\prog\casmi\pc3\hcT1_ins.csv > C:\prog\casmi\pc5\eq_ins.txt 2>&1
)
if "%~1"=="B1" call C:\prog\casmi\fwdmodel\pc5\run5.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hc33b_1 phys_v33pc 2 ",rm=1,pcrev=tan%BO%"
if "%~1"=="B0" call C:\prog\casmi\fwdmodel\pc5\run5.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hc33r_0 phys_v33pc 2 ",pcrev=tan%BO%,ba=0.25"
if "%~1"=="R1" call C:\prog\casmi\fwdmodel\pc5\run5.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hc33r_1 phys_v33pc 2 ",rm=1,pcrev=tan%BO%,ba=0.25"
if "%~1"=="BS" call C:\prog\casmi\fwdmodel\pc5\run5.bat hasqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin has33b_0 phys_v33pc 2 ",pcrev=tan%BO%"
echo ALLDONE %~1
