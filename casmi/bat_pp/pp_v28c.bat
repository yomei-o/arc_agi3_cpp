@echo off
rem phys_v28fgr checks on cqry: formula-group rerank by TFM (lam 0.5, N40) and the random control
set E=C:\prog\casmi\fwdmodel\v28\phys_v28fgr.exe
call C:\prog\casmi\pp\pp_run.bat %E% cqry xlib fwd_mlp_rank_xlib.bin rev2_vh.bin c28tfm ",tfm=tfm_A.bin,tw=0,fgr=tfm,fgrn=40,fgrl=0.5"
call C:\prog\casmi\pp\pp_run.bat %E% cqry xlib fwd_mlp_rank_xlib.bin rev2_vh.bin c28rand ",tfm=tfm_A.bin,tw=0,fgr=rand,fgrn=25"
echo ALLDONE
