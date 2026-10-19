@echo off
rem baseline dumps for the post-processing prototypes (phys_v27, v27a config)
set E=C:\prog\casmi\fwdmodel\v27\phys_v27.exe
call C:\prog\casmi\pp\pp_run.bat %E% cqry xlib fwd_mlp_rank_xlib.bin rev2_vh.bin c27 ",tfm=tfm_A.bin,tw=0"
call C:\prog\casmi\pp\pp_run.bat %E% tqry xlib fwd_mlp_rank_xlib.bin rev2_vh.bin t27 ",tfm=tfm_A.bin,tw=0"
call C:\prog\casmi\pp\pp_run.bat %E% hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hc27 ""
call C:\prog\casmi\pp\pp_run.bat %E% haqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin ha27 ""
call C:\prog\casmi\pp\pp_run.bat %E% hbqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hb27 ""
call C:\prog\casmi\pp\pp_run.bat %E% hasqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin has27 ""
call C:\prog\casmi\pp\pp_run.bat %E% hbsqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hbs27 ""
echo ALLDONE
