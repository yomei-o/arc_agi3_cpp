@echo off
rem hn_eval.bat : hnqry (natural products x timsTOF, denovo) with the nlib models — does the yardstick reproduce the LB order?
rem   LB: v36 0.344 (lib 8/24, fp 1.0 dot) > v39c 0.339 (fp 1.5) > v39a 0.328 (lib x1.5) > v39b 0.319 (lib x2) > v38 0.303 (lib 0); v40 0.345 (fp tan)
rem   pop = production pool popularity (ppopf + ppop 0.15); nopop = none (the hnqry molecules are famous, pop 9.3)
setlocal
set R=C:\prog\casmi\hn\run_hn.bat
set FP=hn\own_n_hnqry_fp.txt
set FT=hn\own_n_hnqry_fptan.txt
set POP=,ppopf=pc2\pool730_pop.bin,ppop=0.15
cd /d C:\prog\casmi
echo start %date% %time% > hn\eval_progress.txt
call %R% n_base_nopop 8 24 %FP% 1.0 ""
call %R% n_base_pop 8 24 %FP% 1.0 "%POP%"
call %R% n_lib0_nopop 0 0 %FP% 1.0 ""
call %R% n_lib15_nopop 12 36 %FP% 1.0 ""
call %R% n_lib2_nopop 16 48 %FP% 1.0 ""
call %R% n_fp15_nopop 8 24 %FP% 1.5 ""
call %R% n_fptan_nopop 8 24 %FT% 1.0 ""
call %R% n_lib0_pop 0 0 %FP% 1.0 "%POP%"
call %R% n_fptan_pop 8 24 %FT% 1.0 "%POP%"
echo ALLDONE %date% %time% >> hn\eval_progress.txt
