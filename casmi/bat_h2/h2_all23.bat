@echo off
rem holdout2: v23..v26 on all five holdouts (+ rm=1 on the denovo ones), ylib models
set B=C:\prog\casmi\h2\h2_v.bat
set MOD=ylib fwd_mlp_rank_ylib.bin rev_y3072.bin rev2_y.bin
for %%q in (haqry hbqry hcqry hasqry hbsqry) do (
  for %%v in (v23 v24 v25 v26) do call %B% %%q %%v 0 %MOD% %%q_%%v
)
for %%q in (haqry hbqry hcqry) do (
  for %%v in (v24 v25 v26) do call %B% %%q %%v 1 %MOD% %%q_%%v_rm
)
echo ALL23DONE
