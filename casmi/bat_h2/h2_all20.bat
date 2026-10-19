@echo off
rem holdout2: v20 / v21 (phys_ana, rev v1) on all five holdouts, ylib models
set B=C:\prog\casmi\h2\h2_v.bat
set MOD=ylib fwd_mlp_rank_ylib.bin rev_y3072.bin rev2_y.bin
for %%q in (haqry hbqry hcqry hasqry hbsqry) do (
  for %%v in (v20 v21) do call %B% %%q %%v 0 %MOD% %%q_%%v
)
echo ALL20DONE
