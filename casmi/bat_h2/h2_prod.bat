@echo off
rem holdout2: train/serve-skew test for LambdaMART - v25/v26 with the PRODUCTION models
rem (fwd_mlp_rank_v2, rev2_all_v4k; trained on all of train = leaky on these holdouts, same leak for hand and ranker)
set B=C:\prog\casmi\h2\h2_v.bat
set MOD=ylib fwd_mlp_rank_v2.bin rev_all3072.bin rev2_all_v4k.bin
for %%q in (hasqry hcqry) do (
  for %%v in (v25 v26) do call %B% %%q %%v 0 %MOD% %%q_%%v_prod
)
for %%v in (v25 v26) do call %B% haqry %%v 1 %MOD% haqry_%%v_rm_prod
echo PRODDONE
