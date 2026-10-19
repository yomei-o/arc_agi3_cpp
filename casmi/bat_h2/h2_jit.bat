@echo off
rem holdout2: precursor jitter 1.5 ppm versions (same molecules / spectra as haqry, hcqry, hasqry)
set B=C:\prog\casmi\h2\h2_v.bat
set MOD=ylib fwd_mlp_rank_ylib.bin rev_y3072.bin rev2_y.bin
for %%q in (hajqry hcjqry hasjqry) do (
  for %%v in (v25 v26) do call %B% %%q %%v 0 %MOD% %%q_%%v
)
for %%q in (hajqry hcjqry) do (
  for %%v in (v25 v26) do call %B% %%q %%v 1 %MOD% %%q_%%v_rm
)
for %%q in (hajqry hcjqry hasjqry) do (
  for %%v in (v20 v21 v23 v24) do call %B% %%q %%v 0 %MOD% %%q_%%v
)
for %%q in (hajqry hcjqry) do call %B% %%q v24 1 %MOD% %%q_v24_rm
echo JITDONE
