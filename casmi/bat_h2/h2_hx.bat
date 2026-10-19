@echo off
rem holdout2: xinstr holdout hxqry (timsTOF queries; answer's other-instrument spectra stay in zlib)
rem MLP = fwd_mlp_rank_ylib (17-dim, 20000 random molecules; leak negligible), rev models retrained on zlib
set B=C:\prog\casmi\h2\h2_v.bat
set MOD=zlib fwd_mlp_rank_ylib.bin rev_z3072.bin rev2_z.bin
for %%v in (v25 v26 v20 v21 v23 v24) do call %B% hxqry %%v 0 %MOD% hxqry_%%v
echo HXDONE
