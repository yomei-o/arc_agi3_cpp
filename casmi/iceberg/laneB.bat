@echo off
rem v30 + feature dump: hasqry (sibling), then hxqry (xinstr; zlib library and zlib REV2 as holdout2_STATUS)
call C:\prog\casmi\iceberg\v30dump.bat hasqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin has0 2 ""
call C:\prog\casmi\iceberg\v30dump.bat hxqry zlib fwd_mlp_rank_ylib.bin rev2_z.bin hx0 2 ""
echo ALLDONE
