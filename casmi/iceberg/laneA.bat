@echo off
rem v30 + feature dump: hcqry (answer in pool), then hcqry rm=1 (answer removed)
call C:\prog\casmi\iceberg\v30dump.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hc0 2 ""
call C:\prog\casmi\iceberg\v30dump.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hc1 2 ",rm=1"
echo ALLDONE
