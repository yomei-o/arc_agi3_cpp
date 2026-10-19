@echo off
rem pass 1: library lists (match.cpp) and v28a main + single-engine lists on the holdout2 components
cd /d C:\prog\casmi\rf
rem match.cpp = the notebook's CPP_SOURCE (cell 0), identical to cpp/match.cpp; g++ needs w64devkit on PATH
set PATH=C:\prog\w64devkit\bin;%PATH%
if not exist match.exe g++.exe -O3 -std=c++17 -pthread -w -o match.exe match.cpp
call C:\prog\casmi\rf\match.bat hasqry ylib
call C:\prog\casmi\rf\match.bat hcqry ylib
call C:\prog\casmi\rf\match.bat haqry ylib
call C:\prog\casmi\rf\match.bat hxqry zlib
call C:\prog\casmi\rf\run.bat hasqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin has0 ""
call C:\prog\casmi\rf\run.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hc0 ""
call C:\prog\casmi\rf\run.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hc1 ",rm=1"
call C:\prog\casmi\rf\run.bat hxqry zlib fwd_mlp_rank_ylib.bin rev2_z.bin hx0 ""
call C:\prog\casmi\rf\run.bat haqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin ha0 ""
echo ALLDONE
