@echo off
rem pass 2: the timsTOF components (hidden test is timsTOF): hbqry denovo, hbsqry sibling
call C:\prog\casmi\rf\match.bat hbqry ylib
call C:\prog\casmi\rf\match.bat hbsqry ylib
call C:\prog\casmi\rf\run.bat hbqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hb0 ""
call C:\prog\casmi\rf\run.bat hbsqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin hbs0 ""
echo ALLDONE
