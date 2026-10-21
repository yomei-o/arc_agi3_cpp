@echo off
rem run.bat <qry> <logtag> [extra options...] : attn_eval with REV2 + fp3 big models, both vocabularies, log to attn\<qry>_<logtag>.log
rem options with "=" must be passed quoted, e.g. run.bat hcqry m4nt "mpen=4" "notwin=1"
setlocal
cd /d C:\prog\casmi
set Q=%~1
set T=%~2
shift
shift
set EX=
:loop
if "%~1"=="" goto go
set EX=%EX% %~1
shift
goto loop
:go
attn\attn_eval.exe qry=%Q% lib=wlib vocabs=v4k,fp3c16k models=rev2_w.bin:v4k,rev3_fp3c16k_big.bin:fp3c16k pop=1 threads=6 %EX% > attn\%Q%_%T%.log 2>&1
echo done %Q% %T%
