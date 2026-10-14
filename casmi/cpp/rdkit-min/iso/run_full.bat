@echo off
rem run_full.bat <qry> <log> <cfg> <lib: xlib, slib for sqry>  - current submission settings + iso term, features dumped
cd /d C:\prog\casmi
C:\prog\casmi\fwdmodel\iso\build\phys_iso.exe eval C:\prog\casmi %1 pool730 10 phys2_fqry.txt "" fwd_mlp_rank_xlib.bin 0 "" rev_x3072.bin 0 %4 8 32 24 fwdmodel/iso/feat_iso_%1.txt 4 3 8000 "" "" "%3" > C:\prog\casmi\fwdmodel\iso\%2 2>&1
