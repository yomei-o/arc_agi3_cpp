@echo off
rem run_nolib.bat <qry> <log> [cfg]
cd /d C:\prog\casmi
C:\prog\casmi\fwdmodel\iso\build\phys_iso.exe eval C:\prog\casmi %1 pool730 10 phys2_fqry.txt "" "" 0 "" "" 0 "" 0 32 0 "" 4 3 8000 "" fwdmodel/iso/isostats_%1.txt "%3" > C:\prog\casmi\fwdmodel\iso\%2 2>&1
