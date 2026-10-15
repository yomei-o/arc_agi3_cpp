@echo off
rem run.bat <qry> <lib> <tag> <extra opts> : phys_pc eval, v20/rev2 hand config + gen (v24 production) + PubChem channel; outputs in C:\prog\casmi\pcslot
cd /d C:\prog\casmi\fwdmodel\pcslot
phys_pc.exe eval C:\prog\casmi %1 pool730 10 phys2_fqry.txt pcslot\%3.csv fwd_mlp_rank_xlib.bin 0.25 "" rev2_vh.bin 1.0 %2 8 32 24 "" 4 3 8000 "" "" 1 gen=1,cap=300,prior=gen2o\gen2_prior_c.txt,gw=-2:0:-1:4:0:1.5,thr=8,grid=0,pc=1,pcout=pcslot\%3_ins.csv,pcdet=pcslot\%3_det.txt,%~4 > C:\prog\casmi\pcslot\%3.log 2>&1
"C:\Program Files\Python312\python.exe" C:\prog\casmi\mrr.py C:\prog\casmi\%1_mols.txt C:\prog\casmi\pcslot\%3.csv C:\prog\casmi\pcslot\%3_ins.csv >> C:\prog\casmi\pcslot\%3.log
echo done %3
