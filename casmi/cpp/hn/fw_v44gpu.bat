@echo off
rem GPU fw_score (GLACIER + ICEBERG) over the e2e v44 fw_in.txt (pool candidates of the fused top 60) for fw_sweep2.py
cd /d C:\prog\casmi\glacier
fw_score_gpu.exe C:\prog\casmi\e2e\work_v44\fw_in.txt C:\prog\casmi\e2e\work_v44\fw_out_gi.tsv gl=C:\prog\casmi\ds_fwm\glacier_w.bin ice=C:\prog\casmi\ds_fwm\ice_gen_w.bin,C:\prog\casmi\ds_fwm\ice_inten_w.bin threads=4 gpu=1
echo ALLDONE %date% %time%
