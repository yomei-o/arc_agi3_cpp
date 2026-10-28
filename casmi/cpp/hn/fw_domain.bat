@echo off
rem fw_domain: public ICEBERG / GLACIER discrimination by instrument on train.parquet (GPU fw_score)
cd /d C:\prog\casmi\glacier
"C:\Program Files\Python312\python.exe" fw_domain.py make C:\prog\casmi\e2e\work_v44\train.parquet C:\prog\casmi\glacier\dom_in.txt 400
fw_score_gpu.exe dom_in.txt dom_out.tsv gl=C:\prog\casmi\ds_fwm\glacier_w.bin ice=C:\prog\casmi\ds_fwm\ice_gen_w.bin,C:\prog\casmi\ds_fwm\ice_inten_w.bin threads=4 gpu=1
"C:\Program Files\Python312\python.exe" fw_domain.py eval dom_in.txt dom_out.tsv
echo ALLDONE %date% %time%
