@echo off
rem v36 cnninst check: hcqry has no timsTOF spectrum, so cnnmodel + cnnpool=2 + cnninst=0 must give exactly the
rem cnnpool-0 run p36c_0 (= v35_0). Rebuilds phys_v36pc.exe first (cnninst added).
call C:\prog\casmi\v36\build_v36.bat
call C:\prog\casmi\v36\run36.bat hcqry ylib fwd_mlp_rank_ylib.bin rev2_y.bin p36c_inst 2 ",pcrev=tan,pcfp=rev3_fp3c16k_big.bin,attn=1,pcrank=rev,attpool=2,alloc=boltz,bT1=2,bT2=0.5,bpi=0.2,ba=0.5,cnnmodel=cnn_a_c.bin,cnnpool=2,cnninst=0"
cd /d C:\prog\casmi\v36\out
fc /b p36c_inst.csv p36c_0.csv > chkinst_fc.txt
fc /b p36c_inst_ins.csv p36c_0_ins.csv >> chkinst_fc.txt
findstr /c:"cnninst" /c:"cnn model" p36c_inst.log >> chkinst_fc.txt
type chkinst_fc.txt
echo ALLDONE
