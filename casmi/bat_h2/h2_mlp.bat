@echo off
rem holdout2: holdout-safe forward MLP on ylib (same commands as train_np.bat for xlib)
cd /d C:\prog\casmi\fwdmodel\rdkit-min\build
train_forward_model_nn.exe C:\prog\casmi ylib 20000 32 30000 256 0.002 fwd_mlp_ylib.bin 0.05 > C:\prog\casmi\h2\mlp1.log 2>&1
train_forward_model_rank.exe C:\prog\casmi ylib fwd_mlp_ylib.bin 3000 20 5 0.1 5000 0.0002 fwd_mlp_rank_ylib.bin > C:\prog\casmi\h2\mlp2.log 2>&1
echo MLPDONE >> C:\prog\casmi\h2\mlp2.log
