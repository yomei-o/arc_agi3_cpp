@echo off
rem pilot.bat : 1 epoch of masked-peak pretraining on 10% of wlib, then embedding neighbours and the window yardstick
rem (hcqry, hxqry) -> C:\prog\casmi\pretrain\pilot_*.log
setlocal
cd /d C:\prog\casmi
set P=C:\prog\casmi\pretrain
%P%\pt_train.exe train lib=wlib out=pt_pilot frac=0.1 epochs=1 B=128 maxPeaks=64 lr=3e-4 warm=500 logEvery=200 ckptEvery=0 threads=3 > %P%\pilot_train.log 2>&1
%P%\pt_train.exe nbr model=pt_pilot.bin lib=wlib qrys=hcqry,hxqry tag=pilot threads=3 > %P%\pilot_nbr.log 2>&1
for %%q in (hcqry hxqry) do (
  %P%\pt_eval.exe qry=%%q lib=wlib vocabs=fp3c16k "models=rev3_fp3c16k_big.bin:fp3c16k" pop=1 threads=3 mpen=4 embnbr=ptnbr_%%q_pilot.bin "refcfg=3:10:0.1:0" "sims=0,1,2" "ks=1,10,50" "Ts=0.01,0.03,0.1" lams=0 "ensw=0.5,1,2" > %P%\pilot_eval_%%q.log 2>&1
)
echo pilot done
