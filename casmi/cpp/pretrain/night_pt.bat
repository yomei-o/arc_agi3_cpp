@echo off
rem night_pt.bat : overnight masked-peak pretraining on wlib (pretrain_STATUS.md), then
rem   (a) embedding-neighbour attention on the window yardstick (hcqry, hxqry, haqry) -> SUMMARY.txt
rem   (b) fingerprint fine-tuning (fp3c16k) from the pretrained encoder vs from scratch, 1 epoch each, window MRR -> SUMMARY.txt
rem Re-running the same bat resumes the pretraining from pt_w_ck.bin (resume=1) and redoes the evaluations.
rem Launch detached: powershell -ExecutionPolicy Bypass -File C:\prog\casmi\night\launch.ps1 pt_night C:\prog\casmi\pretrain\night_pt.bat
setlocal
cd /d C:\prog\casmi
set P=C:\prog\casmi\pretrain
set PR=%P%\night_progress.txt
set S=%P%\SUMMARY.txt
set PY="C:\Program Files\Python312\python.exe"
set EP=6
set STOP=540
echo start %date% %time% > %PR%

rem ---------------- 1. pretraining (d 512, L 6; stopAfter %STOP% min caps it so (a) runs before morning)
set RES=0
if exist C:\prog\casmi\pt_w_ck.bin set RES=1
echo pretrain resume=%RES% %time% >> %PR%
%P%\pt_train.exe train lib=wlib out=pt_w epochs=%EP% B=128 d=512 L=6 H=8 ff=4 HH=1024 maxPeaks=64 lr=2e-4 warm=3000 logEvery=1000 ckptEvery=3000 stopAfter=%STOP% threads=3 resume=%RES% >> %P%\night_train.log 2>&1
echo pretrain done rc=%errorlevel% %time% >> %PR%

rem ---------------- 2. (a) embedding neighbours + window yardstick
%P%\pt_train.exe nbr model=pt_w.bin lib=wlib qrys=hcqry,hxqry,haqry tag=w threads=3 > %P%\night_nbr.log 2>&1
echo nbr done rc=%errorlevel% %time% >> %PR%
for %%q in (hcqry hxqry haqry) do (
  %P%\pt_eval.exe qry=%%q lib=wlib vocabs=fp3c16k "models=rev3_fp3c16k_big.bin:fp3c16k" pop=1 threads=3 mpen=4 embnbr=ptnbr_%%q_w.bin "refcfg=3:10:0.1:0" "sims=0,1,2" "ks=1,10,50" "Ts=0.01,0.03,0.1" lams=0 "ensw=0.5,1,2" > %P%\night_eval_%%q.log 2>&1
  echo eval %%q done %time% >> %PR%
)
echo ==== (a) masked-peak pretraining pt_w.bin (wlib), embedding attention on the window yardstick > %S%
type %P%\night_train.log | findstr /c:"model:" /c:"val" /c:"epoch" /c:"stopped" /c:"wrote" >> %S%
type %P%\night_nbr.log | findstr /c:"wrote" >> %S%
%PY% C:\prog\casmi\pretrain\attn_sum.py %P%\night_eval_hcqry.log %P%\night_eval_hxqry.log %P%\night_eval_haqry.log >> %S% 2>&1
echo (a) done %time% >> %PR%

rem ---------------- 3. (b) fine-tuning vs scratch (equal budget: 1 epoch, same hyper-parameters, fp3c16k)
%P%\pt_train.exe export model=pt_w.bin lib=wlib fp=fp3c16k HH=2048 out=ft_pre_ck.bin > %P%\night_export.log 2>&1
%P%\tfm_train.exe train lib=wlib fp=fp3c16k out=ft_pre resume=1 epochs=1 B=128 lr=3e-4 warm=2000 eval=none hold=20 ckptEvery=0 threads=3 > %P%\night_ft_pre.log 2>&1
echo ft_pre done %time% >> %PR%
%P%\tfm_train.exe train lib=wlib fp=fp3c16k out=ft_scr d=512 L=6 H=8 ff=4 HH=2048 maxPeaks=64 epochs=1 B=128 lr=3e-4 warm=2000 eval=none hold=20 ckptEvery=0 threads=3 > %P%\night_ft_scr.log 2>&1
echo ft_scr done %time% >> %PR%
for %%m in (ft_pre ft_scr) do (
  for %%q in (hcqry hxqry) do (
    %P%\pt_train.exe mollogits model=%%m.bin qry=%%q out=pretrain\%%m_%%q_mol.bin > %P%\night_mol_%%m_%%q.log 2>&1
  )
)
echo ==== (b) fine-tuning from the pretrained encoder (ft_pre) vs from scratch (ft_scr), 1 epoch each, fp3c16k >> %S%
type %P%\night_ft_pre.log %P%\night_ft_scr.log | findstr /c:"val" /c:"F1" /c:"epoch" >> %S%
for %%q in (hcqry hxqry) do (
  C:\prog\casmi\fpown\build\win_eval.exe dir=C:\prog\casmi qry=%%q win=w_%%q "models=pretrain\ft_pre_%%q_mol.bin:fp3c16k,pretrain\ft_scr_%%q_mol.bin:fp3c16k" threads=3 > %P%\night_win_%%q.log 2>&1
  echo == %%q >> %S%
  type %P%\night_win_%%q.log | findstr /c:"MRR_in" >> %S%
)
echo all done %time% >> %PR%
echo all done >> %S%
