# rev2 integration status (2026-10-14 16:20, DONE)

## Code
- phys_ana2.cpp = phys_ana.cpp + REV2 support:
  local C:\prog\claude4\arc\arc3\casmi\cpp\rdkit-min\phys_ana2.cpp
  - reads the first 4 bytes of the reverse-model file; magic "REV2" (0x32564552) -> rev2::Model
    (input layout from the file, rev2::featurize(rm.fc,...)), candidate bits = fp2::Vocab built from the
    embedded vocab (rE/rF/hashes); otherwise the old rev::Model + ecfp r2 path, unchanged.
  - score = sum over candidate bits of logit(mean p over the query's spectra), p clipped to [0.01,0.99]
    (env REV2_NOCLIP=1 -> old [1e-6,1-1e-6] clip, for comparison). Errors out if a REV2 file has no vocab.
  - headers copied into rdkit-min/rev/: fp2.h, rev2_model.hpp (they need only backend.hpp, already there).
  - CLI args identical to phys_ana.
- Remote private build: C:\prog\casmi\fwdmodel\ana2\ (phys_ana2.cpp, rev\*, build2.bat = cl directly,
  linking the existing libs in fwdmodel\ana\build\rdkitmin read-only; no ninja/CMake touched).
  -> C:\prog\casmi\fwdmodel\ana2\phys_ana2.exe
- g++ check (w64devkit gcc 14.2, `g++ -c -O1 -std=c++20 -pthread -w -I . -I boost-min -I .. phys_ana2.cpp`
  in rdkit-min): compiles OK.

## Real-tool holdout results (mrr.py on the written CSVs; pool730, K=8000)
| reverse model | wr | cqry | tqry |
|---|---|---|---|
| rev_x3072.bin (old path, regression check) | 1.0 | 0.551 | 0.612 |
| rev2_vh.bin, no clip (REV2_NOCLIP=1) | 1.0 | 0.567 | 0.635 |
| rev2_vh.bin, clip [0.01,0.99] (default) | 1.0 | 0.568 | 0.642 |
| rev2_vh.bin, clip | 1.5 | **0.573** | **0.643** |
Matches the harness (0.550/0.612 old, 0.564/0.635 noclip, 0.5655/0.6411 clip, 0.5704/0.6429 wr1.5).
Logs/CSVs: C:\prog\casmi\fwdmodel\ana2\*.log, C:\prog\casmi\{cqry,tqry}_{r2,r2w15,old,r2nc}.csv

## Production model (trained)
- C:\prog\casmi\rev2_all_v4k.bin (248 MB), trained on alllib (275810 mols, 2.54M spectra), hold=0, 12 epochs, 626 s.
  Command (cwd C:\prog\casmi):
    fwdmodel\rev2\fp2_prep.exe apply C:\prog\casmi alllib v4k.bin
    fwdmodel\rev2\rev2_train.exe train lib=alllib fp=v4k out=rev2_all_v4k.bin epochs=12 hold=0
  Log: C:\prog\casmi\fwdmodel\rev2\ALL.log. No Platt (hold=0) -- not used by phys_ana2 anyway.
  Same vocab v4k.bin (built on xlib) embedded. Smoke test in phys_ana2 on cqry: loads, 0.602
  (leaky: cqry is in alllib, so NOT an estimate).

## Notebook args
  phys_ana2 eval <dir> <qry> <pool> 10 phys2_fqry.txt <out.csv> fwd_mlp_rank_<lib>.bin 0.25 "" rev2_all_v4k.bin 1.5 <lib> 8 32 24 "" 4 3 8000
  (identical to the current phys_ana line except exe name, rev file and wr; wr 1.0 is the safer choice, 1.5 was +0.005/+0.001)
  Dataset needs: rdkit-min/phys_ana2.cpp, rdkit-min/rev/{fp2.h,rev2_model.hpp} added, and rev2_all_v4k.bin.
  Not uploaded / no kernel pushed.
