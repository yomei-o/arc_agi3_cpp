# FPNet (ahmedberatozer/casmi26-fpnet-full1) in C++/ggml — status 2026-10-14

## Done
- Weights: fpnet_full1.pt is NOT the pv_fp.py architecture. It is casmi/fpnet.py of ahmedberatozer/casmi26-v4b-models
  (copied to build:C:\prog\casmi\fpnet\code): d=512, 8 layers, 8 heads, nbits=10226, 60 adducts (chem.ADDUCT_LIST),
  int_emb Linear(2: it, it^2), ce_emb SinEmb(128, lo 0, hi 2.2) on ce/10+1, global = [prec_emb, ce_emb, ce_known, merged(n/4), mode, log1p(prec)/10],
  SinEmb hi=3.3, prep_peaks: mz<=prec+2, floor 1e-3, top-160 by intensity, sqrt. No instrument input. elem_head unused.
  Molecule logits = 0.5*(mean per-spectrum + mean over polarities of merged-spectrum logits) (probe_core2.molecule_logits).
  Converted to raw f32: build:C:\prog\casmi\fpnet\fpnet_full1.bin (199 MB) by cpp/fpnet/convert.py.
- Fingerprint (10226 bits): chem.raw_fingerprint = ECFP4 4096 | ECFP6 4096 | FCFP4 2048 | RDKit maxPath6 2048 | AtomPair 2048 |
  TopTorsion 2048 | MACCS 167 (=16551 raw) -> [fp_bits.npy of ahmedberatozer/casmi26-v2-pool] -> np.packbits (1279 B/row).
  Verified 300/300 random structures bit-identical to that dataset's pool_fp.npy.
- build:C:\prog\casmi\pool730_fp10226.bin (729389 x 1279 B = 933 MB, row = line of pool730_mols.txt) + .ok (1 byte/row, 3 unparsable).
- ggml: llama.cpp's bundled ggml 0.24.0 (tree of 2026-09-18) copied to build:C:\prog\casmi\fpnet_src\cpp\ggml; CPU backend, AVX2.
  Source: cpp/fpnet/fpnet_ggml.cpp + CMakeLists.txt + build.bat (build copy in build:C:\prog\casmi\fpnet_src\cpp, exe build\fpnet_ggml.exe).
- Validation vs PyTorch (ref_logits.py, cqry first 5 mols, 15 spectra): max |diff| 3e-5..1e-4 (logit range +-163) on 14/15 spectra and 4/5 molecules;
  1 spectrum differs by 0.43 because of an intensity TIE at the 160th peak (numpy unstable argsort picks a different peak) — not a port error.
- Runtime (4 threads, AVX2, desktop CPU): 40-56 ms/forward pass uncontended (sqry 1565 passes 62 s, tqry 707 passes 40 s);
  hidden test ~1200 spectra -> ~1600 passes -> ~1.5-2 min on Kaggle 4 cores (estimate).
- Scores: build:C:\prog\casmi\fpnet_{cqry,tqry,sqry}.txt (same Q lines as comb_*.txt, "<name> <f.z>"; 0 missing names).
  Molecule logits: build:C:\prog\casmi\fpnet\gg_{q}_mol.bin (i32 n, i32 nbits, then i32 mol, f32[nbits]).

## FPNet-alone MRR@25 (score = cand_fp . logits)  — LEAKED, see caveat
| set | FPNet | best existing single column (comb) |
|---|---|---|
| cqry (400) | 0.750 (top1 0.677) | ana 0.567 |
| tqry (176) | 0.907 (top1 0.864) | ana 0.629 |
| sqry (400) | 0.756 (top1 0.670) | inLib 0.913 / ana 0.835 |
CAVEAT: fpnet_full1 ("full") was almost certainly trained on all of train.parquet, which contains every holdout molecule; these
numbers are optimistic (memorisation). (Author notebooks: v4m = bank swapped to fpnet_full1, LB 0.378; v4l without it 0.380 — no clear LB gain there.)
Holdout weights for a fusion with this column will be overestimated; use with heavy shrinkage / fit weights on something leak-free.

## Caveats in our inputs
- Many cqry spectra have tag -1 (adduct unknown) -> adduct <unk>, mode +1 (negative spectra treated as positive). CE unknown -> ce_known=0.
- Our peaks already have a 0.002 floor (model floor 1e-3): peaks 0.1-0.2% are lost; minor.

## Remaining
1. Fusion: add fpnet column into fuse_fit (fpnet_{q}.txt aligned by (query, name)); leak-aware weighting.
2. Kaggle: upload fpnet_full1.bin (199 MB; f16 would halve it) + pool730_fp10226.bin (933 MB, compresses well) + ggml source;
   build in notebook with cmake (~1-2 min, CPU AVX2, -DGGML_NATIVE=OFF) or ship the Linux binary built once in a notebook.
   Or use ahmedberatozer/casmi26-fpnet-full1 directly as the weights dataset and convert in-notebook (torch is available there).
3. Optional: batch several spectra per graph (masked softmax) for speed; currently not needed.
