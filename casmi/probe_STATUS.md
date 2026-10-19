# probe — LB probe: how many hidden-test answers are outside pool730 (state file)

2026-10-19 (afternoon). Motivation: RESUME.md "2026-10-19 夕方の方針転換" (a): LB arithmetic says R ≈ 0.7 of hidden answers are
outside pool730. One submission whose list contains ONLY non-pool PubChem candidates measures it:
LB_probe ≈ f_out · m_out + (1 − f_out) · leak, where m_out = MRR of the list on out-of-pool answers (hcqry rm=1) and
leak = MRR on in-pool answers (hcqry normal; should be ~0 if the exclusion works). → f_out ≈ (LB_probe − leak) / (m_out − leak).

## Files
- `cpp/rdkit-min/phys_probe.cpp` = phys_v28pc.cpp + `probe=1` (needs pc=1,pcv=2; use pcrank=fused). Without probe=1 the output is
  byte-for-byte phys_v28pc's logic. With probe=1:
  - exclusion set = stereo-blind canonical SMILES (gen::key re-parsed, the same `canon` phys_pc uses for dedup) of the pool730
    molecules in the query's mass windows (same windows as the PubChem query). Generated candidates are NOT excluded, and the truth
    is never used to filter (the `isT && truthInPool` skip of v28pc is disabled under probe).
  - order = PubChem v2 fused: z_window(REV2) + 0.15 pop prefilter (top pcpre=300), dedup, top pcn=50 fully scored, sorted by
    z(hand fusion over them) + 0.15 pop; top 25; no gate (every query).
  - out.csv = probe list; `probesmi=file` = same list as SMILES; `pcout=file` = the normal (plain) list.
  - log line: `PROBE: non-pool PubChem list only: N queries, average list length L, empty lists E; eval (canonical-key truth) MRR@25 m, truth in list k`
- `bat_probe/build.bat`, `run.bat`, `all.bat` (build machine C:\prog\casmi\fwdmodel\probe, outputs C:\prog\casmi\probe\pr_*.{csv,log})
- `py/probe_ik_mrr.py` — scores a SMILES list by InChIKey14 (RDKit 2026.03 on the build machine), like Kaggle, with a 95% bootstrap CI
- `py/make_probe_nb.py` → `notebook/casmi_match_probe.ipynb` (from casmi_match.ipynb v28a), `notebook/kernel-metadata-probe.json`
- staged dataset file: build machine `C:\prog\casmi\probe\ds_add\rdkit-min\phys_probe.cpp` (→ dataset yomeiotani/casmi26-rdkit-min, folder rdkit-min/)

## Measurement (holdout-safe: ylib, fwd_mlp_rank_ylib.bin, rev2_y.bin; FPNet off; v27a config = phys_size_fqry, libcap, gen=1 cap 300)
(rev v1 rev_y3072.bin is not used by phys_v28pc/phys_probe: REV2 replaced it.)

| run | what | probe list MRR@25 (canonical-key truth, C++ / mrr.py) | **IK14 (Kaggle-like, probe_ik_mrr.py)**, 95% bootstrap CI | hits / top1 | mean len |
|---|---|---|---|---|---|
| pr_hc1 | hcqry rm=1 (answer out of pool) = **m_out** | 0.484 | **0.497 [0.452, 0.543]** | 239 / 178 of 400 | 24.4 |
| pr_hc0 | hcqry normal (answer in pool) = **leak** | 0.000 | **0.013 [0.004, 0.024]** | 10 / 4 | 24.4 |
| pr_ha1 | haqry rm=1 (COCONUT ≥1 spectrum, pop 4.94 = famous) | 0.540 | **0.550 [0.504, 0.598]** | 263 / 202 | 24.6 |
| pr_hb1 | hbqry rm=1 (timsTOF 245-461 = enveda-180-like synthetic, pop 1.30, window 14,545/query) | 0.274 | **0.278 [0.236, 0.320]** | 133 / 98 | 25.0 |

- For reference the normal (plain) list on hcqry rm=1 scores 0.151 (generation only), and the window has 6,677 PubChem compounds per
  query (median 3,687); the truth is among the ~48 fully scored PubChem candidates in 58.5% of queries.
- **The non-pool PubChem list alone finds an out-of-pool answer with MRR ~0.5 on hcqry** — far above what the inserted channel
  showed (pc2: slots 4..20 give rm=1 0.151 → 0.20, because the slots cap each hit at 1/4 and generated duplicates are dropped).
- Leak (sanity): canonical-key exclusion is exact (0 hits). Under IK14 10/400 in-pool answers still appear, all as **IK14-identical
  but differently written forms**: COCONUT's imidic-acid tautomers `C(O)=N` vs PubChem's amides, `n...c(O)` vs lactams,
  zwitterions/phosphate anions (`py/probe_leak.py`). Small (0.013) but it is a floor on the probe's LB.
- hcqry is popularity-matched (PubChem pop 2.09 = "COCONUT molecules not in train"), i.e. ordinary, unlike cqry/tqry (5–7.7).
  The popularity weight here is the light 0.15 (both prefilter and final), so the dependence on fame is mild.

## Reading the LB of the probe
LB_probe ≈ f_out · m_out + (1 − f_out) · leak  →  **f_out ≈ (LB_probe − 0.013) / (0.497 − 0.013) = (LB_probe − 0.013) / 0.484**.
| LB_probe | 0.03 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.35 |
|---|---|---|---|---|---|---|---|
| f_out (m_out 0.497) | 0.04 | 0.08 | 0.18 | 0.28 | 0.39 | 0.49 | 0.70 |
| range from the m_out CI (0.452–0.543) | | | 0.16–0.20 | 0.25–0.31 | 0.35–0.43 | 0.45–0.54 | 0.64–0.76 |
| f_out if the hidden out-of-pool answers are enveda-180-like (m_out 0.278, hbqry) | 0.06 | 0.14 | 0.33 | 0.52 | 0.71 | 0.89 | >1 |
(plus the public-LB sampling error, ~130 molecules: per-query RR² mean ~0.46 on hcqry rm=1 → SE of LB_probe ≈ 0.035 at f=0.5, ≈ 0.025 at f=0.2.)

Caveats:
1. The probe measures f_out · m_out_hidden; reading it as f_out assumes the hidden out-of-pool answers are as findable as hcqry's.
   If they are e.g. enveda-180-like synthetic molecules (PubChem pop ~1.3, 92% in PubChem) or not in PubChem at all, m_out_hidden
   is smaller and f_out is underestimated. Measured: NP-like hcqry 0.497, famous haqry 0.550, **enveda-180-like hbqry 0.278**
   (bigger windows, 14.5k vs 6.7k; truth among the 50 fully scored only 33% vs 59%). So m_out spans ~0.28–0.50 by molecule
   type, and the f_out read from one LB number carries that factor ~1.8 (rows above). Since 65% of train molecules are
   enveda-180 and the hidden test is timsTOF, the enveda-like row may be the more relevant one for the synthetic part.
2. **Prior expectation is already in tension with f_out ≈ 0.7**: with m_out ~0.5, f_out = 0.7 would give a probe LB ~0.35 (~0.20 with the enveda-like m_out 0.278), and the
   old PubChem channel B (v25) would have moved the LB by much more than ±0.001. Either f_out is small, or the hidden out-of-pool
   answers are much harder than hcqry rm=1 (outside PubChem / outside the ±10 ppm window / unusual adducts). The probe LB alone
   cannot separate "few out of pool" from "out of pool and not findable"; both say "selecting inside the PubChem window" is not
   the lever it looked like — unless the probe LB comes out high.
3. Kaggle uses rev2_all_v4k (trained on all train) and FPNet does not enter the probe list (hand fusion only: phys, MLP, REV2,
   analog); hidden molecules are not in train, so this matches the holdout-safe setting.

## Kaggle run (not done: no upload, no push, no submit)
1. Dataset yomeiotani/casmi26-rdkit-min: add `rdkit-min/phys_probe.cpp` (staged at build machine
   `C:\prog\casmi\probe\ds_add\rdkit-min\phys_probe.cpp`, = casmi/cpp/rdkit-min/phys_probe.cpp; next to phys_v28pc.cpp, which is
   already in the dataset, as is pool730/pool730_pop.bin). Nothing else is new. New dataset version, e.g. copy ds_rdkit to a
   scratch dir + the file, `kaggle datasets version -p <dir> -m "phys_probe.cpp" -r zip` (or upload the one file in the UI).
2. Kernel: `casmi/notebook/casmi_match_probe.ipynb` + `kernel-metadata-probe.json` (id yomeiotani/casmi26-cpp-library-search-probe,
   title "CASMI26 cpp library search probe", datasets + yomeiotani/casmi26-pubchem-whole). Copy both into one folder with the
   metadata named kernel-metadata.json and `kaggle kernels push -p <folder>`. Regenerate the notebook with `python py/make_probe_nb.py`.
3. Log lines to check:
   - `fpnet stage done` (if instead `FPNET DISABLED`, the fallback `PROBE fallback phys_probe compile rc: 0` must appear)
   - `pubchem tables: /kaggle/input/...` (if `PROBE FAILED: pc_mass.bin not found` the run stops — no submission file)
   - `v28 options: ...,pc=1,pcv=2,pcrank=fused,probe=1,pcout=plain_candidates.csv,pcdir=...`
   - `pubchem v2: 112357994 compounds; ...` and `pubchem: ... thread-s ...; window N/query`
   - `C++ PROBE: non-pool PubChem list only: <n> queries, average list length ~24.4, empty lists <few>; ...`
     (the eval MRR there is 0 on Kaggle: no truth)
   - `PROBE submission: <n> queries, average list length ~24.4, full (25) ..., empty ...`
   - `final: rows ...` ; then submit that version once.
   Expected time: as v28b (PubChem v2 ~1.4–2.5 thread-s/query → 400 queries ~4–8 min on 4 cores + table load 1–2 min), memory +1.8 GB.
4. Interpret with the table above.

## Resume
Everything above is finished (16:35, 2026-10-19); nothing is running. Outputs on the build machine: C:\prog\casmi\probe\pr_{hc1,hc0,ha1,hb1}.{csv,_smi.csv,_plain.csv,log}.
Rerun one: `C:\prog\casmi\fwdmodel\probe\run.bat <qry> ylib fwd_mlp_rank_ylib.bin rev2_y.bin <tag> ",rm=1"` (6 threads, ~10-13 min).
Possible improvement (not done): IK14-level exclusion (tautomer/charge-normalized key) would remove the 0.013 leak; rdkit-min has no InChI.
