# speed — profile of the v37 ranker, redundancy counts, phys_v38pc (status file)

2026-10-22 afternoon. Goal: the ranker (phys_v37pc) takes 1,270-1,660 s on Kaggle (4 cores); find where the time goes,
how much of the work is repeated, and remove repeated work without changing a single output byte.
Nothing uploaded / pushed to Kaggle / submitted.

## phys_v38pc (`cpp/rdkit-min/phys_v38pc.cpp` = phys_v37pc + speedups, same options, outputs byte-identical)
- **Profile lines** (stdout): `v38 profile: <stage> <thread-s>` per stage, `v38 profile: enumeration wall ...; setup before
  enumeration ...`, `v38 profile: after enumeration (fusion, merge, output) ...; total ...`, and
  `v38 redundancy: pool lookups / distinct / hits; pubchem window lookups / distinct / hits; pubchem full-scoring
  candidates, stereo-blind duplicates; fp3c vocabulary features per parsed PubChem molecule`.
- **Per-molecule / per-spectrum split**: every candidate's query-independent work (SMILES parse + stereo-blind key,
  physics fragment table `enumerate`, MLP fragments, REV2 bits, fp3 features, CNN bits, ECFP 2048) is computed once into
  a `CandFeat`, and the per-query part (predict + loglik per spectrum, MLP per spectrum, dot products with the query's
  p vectors, libSim, analog Tanimoto) is scored from it with the same operations in the same order.
  fp3::features is computed once per candidate for both the pcfp and the FT vocabularies (v37 computed it twice).
- `pccache=1` (default): PubChem window entries (parse ok + fp3 features found in the pcfp vocabulary, feature order)
  kept per run, keyed by the PubChem table index, in an insert-only 64-shard map (thread-safe); a window member seen by
  an earlier query is scored from the list without parsing. `pccachemax=N` (default 3,000,000 entries, ~0.9 KB of process memory each) caps
  the memory; beyond it entries are computed and not stored. Active only in the production path (pcfp, pcfptop 0, no
  pcrecall, vocabulary <= 65,536).
- `poolcache=1` (default **0**): the same for pool candidates keyed by pool index (CandFeat incl. fragment tables).
  Off by default: only 13% of pool lookups repeat and the fragment tables cost GBs (see below).
- Any combination of the two options gives the phys_v37pc bytes (verified below).

## Profile (build machine, hbqry, chk37b config + ftmodel ft8_ep5 / ftpool 2, 4 threads, MSVC /O2 /arch:AVX2)
The machine was shared (an e2e test, model training, and my 2-thread lane), so absolute times carry ~10% noise.
| stage (thread-s, summed over the 4 threads) | v38 caches off (= v37 path) | pccache + poolcache on |
|---|---|---|
| **PubChem window: parse + fp3 score (14,545 molecules/query)** | **1,418.5 (60%)** | **1,237.8** |
| attention retrieval (index scan + 10 analog fp3) | 414.9 (18%) | 387.3 |
| analog search (stage 1 index + stage 2 modcos + ECFP) | 168.5 (7%) | 152.2 |
| pool window: parse + key + per-molecule features (212 cand/query) | 146.9 (6%) | 122.1 |
| - of which SMILES parse + canonical key | 23.4 | 19.3 |
| - physics fragment tables | 90.1 | 79.1 |
| - MLP fragments | 76.3 | 67.0 |
| - fp3 features (pcfp + FT) | 15.1 | 13.2 |
| - REV2 bits / CNN bits / ECFP 2048 | 2.3 / 2.2 / 0.5 | 1.9 / 1.9 / 0.5 |
| generation (incl. features of the ~75 kept products/query) | 96.9 (4%) | 93.9 |
| score: MLP (per spectrum) | 68.1 (3%) | 85.8 |
| score: libSim + analog Tanimoto | 18.2 | 17.9 |
| score: physics predict + loglik | 5.9 | 5.7 |
| score: bit dot products (REV2 / fp3 / CNN / FT) | 0.4 | 0.4 |
| PubChem full scoring (50/query) / rel key search | 10.4 / 1.8 | 10.0 / 1.8 |
| **timed stages total** | **2,350** | **2,115** |
| enumeration wall (4 threads) | 591.5 s | 533.4 s |
| setup before enumeration (load library + indexes, CNN 103 s, FT 178 s, pcfp 27 s) | 335.5 s | 266.0 s |
| fusion, merge, output | 0.5 s | 0.5 s |
| **total** | **927.5 s** | **799.9 s** |
| final v38 defaults (pccache only, uncapped run b38_fin): PubChem 1,243.3; timed 2,184.7; enumeration wall 550.5 s; total 828.0 s | | |

Second holdout, **hasqry** (2 threads, bT1 3 / bpi 0.1; COCONUT-like, smaller windows 6.9k/query, more generation):
v37 enumeration wall 886.2 s vs v38 (both caches) 889.4 s — **no gain** (hits: PubChem 3.7%, pool 4.9%). v38 stages
(thread-s): PubChem window 727.5 (41%), generation 503.6 (28%, of which most of physics fragments 185 / MLP fragments
140 are the generated products), attention 184.4, pool window 116.0, MLP scoring 96.2, analog 95.1.
(The setup difference is load noise: CNN / FT inference are identical code in both runs. FT on g++ -O3 is ~3x faster
than MSVC: 58.9 s for 1,163 spectra in the e2e run.) On Kaggle the same split holds: the e2e v37 log (800 queries,
g++, 4 threads) has pubchem 1,862 thread-s of a 1,012 s enumeration wall (~46% of 4,050 thread-s), generation 396,
analog 198.

## Redundancy (counted in the run)
| what | lookups | distinct | repeated |
|---|---|---|---|
| pool candidates over the 400 queries (by pool index) | 54,692 | 47,550 | 7,142 (13.1%) |
| PubChem window members over the 400 queries (by table index) | 5,818,261 | 5,158,968 | 654,488 (11.2%) |
| PubChem full-scoring candidates that are stereo-blind duplicates of the main list / each other | 26,890 parsed | | 6,890 (25.6%; cheap: 10 thread-s total) |
| same molecule across spectra of one query | 0 by construction: the windows of a query's spectra are merged (pool: std::set; PubChem: merged ranges) and per-molecule work was already per query, not per spectrum | | |
| pool vs PubChem | PubChem members are only deduplicated (by key) in the full-scoring step, i.e. 25.6% of 50/query — the window score itself is needed for every member (it sets the window mean / sd and the order) | | |
| hasqry: pool / PubChem | 33,751 / 2,766,726 | 32,101 / 2,665,150 | 4.9% / 3.7% |
| hasqry: PubChem full-scoring stereo-blind duplicates | 44,645 parsed | | 25,494 (57%; 21 thread-s total) |
Reading: the candidate sets of different queries barely overlap (different masses); **reuse inside a run can save at
most ~4-13% of the dominant work** (hbqry 11-13%, hasqry 4-5%), and that is what pccache gets (hbqry PubChem stage
1,418 -> 1,238-1,243 thread-s, enumeration 591 -> 533-550 s; hasqry nothing measurable). Memory: the PubChem molecules
carry **372.8 fp3c16k vocabulary features each** (with repeats), so an entry is ~0.75 KB of data; with 5.16M entries the
process went 7.1 GB -> 12.0 GB (b38_fin, pccache only). Hence `pccachemax` default 3,000,000 (~+2.8 GB) in the final
source. The pool cache adds little memory but saves only ~25 thread-s (1%) -> poolcache default off.
**Negative result, stated plainly: v38 is byte-identical and gives the stage profile, but its speed gain is 0-7% of the
enumeration; per-run reuse is not where the time is.**

## Identity (fc /b)
| run | reference (phys_v37pc) | .csv | _ins.csv | _det.txt | _pp.txt | _cf.txt | _mol.bin (ftdump) |
|---|---|---|---|---|---|---|---|
| hbqry 4 thr, v38 poolcache=0 pccache=0 (b38_off) | v37\out\c37b_ft2 (3 thr) | same | same | same | same | same | same |
| hbqry 4 thr, v38 poolcache=1 pccache=1 (b38_on) | c37b_ft2 | same | same | same | same | same | same |
| hbqry 4 thr, v38 pccache only, uncapped (b38_fin) | c37b_ft2 | same | same | same | same | same | same |
| hbqry 4 thr, **final source** (pccachemax 3M, cap reached: 3.0M stored, 433k hits; enumeration 549.1 s, total 814.9 s; b38_c3m) | c37b_ft2 | same | same | same | same | same | same |
| hasqry 2 thr, v38 both caches (s38_on) vs v37 run today (s37_ft2, bT1 3 / bpi 0.1) | v37\out\s37_ft2 | same | same | same | same | same | - |
| hasqry 2 thr, v38 exe (= final logic), pccachemax=1000000 (cap path: 1.0M stored, 73k hits; enumeration 865.5 s vs v37 886.2 s; s38_cap) | s37_ft2 | same | same | same | same | same | - |
MRR (mrr.py) hbqry 0.8129 / 0.8100 in every run = v37.

## The user's signal-processing suggestions — checked, not used
- FFT cross-correlation: modcos evaluates exactly two shifts per pair (0 and the precursor difference) on the stage-1
  shortlist; stage 1 itself is already a sparse correlation of the query against all 2.5M library spectra through an
  inverted 0.01-Da bin index (only the bins of the query's ~50 peaks are touched). A dense FFT over 0.01-Da bins
  (2^18 bins per spectrum) would be orders of magnitude slower for sparse spectra. Analog search is 7% anyway.
- Recursive (Welford) running statistics: the z statistics are O(n) over <= 40k values per query (microseconds) next
  to 3.5 s/query of SMILES parsing; not a bottleneck, and changing the summation would change float bits.
- Prefix sums over sorted masses: the windows are already binary searches in mass-sorted tables, merged into
  contiguous reads; libSim's per-pair norm sums are part of an 18 thread-s stage. Nothing to gain.
- The "difference / reuse" idea is what pccache / poolcache implement; the measurement says the overlap between
  queries is small, so the big reuse has to be **across runs** (offline), below.

## Where a real gain would have to come from (not done; next steps)
40-60% of the enumeration is parsing the PubChem window (~7-15k SMILES per query) and computing their fp3 features,
and 87-96% of those molecules are seen once per run, so only reuse **across runs** (offline) removes it.
- Byte-identical offline lists are **too big**: 372.8 in-vocabulary fp3 features per molecule (with repeats, in feature
  order, which the double sum needs for identical bits) x 2 B x 112.36M PubChem compounds ~ **84 GB**. Not shippable.
- Compact offline form = the distinct indices + multiplicity (sorted, delta-coded): probably ~10x smaller (~8 GB,
  to be measured), but the sum order changes -> last-bit float differences in the window score. Would need an option
  and the "final lists identical or differ only in exact ties, max |score diff|" check the user described.
  Offline cost ~112M x 0.24 ms ~ 7.5 core-hours (~1.3 h on 6 build-machine cores). The production pcfp model and
  ft_all_ep5 share the fp3c16k vocabulary (md5 a15a6010…), so one file serves both; it must carry the vocab md5.
- Cheaper, in-run and exact: a PubChem molecule's score only needs (in, nb) = (sum of p over its features, count);
  nothing to cache there beyond what pccache does. The parse itself (RDKit SmilesToMol with sanitization) is the cost;
  a lighter parser would change features -> not identical.
- Generation (28% on hasqry): the physics / MLP fragment tables of the generated products dominate; products recur
  across queries only when analogs recur — a string-keyed product cache is the analog of poolcache (not measured).
- Attention retrieval (8-18%): inverted-index scan of 57M+57M postings in 0.01-Da bins; candidates for exact speedups
  are a precursor-polarity split of the index and reuse of the 10 analog fp3 lists per library molecule (small).
- Hardware: FT inference is already 3x faster on g++ than MSVC; the setup stage (library/index load, CNN, FT) is
  ~270-360 s here, ~100-150 s on Kaggle.

## Kaggle readiness
- `g++ -std=c++20 -fsyntax-only` with the Kaggle include layout (ds_rdkit + v36 ds_add headers + phys_v38pc):
  **rc 0** (`C:\prog\casmi\speed\kcheck.log`, `cpp/v38/kcheck38.bat`).
- Staging (NOT uploaded): `C:\prog\casmi\speed\ds_add\rdkit-min\phys_v38pc.cpp` (md5 fb04f93dab4bed6f06759a6460a7a4a5)
  -> casmi26-rdkit-min `rdkit-min/`. No new headers, no new data file (ds_rdkit untouched).
- Notebook (v37nofp, cell 7; generator `py/make_v37_nb.py`): the compile loop
  `for _main in (["phys_v37pc.cpp"] if V37 else []) + [_main_old]:` becomes
  `for _main in (["phys_v38pc.cpp"] if V37 and (RDKIT_MIN / "phys_v38pc.cpp").exists() else []) + (["phys_v37pc.cpp"] if V37 else []) + [_main_old]:`
  and the fallback test `if _main == "phys_v37pc.cpp":` stays (a failed v38 compile just falls through to v37).
  Options unchanged (pccache is on by default). After the upload re-run casmi26-cpp-build (it globs phys_v*.cpp, so
  phys_v38pc gets a prebuilt). Memory: up to ~+2.8 GB (pccachemax 3M; add `,pccache=0` to the ranker options for
  v37 memory). Expected Kaggle gain of v38 alone: 0-7% of the enumeration — **not worth an upload on its own**; its
  value is the profile lines. Recommendation: keep v37 in the notebook unless the profile lines are wanted on Kaggle.

## Files
`cpp/rdkit-min/phys_v38pc.cpp`; `cpp/v38/`: build_v38.bat (MSVC), run38.bat / run38b.bat (= run37 with the v38 exe),
lane1.bat (hbqry off/on + fc), lane2.bat (hasqry v37 + v38), lane3.bat (final, hbqry), lane4.bat (final, hasqry, cap),
kcheck38.bat. Outputs `C:\prog\casmi\v38\out\`, references `C:\prog\casmi\v37\out\c37b_ft2*`, `s37_ft2*`.

## Resume
- Re-run a check: on the build machine `powershell -ExecutionPolicy Bypass -File C:\prog\casmi\night\launch.ps1 <name> C:\prog\casmi\v38\lane5.bat`
  (log `C:\prog\casmi\night\<name>.log`, SAME/DIFF lines per file). Exes: `v38\phys_v38pc.exe` (first build, poolcache
  default 1), `v38\b\` (pccachemax 8M), `v38\c\` (final source, md5 1cf1b28f4b98e8c2daa6349461a7fbf0).
- Next measurement if pursued: size of a distinct-index + multiplicity PubChem fp3 file (sample 1M compounds), and the
  float-difference / tie check of that order-changed sum on hbqry + hasqry.
