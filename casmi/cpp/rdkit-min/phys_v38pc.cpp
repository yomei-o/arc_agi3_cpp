// phys_v38pc: phys_v37pc + speedups, outputs byte-identical (speed_STATUS.md): per-molecule work (parse, canonical key,
// fragment tables, fingerprint bit lists) computed once per run per pool index (poolcache=1, default 0: only 13% of
// pool lookups repeat, ~1% of the time) and per PubChem index (pccache=1, default; pccachemax=N
// entries, default 3M ~2.8 GB) and only the per-spectrum part per query; fp3 features computed once for pcfp + FT;
// stage timing ("v38 profile:" / "v38 redundancy:" lines). Every combination gives the v37 output bytes.
// phys_v37pc: phys_v36pc + ftmodel=/ftpool=/ftshuf=/ftdump=/ftbudget= (v37_STATUS.md, pretrain_STATUS.md 8): the own
// masked-peak-pretrained transformer fine-tuned on fp3c16k (TFM1 file, CPU forward of rev/tfm_model.hpp as pt_train
// mollogits) -> per query p = sigmoid(mean logit over its spectra) -> continuous Tanimoto over each candidate's fp3 bits
// in the model's own vocabulary -> + ftpool * z(ft) in the pool fusion (next to cnnpool). cnninst also gates it.
// Defaults (no ftmodel) identical to phys_v36pc.
// phys_v36pc: phys_v35pc + cnnmodel=/cnnpool=/cnnshuf=/cnninst=/cfdump= (pool36_STATUS.md, v36_STATUS.md); defaults identical to v35.
// phys_v35pc: phys_v33pc (Boltzmann merge, alloc=boltz) + phys_v34pc (attention engine, attn=1 / attpool)
// combined (2026-10-21, v35_STATUS.md). Both derived from phys_v32pc; options independent:
// no new options = phys_v32pc, alloc=boltz alone = phys_v33pc, attn=... alone = phys_v34pc.
//
// phys_v33pc: phys_v32pc + a two-population Boltzmann merge of the main list and the fused PubChem
// list (2026-10-21, pc5_STATUS.md; prototype casmi/py/boltz_merge.py):
//   alloc=boltz   replaces the gated fixed slots + promotion (pcv=2). Per gate-open query
//                 (best libSim < gate, PubChem list non-empty):
//                   p_i = (1 - pi) exp(s_i / bT1) / Z1   (main list: s = final score, top bK1 by score)
//                   q_j = pi exp(t_j / bT2) / Z2          (fused PubChem list: t = its 'fused' key)
//                 and the union is ranked by probability (one place per identity: the PubChem list is
//                 already stereo-blind deduplicated against the main list; ties keep the main item
//                 first). The main order itself (incl. the pool-popularity re-fill) is kept; s_i is the
//                 score at that rank before the re-fill. Gate closed -> the plain list (as v30).
//   bT1=x (2) bT2=x (1) bpi=x (0.2) bK1=N (60)
//   ba=x (0)      logit(pi) += ba * relz (relz as v28pc; 0 when there is no pool candidate)
//   Absent alloc = phys_v32pc output.
//
// phys_v34pc: phys_v32pc + a memory-based "attention" engine for the PubChem window (2026-10-21,
// attn_STATUS.md; evaluation tool cpp/attn/attn_eval.cpp). Options (argv[24]; all need pcfp + a library):
//   attn=1        for each query, retrieve the library molecules whose spectra are most similar
//                 (inverted peak index over every library spectrum, no precursor prefilter;
//                 similarity attsim: 0 sqrt-intensity cosine, 1 entropy similarity, 2 shift-aware
//                 cosine, 3 shift-aware entropy (default) = per query peak the better of the
//                 fragment / neutral-loss match, pairs not double-counted, x min(1, matched/attmp)),
//                 per library molecule the max over its spectra and the query's spectra;
//                 weights w = softmax(sim / attT) over the top attk molecules; predicted fp3 bit
//                 probabilities p = sum_j w_j x_j (x_j = the molecule's fp3 vocabulary bits of the pcfp model).
//                 Every PubChem window candidate (and pool candidate, for relz / z statistics) gets
//                 att = continuous Tanimoto of p vs its fp3 bits (as pcfp), and its pcfp score is replaced
//                 by z_win(fp3) + attw * z_win(att) (z over the query's PubChem window).
//   attk=10 attT=0.1 attw=1 attsim=3 attpk=40 (peaks per spectrum) atttol=0.01 (Da) attmp=4
//   attpool=w    also add w * z(att) over the query's pool + generated candidates to the main fusion (default 0).
//   Absent attn (or attn=0) = phys_v32pc output.
//
// phys_v32pc: phys_v30pc + an own fp3 reverse model for the PubChem window (2026-10-20, pc4_STATUS.md):
//   pcfp=<model>  a reverse model on the fp3 vocabulary (cpp/rev3: rev3_train, same REV2 file
//                 format, vocabulary rE = 300; e.g. rev3_all_fp3c16k_big.bin; relative to dir unless
//                 absolute). PubChem window candidates are scored by ITS continuous Tanimoto
//                 sum_{cand fp3 bits} p / (n_bits + sum_all p - sum_{cand bits} p) (p = mean over the
//                 query's spectra, unclipped; as cpp/rev3/win_eval.cpp) instead of REV2's pcrev form.
//                 The pool candidates get the same fp3 score for the window statistics only (relz,
//                 z of the fused PubChem list's rev term); their rev term in the main fusion stays
//                 REV2 (poolrev). Absent = phys_v30pc output.
//   pcfptop=N     fp3 bits only for the top N window candidates by the REV2 pcrev score (the rest
//                 get the lowest computed fp3 score - 1e-6); 0 (default) = every candidate.
//
// phys_v30pc: phys_v28pc + the REV2 score form as an option (2026-10-19, pc3_STATUS.md):
//   pcrev=logit|tan|cal   score of PubChem window candidates (prefilter order, window
//                 z / pcMean / pcSd, relz, S, the rev term of the fused PubChem list).
//                 logit (default) = sum of logit(p_b) over the vocabulary bits = phys_v28pc;
//                 tan = continuous Tanimoto sum_b p_b / (n_bits + sum_all p - sum_b p) with
//                 p = mean over the query's spectra (unclipped); cal = sum of the model's Platt
//                 calibrated logits (calA * logit(p clip 1e-6) + calC, clip +-13.8; plain
//                 logit if the model has no calibration). Exactly cpp/rev3/win_eval.cpp.
//                 Pool candidates' relz / z statistics use the same form.
//   poolrev=logit|tan|cal the pool / generated candidates' rev term in the main fusion
//                 (default logit = phys_v28pc).
//
// phys_v28pc: phys_v27 + the PubChem channel of phys_pc (pc=1, unchanged as pcv=1)
// + a second PubChem channel shaped like the top public pipelines (pcv=2)
// + pool popularity (ppop=w). All off by default = phys_v27 output. (2026-10-19, pc2_STATUS.md)
//
// Pool popularity (argv[24] options):
//   ppop=w        among the top ppn of the final list (after every score term incl.
//                 FPNet), the places held by pool candidates are re-filled in the order
//                 z(final score) + w * pop (z over those pool candidates); generated
//                 candidates keep their places. pop = PubChem log1p(#SID)+log1p(#PMID)
//                 of the pool molecule's stereo-blind match (pool_pop.cpp), 0 if none.
//   ppopf=file    float32 per pool molecule, aligned with <pool>_mols.txt
//                 (default <dir>/pool730_pop.bin; absolute path allowed)
//   ppn=N         re-ordered depth (default 60)
//   ppgate=x      only when the main candidates' best library similarity < x (default off;
//                 0.9 recommended: a library identity match is not overridden by popularity)
//   ppdet=file    per query the top max(ppn,25) before this step (offline grids)
// PubChem channel v2 (pc=1,pcv=2 + the pc/pcdir keys below):
//   every compound of the +-ppm windows gets its REV2 score r (sum of logits over its
//   vocabulary bits); per query z = (r - mean_window) / sd_window.
//   list: z + pcw * pop (pcw default 0.15), stereo-blind duplicates of pool /
//   generated candidates and of each other dropped (pcrank=fused: the top pcn by that,
//   re-ordered by z(hand fusion over them) + pcw * pop).
//   gate: best library similarity of the main candidates < gate (0.9).
//   relz = (best r of a PubChem structure not in the main list - best r of a pool
//   candidate) / sd_window (1e9 when the window has no pool candidate).
//   slots: relz > relhi -> slotshi (default 2:4:6:8:10), else slotslo (4:8:12:16:20);
//   picks = the list's top len(slots).
//   promotion: relz > relp or (S > sp and pop_top >= popp), S = z + pcw * pop of the top
//   pick -> the top pick goes to rank 1 (others keep their slots; noprom=1 disables).
//   Defaults relhi/relp/sp = 1e9 (no dense slots, no promotion) - see pc2_STATUS.md
//   for the chosen values. pcdet adds a "V2 mean sd relz S pop_top r_topNP r_pool havePool" line.
//
// phys_pc: phys_gen + a whole-PubChem candidate channel merged into the final
// list (the top public notebooks' "PubChem slots", here by calibrated score).
//
// Extra keys in the options string (argv[24]):
//   pc=1          enable; tables pc_mass.bin / pc_off.bin / pc_smi.bin /
//                 pc_pop.bin (pc_build.cpp, pop_build.cpp) in pcdir (default <dir>)
//   pcdir=path    directory of the pc_*.bin tables (absolute)
//   pcpre=N       prefilter: per query, every PubChem compound in the +-ppm
//                 windows is parsed and scored rev + pcpp*pop (REV2 vocabulary
//                 bits, popularity); the top N go on (default 300)
//   pcpp=x        popularity weight in the prefilter (default 20)
//   pcn=N         after dropping stereo-blind duplicates, the top N are listed (default 50)
//   pcpop=x       (pcv=1) popularity weight in the PubChem list's final score (default 2)
//   pck=K         (pcv=1) PubChem candidates inserted (default 2)
//   gate=x        insert only when the best library similarity < x (default 0.9)
//   gmarg=x       (pcv=1) ... and the main list's top1 - top2 margin < x (default 1e9)
//   pcmode=score  (pcv=1 default) merge by score (pcsw 1, pcbias -7, pcmin 2); pcmode=slot + slots=a:b:..
//   pcout=file    the inserted list; out.csv stays the plain list. Without pcout, out.csv = inserted
//   pcdet=file    per query gate stats + every listed PubChem candidate
//   pcall=file / pcrecall=1   recall studies (all window candidates / the truth's rank)
//
// phys_v27: phys_gentfm + two hand-set, theory-side changes (2026-10-19):
//   * fragment size (phys_size.cpp): each cut gets the size bin of its smallest
//     piece (heavy atoms 1, 2, 3-4, 5-8, 9-15, 16+) and the physics energy is
//     lowered by sizeW[bin]; sizeW are 6 extra numbers at the end of the params
//     file (phys_size_fqry.txt). An older params file without them = sizeW 0,
//     i.e. exactly phys_gentfm.
//   * libcap=1 (argv[24] option): the library similarity is capped at 1. The
//     sqrt-intensity cosine lets several query peaks reuse one library peak, so
//     it reached 1.86 and 8*lib^32 then forced such candidates to rank 1.
//
// phys_gentfm: phys_gen + the own spectrum -> fingerprint Transformer
// (rev/tfm_model.hpp, TFM1 files from tfm_train, CPU inference) as a reverse
// term. Extra options in the argv[24] string:
//   tfm=a.bin+b.bin  TFM1 models (an ensemble averages the molecule
//                    probabilities). Molecule probability p = mean over the
//                    query's spectra of sigmoid(logit) (as tfm_train eval).
//   tw=x             weight of + tw * z(tfm column `tv`) in the hand fusion
//                    (and after the learned weights); 0 = features only
//   tv=k             column used in the fusion (default: the ensemble's
//                    logit column, or model 0's when only one model)
//   tshuf=1          control: the fusion column is permuted within each query
//                    (seeded), the features keep the true values
//   tsign=-1         control: the fusion column enters with reversed sign
// Columns (appended to the features file after the 6 generation columns, then
// the candidate's SMILES): per model k [sum logit(p_b) over the candidate's
// vocabulary bits (p clipped to [1e-6, 1-1e-6]), tanimoto-style
// sum p_b / (n_bits + sum_all p - sum p_b)], then the ensemble's pair when
// more than one model is given.
//
// phys_gen: phys_ana3 + structure generation from spectral analogs.
//
// Same arguments as phys_ana3 plus one options string after [wf] (argv[24]),
// comma-separated key=value:
//   gen=0|1      generate candidates: the top `nan` analogs (variant `gvar`,
//                default 4 = modified cosine with >= 6 matched peaks) whose
//                precursor-mass difference matches a transformation (or a
//                2-step chain) of gen_ops.h are edited at every valid site;
//                products are deduped (non-isomeric canonical SMILES) against
//                each other and the pool window, mass-checked against the
//                window, ordered by (source sim desc, steps, rank) and capped
//   rm=0|1       evaluation: remove the true structure (and stereo-blind
//                duplicates) from the pool window - "answer not in pool"
//   cap=N        generated candidates per query (default 1000)
//   nan=N        analogs used for generation (default 80)
//   two=0|1      two-step chains (default 1)
//   prior=file   per-op log prior ("op logprior" lines) for the op-prior feature
//   gw=a:b:...   hand weights for the generation features (added to the hand
//                fusion); a learned weights file ([weights.txt]) with 8+NG
//                numbers supplies them instead
//   det=file     per generated candidate details (query key truth sim rank steps nsup nsib chains)
//   thr=N        threads (default: all cores)
//   grid=0       skip the fusion grids
// Generation features (extra columns after the name in [features.txt]):
//   isGen, isGen*sim^3, isGen*(steps==2), isGen*log1p(nSupport),
//   isGen*log1p(nSib), isGen*opPrior
// Output: generated candidates are written as SMILES with `smiles`; without,
// a generated candidate equal to the truth is written under the truth's name
// (others as G<k>), so mrr.py scores the CSV.
//
// phys_ana: phys_rank + shift-aware (modified-cosine) analog search.
//
// Analog search in two stages per query spectrum:
//   stage 1: two inverted indexes over 0.01 Da bins of all library spectra -
//     fragment m/z (catches fragments that do NOT carry the modification:
//     same m/z in query and analog) and neutral loss precursor - m/z (catches
//     fragments that DO carry it: both fragment and precursor move by Delta,
//     so the loss is invariant). Sum of both scores ~ an upper estimate of the
//     modified cosine; the top K library spectra (|Delta| <= 200) go on.
//   stage 2: exact modified cosine (sqrt intensity, 0.01 Da, greedy
//     one-to-one over unshifted and Delta-shifted pairs) on the shortlist.
// Analog feature variants (max_a Tanimoto * sim_a^p, top 80 molecules):
//   0 old binned unshifted (phys_rank), 1 exact unshifted cosine,
//   2 modified cosine, 3 modified cosine without precursor-region peaks,
//   4 modified cosine requiring >= 6 matched peaks, 5-7 variant 3 requiring
//   >= 4/6/8 matched peaks, 8 exact cosine requiring >= 6 matched peaks.
//   extra args after [features.txt]: [anaVariant] [anaPow] [K] [weights.txt]
//   recommended: wa 16-24, variant 4, pow 3, K 8000 (K saturates by 8000;
//   cqry 0.484 -> 0.550, tqry 0.534 -> 0.612, sqry 0.839 -> 0.840)
//
// Physics-shaped fragment intensities instead of a learned MLP.
//
// The learned MLP adds +0.19 MRR over equal-intensity fragments on the
// synthetic holdout (dqry) but only +0.047 on natural products (cqry): what
// it learned is mostly class-specific. This replaces it with ~20 parameters
// shaped by gas-phase ion chemistry (PHYSCHEM_IDEAS.md), fitted on one
// natural-product holdout and evaluated on another:
//
//   intensity(fragment X of cut c, H-shift k)
//       = exp(-E[class(c)]) * share(X | c) * w_k
//   share(X | c) = exp(beta * S(X)) / sum_{Y in c} exp(beta * S(Y))
//
//   - Cleavage propensity: one barrier E per bond class (acyl C-O/C-N,
//     glycosidic, other C-heteroatom, aryl/benzylic C-C, C-C, ring opening,
//     ring opening through an aromatic bond, other), an Arrhenius factor at
//     one effective temperature folded into E.
//   - Charge retention (Cooks' kinetic method): the fragment with the higher
//     proton affinity keeps the proton, ln(I_X/I_Y) = beta (PA_X - PA_Y);
//     in negative mode the more acidic fragment keeps the charge. S(X) is a
//     soft maximum over the fragment's sites from a small functional-group
//     table (kcal/mol).
//   - Hydrogen rearrangement: a weight per shift k in -2..2, per polarity.
//
// Scoring is the background-mixture log-likelihood of pool_candidate_rank
// (loglik), mean over the molecule's spectra.
//
//   phys_rank fit  <dir> <qry> <pool_stem> <ppm> <params_out> [params_init] [instrument]
//   phys_rank eval <dir> <qry> <pool_stem> <ppm> <params_in> [out.csv] [mlp.bin] [w] [smiles]
//                  [rev.bin] [wr] [lib_stem] [wl] [lib_pow] [wa] [features.txt]
#include <algorithm>
#include <atomic>
#include <deque>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/PeriodicTable.h>
#include <GraphMol/RingInfo.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "forward_model.h"
#include "forward_model_nn.h"
#include "rev/ecfp.h"  // copies of ../rev headers (the dataset ships them inside rdkit-min/)
#include "rev/rev_model.hpp"
#include "rev/fp2.h"           // REV2 models: fp2 vocabulary fingerprints
#include "rev/rev2_model.hpp"
#include "rev/fp3_v32.h"      // v32pc: fp3 features (copy of cpp/rev3/fp3.h)
#include "rev/parallel.hpp"
#include "rev/tfm_model.hpp"     // own Transformer (CPU build of the same source)
#include "rev/cnn_model.hpp"     // v36pc: 1-D CNN reverse model (copy of cpp/cnn headers)
#include "gen_ops.h"
#include <array>
#include <map>
#include <sstream>

namespace {

struct Head { float precursor; int32_t mol, off, n, tag; };

std::vector<char> slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(1); }
  std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<char> buf((size_t(n)));
  f.read(buf.data(), n);
  return buf;
}

constexpr float ADDUCT_SHIFT[10] = {
    1.007276f, -1.007276f, 22.989218f, 18.033823f, 38.963158f,
    34.969402f, 44.998201f, -17.002740f, 1.007276f, 22.989218f,
};
constexpr bool ADDUCT_NEG[10] = {false, true, false, false, false, true, true, false, false, false};
constexpr bool ADDUCT_METAL[10] = {false, false, true, false, true, false, false, false, false, true};

// ---- v34pc: attention retrieval engine (casmi/cpp/attn/attn_eval.cpp, attn_STATUS.md) ----
// Peaks below precursor - 1, top attPk by intensity; s = sqrt(a) (a sum-normalised), e = entropy-weighted
// (Li 2021: a^(0.25 + 0.25 S) when S < 3), sum 1.
struct AttPk { float mz, s, e; };
void attPrep(float prec, const float* mz, const float* it, int n, int maxPk, std::vector<AttPk>& out) {
  out.clear();
  std::vector<int> idx;
  for (int i = 0; i < n; ++i)
    if (it[i] > 0.f && mz[i] > 0.f && (prec <= 0.f || mz[i] < prec - 1.0f)) idx.push_back(i);
  if (int(idx.size()) > maxPk) {
    std::nth_element(idx.begin(), idx.begin() + maxPk, idx.end(), [&](int a, int b) { return it[a] > it[b]; });
    idx.resize(maxPk);
  }
  if (idx.empty()) return;
  double sum = 0;
  for (int i : idx) sum += it[i];
  std::vector<double> a(idx.size());
  for (size_t k = 0; k < idx.size(); ++k) a[k] = it[idx[k]] / sum;
  double S = 0;
  for (double x : a) if (x > 0) S -= x * std::log(x);
  std::vector<double> e = a;
  if (S < 3.0) {
    double w = 0.25 + 0.25 * S, s2 = 0;
    for (auto& x : e) { x = std::pow(x, w); s2 += x; }
    for (auto& x : e) x /= s2;
  }
  for (size_t k = 0; k < idx.size(); ++k) out.push_back({mz[idx[k]], float(std::sqrt(a[k])), float(e[k])});
}
// entropy-similarity kernel of one matched pair: (a+b)ln(a+b) - a ln a - b ln b, over ln 4
inline float attGEnt(float a, float b) {
  double s = double(a) + b;
  return float((s * std::log(s) - double(a) * std::log(double(a)) - double(b) * std::log(double(b))) / std::log(4.0));
}
struct AttPost { uint32_t spec; float v, s, e; };
struct AttIndex {
  float bw = 0.01f; int nb = 0;
  std::vector<uint64_t> off;
  std::vector<AttPost> p;
};

int adductIdx(int32_t tag) {
  if (tag < 0) return -1;
  return tag % 16 == 15 ? tag / 16 + 1 : tag / 16;
}

// ---- site tables -------------------------------------------------------

bool hasDoubleO(const RDKit::ROMol& m, const RDKit::Atom* c) {
  for (auto b : m.atomBonds(c)) {
    const RDKit::Atom* o = b->getOtherAtom(c);
    if (o->getAtomicNum() == 8 && b->getBondType() == RDKit::Bond::DOUBLE) return true;
  }
  return false;
}

// Gas-phase proton affinity of a protonation site, kcal/mol (0 = not a site).
float sitePA(const RDKit::ROMol& m, const RDKit::Atom* a) {
  int z = a->getAtomicNum();
  if (z == 7) {
    if (a->getIsAromatic()) return a->getTotalNumHs() > 0 ? 190.f : 222.f;  // pyrrole / pyridine
    bool amide = false, aryl = false, imine = false;
    int nC = 0;
    for (auto b : m.atomBonds(a)) {
      const RDKit::Atom* o = b->getOtherAtom(a);
      if (b->getBondType() == RDKit::Bond::DOUBLE) imine = true;
      if (o->getAtomicNum() == 6) {
        ++nC;
        if (hasDoubleO(m, o)) amide = true;
        if (o->getIsAromatic()) aryl = true;
      }
    }
    if (amide) return 200.f;   // amides protonate on O (below)
    if (imine) return 225.f;
    if (aryl) return 210.f;
    return 214.f + 4.f * float(std::min(nC, 3));  // primary < secondary < tertiary
  }
  if (z == 8) {
    bool dbl = false;
    for (auto b : m.atomBonds(a))
      if (b->getBondType() == RDKit::Bond::DOUBLE) {
        dbl = true;
        const RDKit::Atom* c = b->getOtherAtom(a);
        bool nNb = false, oNb = false, arNb = false;
        for (auto b2 : m.atomBonds(c)) {
          const RDKit::Atom* x = b2->getOtherAtom(c);
          if (x == a) continue;
          if (x->getAtomicNum() == 7) nNb = true;
          if (x->getAtomicNum() == 8) oNb = true;
          if (x->getIsAromatic()) arNb = true;
        }
        if (nNb) return 212.f;              // amide carbonyl
        if (oNb) return 196.f;              // ester / acid carbonyl
        return arNb ? 205.f : 197.f;        // conjugated / plain ketone
      }
    if (!dbl) {
      bool ar = false;
      for (auto b : m.atomBonds(a)) if (b->getOtherAtom(a)->getIsAromatic()) ar = true;
      if (ar) return 195.f;                  // phenol / aryl ether
      return a->getTotalNumHs() > 0 ? 189.f : 192.f;  // alcohol / ether
    }
  }
  if (z == 16) return 200.f;
  if (z == 6 && a->getIsAromatic()) return 185.f;
  return 0.f;
}

// Deprotonation favourability, 400 - gas-phase acidity (kcal/mol); larger =
// more acidic (0 = not a site).
float siteAcid(const RDKit::ROMol& m, const RDKit::Atom* a) {
  int z = a->getAtomicNum();
  if (a->getTotalNumHs() == 0) return 0.f;
  if (z == 8) {
    for (auto b : m.atomBonds(a)) {
      const RDKit::Atom* c = b->getOtherAtom(a);
      if (c->getAtomicNum() == 6 && hasDoubleO(m, c)) return 55.f;      // COOH
      if (c->getAtomicNum() == 16 || c->getAtomicNum() == 15) return 70.f;  // S/P acids
      if (c->getIsAromatic()) return 50.f;                              // phenol
    }
    return 28.f;                                                        // alcohol
  }
  if (z == 7) {
    for (auto b : m.atomBonds(a)) {
      const RDKit::Atom* c = b->getOtherAtom(a);
      if ((c->getAtomicNum() == 6 && hasDoubleO(m, c)) || c->getAtomicNum() == 16) return 40.f;  // amide/sulfonamide NH
    }
    return a->getIsAromatic() ? 40.f : 10.f;
  }
  if (z == 16) return 45.f;                                             // thiol
  return 0.f;
}

// ---- bond classes ------------------------------------------------------

// CGLYC: the sugar-aglycone C-C bond of a C-glycoside (split out of ARYLC/CC
// to test whether the fit recovers "C-glycosides are harder to cleave than
// O-glycosides" by itself).
enum Cls { INTACT = 0, ACYL, GLYCOSIDIC, CHET, ARYLC, CC, RING, RINGAR, OTHER, CGLYC, NCLS };

bool anomeric(const RDKit::ROMol& m, const RDKit::Atom* c) {
  if (c->getAtomicNum() != 6 || c->getIsAromatic()) return false;
  int nO = 0;
  for (auto b : m.atomBonds(c))
    if (b->getOtherAtom(c)->getAtomicNum() == 8 && b->getBondType() == RDKit::Bond::SINGLE) ++nO;
  return nO >= 2 && m.getRingInfo()->numAtomRings(c->getIdx()) > 0;
}

// sp3 ring carbon carrying exactly one (ring) oxygen: a sugar's C1 when the
// aglycone is attached through carbon
bool cAnomeric(const RDKit::ROMol& m, const RDKit::Atom* c) {
  if (c->getAtomicNum() != 6 || c->getIsAromatic()) return false;
  const RDKit::RingInfo* ri = m.getRingInfo();
  if (ri->numAtomRings(c->getIdx()) == 0) return false;
  int nO = 0, nRingO = 0;
  for (auto b : m.atomBonds(c)) {
    const RDKit::Atom* o = b->getOtherAtom(c);
    if (o->getAtomicNum() == 8 && b->getBondType() == RDKit::Bond::SINGLE) {
      ++nO;
      if (ri->numBondRings(b->getIdx()) > 0) ++nRingO;
    }
  }
  return nO == 1 && nRingO == 1;
}

int bondClass(const RDKit::ROMol& m, const RDKit::Bond* b) {
  const RDKit::Atom* x = b->getBeginAtom();
  const RDKit::Atom* y = b->getEndAtom();
  int zx = x->getAtomicNum(), zy = y->getAtomicNum();
  if (b->getBondType() != RDKit::Bond::SINGLE) return OTHER;
  if (zx == 6 && zy == 6) {
    if ((cAnomeric(m, x) && y->getIsAromatic()) || (cAnomeric(m, y) && x->getIsAromatic())) return CGLYC;
    return (x->getIsAromatic() || y->getIsAromatic()) ? ARYLC : CC;
  }
  const RDKit::Atom* c = zx == 6 ? x : (zy == 6 ? y : nullptr);
  const RDKit::Atom* h = c == x ? y : x;
  if (!c) return OTHER;
  int zh = h->getAtomicNum();
  if ((zh == 8 || zh == 7) && hasDoubleO(m, c)) return ACYL;
  if (zh == 8 && anomeric(m, c)) return GLYCOSIDIC;
  if (zh == 8 || zh == 7 || zh == 16) return CHET;
  return OTHER;
}

// ---- per-candidate fragment table --------------------------------------

struct Frag {
  float mass;   // neutral sub-graph mass with its own hydrogens
  uint8_t cls;
  float pa, paOther;      // soft-max site PA of this fragment / its partner
  float acid, acidOther;  // same for acidity
  uint8_t cls2 = 255;     // second cut's class for a two-bond (sequential) cleavage
  uint8_t sizeBin = 255;  // phys_size: heavy atoms of the smallest piece of the cut, binned
};

float softmax(const std::vector<float>& v, float t = 3.0f) {
  // log-sum-exp at "temperature" t kcal/mol: a fragment with two good sites
  // is slightly more basic than one with a single site
  float mx = 0.f;
  for (float x : v) mx = std::max(mx, x);
  if (mx <= 0.f) return 0.f;
  double s = 0;
  for (float x : v) if (x > 0.f) s += std::exp((x - mx) / t);
  return mx + t * float(std::log(s));
}

std::vector<Frag> enumerate(const RDKit::ROMol& mol) {
  std::vector<Frag> out;
  const int n = int(mol.getNumAtoms());
  std::vector<float> pa(n), ac(n);
  for (int i = 0; i < n; ++i) {
    pa[i] = sitePA(mol, mol.getAtomWithIdx(i));
    ac[i] = siteAcid(mol, mol.getAtomWithIdx(i));
  }
  auto score = [&](const std::vector<int>& idx, const std::vector<float>& s) {
    std::vector<float> v;
    for (int i : idx) if (s[i] > 0.f) v.push_back(s[i]);
    return softmax(v);
  };
  std::vector<int> all(n);
  for (int i = 0; i < n; ++i) all[i] = i;
  out.push_back({float(forward_model::fragmentMass(mol, all)), INTACT, 0.f, 0.f, 0.f, 0.f});

  auto addCut = [&](const std::vector<unsigned>& bonds, int cls, int cls2 = 255) {
    auto frags = forward_model::fragmentsAfterCutting(mol, bonds);
    if (frags.size() < 2) return;
    std::vector<float> p(frags.size()), a(frags.size());
    for (size_t i = 0; i < frags.size(); ++i) { p[i] = score(frags[i], pa); a[i] = score(frags[i], ac); }
    // phys_size: how big is the smallest piece split off by this cut?
    // (observed on fqry: 1 heavy atom 18%, 2: 24%, 3-4: 47%, 5-8: 59%,
    // 9-15: 61%, >=16: 34% of cuts show a fragment; chance matches <1%)
    size_t nmin = 1000;
    for (auto& f : frags) nmin = std::min(nmin, f.size());
    uint8_t sb = nmin <= 1 ? 0 : nmin <= 2 ? 1 : nmin <= 4 ? 2 : nmin <= 8 ? 3 : nmin <= 15 ? 4 : 5;
    for (size_t i = 0; i < frags.size(); ++i) {
      // partner = best competing fragment of the same cut
      float po = 0.f, ao = 0.f;
      for (size_t j = 0; j < frags.size(); ++j)
        if (j != i) { po = std::max(po, p[j]); ao = std::max(ao, a[j]); }
      out.push_back({float(forward_model::fragmentMass(mol, frags[i])), uint8_t(cls), p[i], po, a[i], ao,
                     uint8_t(cls2), sb});
    }
  };
  const RDKit::RingInfo* ri = mol.getRingInfo();
  std::vector<std::pair<unsigned, int>> labile;
  for (const auto& b : mol.bonds()) {
    if (ri->numBondRings(b->getIdx()) > 0) continue;
    int c = bondClass(mol, b);
    addCut({unsigned(b->getIdx())}, c);
    if (c == ACYL || c == GLYCOSIDIC || c == CHET) labile.emplace_back(b->getIdx(), c);
  }
  // Two labile bonds at once: the sequential losses natural products are
  // full of (both sugars of a bis-glycoside, sugar + acyl group, ...), which
  // a single cut can never produce. Capped to keep the cost bounded.
  if (labile.size() > 12) labile.resize(12);
  for (size_t i = 0; i < labile.size(); ++i)
    for (size_t j = i + 1; j < labile.size(); ++j)
      addCut({labile[i].first, labile[j].first}, labile[i].second, labile[j].second);
  for (unsigned r = 0; r < ri->numRings(); ++r) {
    const auto& rb = ri->bondRings()[r];
    for (size_t i = 0; i < rb.size(); ++i)
      for (size_t j = i + 1; j < rb.size(); ++j) {
        bool ar = mol.getBondWithIdx(rb[i])->getIsAromatic() || mol.getBondWithIdx(rb[j])->getIsAromatic();
        addCut({unsigned(rb[i]), unsigned(rb[j])}, ar ? RINGAR : RING);
      }
  }
  return out;
}

// ---- parameters ----------------------------------------------------------

struct Params {
  float E[NCLS] = {0.f, 0.5f, 0.3f, 1.0f, 1.5f, 2.0f, 2.0f, 4.0f, 3.0f, 1.5f};  // E[INTACT] = -log(intact weight)
  float betaPos = 0.15f, betaNeg = 0.10f;  // 1/kcal
  float kPos[5] = {-3.f, -1.f, -0.5f, 0.f, -2.f};  // log weight of H shift -2..2
  float kNeg[5] = {-1.f, 0.f, -0.5f, -2.f, -3.f};
  float logEps = std::log(0.3f);
  float E2 = 1.0f;  // extra barrier for the second cut of a two-bond cleavage
  // size effect: every cut's barrier + dofSlope * ln(heavy atoms / 30). The
  // degrees-of-freedom argument predicts dofSlope > 0 - at one collision
  // energy a bigger ion is internally colder and fragments less.
  float dofSlope = 0.0f;
  float sizeW[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};  // phys_size: log-weight per size bin of the smallest piece

  // file order keeps older files loadable: the original classes, betas,
  // k weights, eps, E2, then E[CGLYC] and dofSlope (added later)
  static constexpr int N = (NCLS - 1) + 2 + 5 + 5 + 1 + 1 + 1 + 1 + 6;
  float* at(int i) {
    if (i < NCLS - 1) return &E[i];
    i -= NCLS - 1;
    if (i == 0) return &betaPos;
    if (i == 1) return &betaNeg;
    i -= 2;
    if (i < 5) return &kPos[i];
    i -= 5;
    if (i < 5) return &kNeg[i];
    i -= 5;
    if (i == 0) return &logEps;
    if (i == 1) return &E2;
    if (i == 2) return &E[CGLYC];
    if (i == 3) return &dofSlope;
    return &sizeW[i - 4];
  }
  void save(const std::string& p) {
    std::ofstream f(p);
    for (int i = 0; i < N; ++i) f << *at(i) << "\n";
  }
  bool load(const std::string& p) {
    std::ifstream f(p);
    // files written before E2 existed hold N-1 values; E2 keeps its default
    for (int i = 0; i < N; ++i) if (!(f >> *at(i))) return i >= N - 9;  // older files: no E2/.../sizeW
    return true;
  }
};

constexpr double MASS_H = 1.00782503207, ELECTRON = 0.00054858;

// Predicted (m/z, intensity) for one polarity / adduct.
void predict(const std::vector<Frag>& fr, const Params& P, int adduct, int nHeavy,
             std::vector<std::pair<double, double>>& out) {
  const double dof = P.dofSlope * std::log(std::max(1, nHeavy) / 30.0);
  out.clear();
  bool neg = adduct >= 0 && adduct < 10 && ADDUCT_NEG[adduct];
  bool metal = adduct >= 0 && adduct < 10 && ADDUCT_METAL[adduct];
  const float* kw = neg ? P.kNeg : P.kPos;
  double z = neg ? -1.0 : 1.0;
  for (const Frag& f : fr) {
    double share = 1.0;
    if (f.cls != INTACT) {
      double d = neg ? P.betaNeg * (f.acid - f.acidOther) : P.betaPos * (f.pa - f.paOther);
      if (metal) d = 0.0;  // a metal cation does not follow proton affinity
      share = 1.0 / (1.0 + std::exp(-d));
    }
    double e = P.E[f.cls] + (f.cls2 != 255 ? P.E[f.cls2] + P.E2 : 0.f) + (f.cls != INTACT ? dof : 0.0)
               - (f.sizeBin < 6 ? P.sizeW[f.sizeBin] : 0.f);
    double base = std::exp(-e) * share;
    for (int k = -2; k <= 2; ++k)
      out.emplace_back(f.mass + k * MASS_H - z * ELECTRON, base * std::exp(kw[k + 2]));
    if (metal) {
      double mm = adduct == 4 || adduct == 9 ? (adduct == 4 ? 38.963158 : 22.989218) : 22.989218;
      out.emplace_back(f.mass + mm, base * 0.5);
    }
  }
  std::sort(out.begin(), out.end());
}

float loglik(const std::vector<std::pair<double, double>>& pred, const float* mz, const float* it,
             int n, float tol, float eps, float range) {
  double tot = 0.0, wsum = 0.0;
  for (auto& pk : pred) tot += pk.second;
  for (int i = 0; i < n; ++i) wsum += std::sqrt(std::max(0.0, double(it[i])));
  if (tot <= 0.0 || wsum <= 0.0) return -1e9f;
  double bg = eps * 2.0 * tol / std::max(1.0f, range);
  double ll = 0.0;
  for (int i = 0; i < n; ++i) {
    auto lo = std::lower_bound(pred.begin(), pred.end(), std::make_pair(double(mz[i]) - tol, -1e300));
    double m = 0.0;
    for (auto p = lo; p != pred.end() && p->first <= mz[i] + tol; ++p) m += p->second;
    double w = std::sqrt(std::max(0.0, double(it[i]))) / wsum;
    ll += w * std::log((1.0 - eps) * m / tot + bg);
  }
  return float(ll);
}

// ---- data ----------------------------------------------------------------

struct Query {
  std::string name;
  std::vector<int> spectra;
  std::vector<int> cand;     // pool indices
  std::vector<std::vector<Frag>> frags;
  std::vector<int> candN;     // heavy atoms per candidate
  int truth = -1;            // position of the true structure in cand, -1 if absent
  std::vector<float> mlpCos; // learned forward MLP's best cosine, when a model is given
  std::vector<float> revLL;  // reverse model: sum of logit(p_bit) over the candidate's ECFP bits
  std::vector<float> logit;  // reverse model's per-bit logits for this query (mean p over spectra)
  // v30pc (pcrev / poolrev != logit): unclipped mean p per bit, its sum, calibrated per-bit weights
  std::vector<float> revP, revCalU;
  double revSumP = 0.0;
  std::vector<float> revPc;  // pool candidates' REV2 score in the pcrev form (when it differs from revLL's)
  // v32pc (pcfp): the fp3 model's mean p per bit (unclipped) and its sum
  std::vector<float> fpP;
  double fpSumP = 0.0;
  // v34pc (attn): attention-predicted fp3 bit probabilities, their sum, pool candidates' att score
  std::vector<float> attP, attPc;
  std::vector<float> cnnP, cnnPc;  // v36pc: CNN mean p per bit (query), candidates' continuous Tanimoto
  double cnnSumP = 0.0;
  std::vector<float> ftP, ftPc;  // v37pc: fine-tuned transformer p per fp3 bit (query), candidates' continuous Tanimoto
  double ftSumP = 0.0;
  double attSumP = 0.0;
  int attN = 0; float attTop = 0.f;
  std::vector<float> libSim; // best spectral similarity to the candidate's own library spectra (0 if none)
  std::vector<float> anaP;   // analog propagation: max_a Tanimoto(cand, a) * sim_a^3 over spectral analogs a
  std::vector<float> anaV[9][3];  // variant x pow {2,3,4}
  // generation: cand[i] < 0 is generated candidate gi = -cand[i] - 1
  std::vector<std::string> gSmi;
  std::vector<std::array<float, 6>> gF;  // per candidate (0 for pool candidates)
  std::vector<float> phys;               // physics score computed in the enumeration pass (eval)
 std::string tsmi;                      // true structure's SMILES (evaluation; may be empty)
  int nGenRaw = 0;                       // distinct products before pool dedupe / cap
  bool truthGenRaw = false, truthGenKept = false;
  std::string det;                       // generated-candidate detail lines
  std::vector<std::vector<float>> tP;    // TFM: molecule probability per model (+ ensemble)
  std::vector<std::vector<float>> tcol;  // TFM columns per candidate
  // PubChem channel
  struct PcCand {
    int32_t cid; std::string smi; bool truth;
    float rev, pop, phys, mlp, ana, fused;
    float zp = 0, zm = 0, zr = 0;
  };
  std::vector<PcCand> pc;                // fully scored, sorted by fused desc
  float libMax = 0.f, top1 = 0.f, top2 = 0.f, anaTop = 0.f;
  std::vector<float> topS;               // final scores of the main list's top 25
  int pcWin = 0, pcTruthWin = -1, pcTruthPre = -1;  // window size; truth's rank in window (rev+pop) / -1
  bool pcTruthInWin = false, pcTruthKept = false;
  std::string pcAll;
  // pcv=2: window rev statistics, confidence quantities of the top pick
  float pcMean = 0.f, pcSd = 1.f, pcRelz = 0.f, pcS = -1e9f, pcTopPop = 0.f, pcRevTopNP = 0.f, poolRevBest = 0.f;
  bool pcHavePool = false;
};
constexpr int NG = 6;

constexpr int NVAR = 9;
const char* VARNAME[NVAR] = {"binned", "cos", "mcos", "mcos-noprec", "mcos-6pk",
                            "noprec-4pk", "noprec-6pk", "noprec-8pk", "cos-6pk"};

// Exact modified cosine between two spectra (peaks sorted by m/z).
// Weights are sqrt(I)/||sqrt(I)||. Pairs: |q - l| <= tol (unshifted) or
// |q - delta - l| <= tol (shifted, delta = precQ - precL); greedy one-to-one
// by descending product. Fills cos (unshifted only), mcos, mcosNP (precursor
// region peaks m/z > prec - 1 removed from both, renormalized), nMatch.
struct McosOut { float cos, mcos, mcosNP; int nMatch, nMatchU, nMatchNP; };
struct PairT { float s; int i, j; uint8_t shifted, prec; };
void modcos(const float* qm, const float* qw, int nq, float qprec, float qnNP,
            const float* lm, const float* li, int nl, float lprec, float lnrm,
            float tol, std::vector<PairT>& pairs, std::vector<uint8_t>& usedQ, std::vector<uint8_t>& usedL,
            McosOut& o) {
  float delta = qprec - lprec;
  pairs.clear();
  // lib weights on the fly
  auto lw = [&](int j) { return std::sqrt(std::max(0.f, li[j])) / lnrm; };
  double lnNP2 = 0;  // squared norm of non-precursor lib peaks (in normalized units)
  for (int j = 0; j < nl; ++j) if (lm[j] <= lprec - 1.f) { float w = lw(j); lnNP2 += double(w) * w; }
  for (int pass = 0; pass < 2; ++pass) {
    float sh = pass ? delta : 0.f;
    if (pass && std::fabs(delta) <= tol) break;  // identical to unshifted
    int j0 = 0;
    for (int i = 0; i < nq; ++i) {
      float m = qm[i] - sh;
      while (j0 < nl && lm[j0] < m - tol) ++j0;
      for (int j = j0; j < nl && lm[j] <= m + tol; ++j) {
        uint8_t pr = (qm[i] > qprec - 1.f) || (lm[j] > lprec - 1.f);
        pairs.push_back({qw[i] * lw(j), i, j, uint8_t(pass), pr});
      }
    }
  }
  std::sort(pairs.begin(), pairs.end(), [](const PairT& a, const PairT& b) { return a.s > b.s; });
  usedQ.assign(nq, 0); usedL.assign(nl, 0);
  // three greedy passes: all pairs, unshifted only, non-precursor only
  double sm = 0; int nm = 0;
  for (auto& p : pairs) if (!usedQ[p.i] && !usedL[p.j]) { usedQ[p.i] = usedL[p.j] = 1; sm += p.s; ++nm; }
  o.mcos = float(std::min(1.0, sm)); o.nMatch = nm;
  usedQ.assign(nq, 0); usedL.assign(nl, 0);
  double su = 0; int nu = 0;
  for (auto& p : pairs) if (!p.shifted && !usedQ[p.i] && !usedL[p.j]) { usedQ[p.i] = usedL[p.j] = 1; su += p.s; ++nu; }
  o.cos = float(std::min(1.0, su)); o.nMatchU = nu;
  usedQ.assign(nq, 0); usedL.assign(nl, 0);
  double sn = 0; int nn = 0;
  for (auto& p : pairs) if (!p.prec && !usedQ[p.i] && !usedL[p.j]) { usedQ[p.i] = usedL[p.j] = 1; sn += p.s; ++nn; }
  o.nMatchNP = nn;
  double den = double(qnNP) * std::sqrt(lnNP2);
  o.mcosNP = den > 1e-9 ? float(std::min(1.0, sn / den)) : 0.f;
}

// Same cosine as pool_candidate_rank (sqrt-intensity, 0.01 Da).
float mlpCosine(const std::vector<std::pair<double, double>>& pred, const float* mz, const float* it, int n) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (auto& pk : pred) na += pk.second;
  for (int i = 0; i < n; ++i) nb += double(it[i]);
  if (na <= 0.0 || nb <= 0.0) return 0.0f;
  for (auto& pk : pred) {
    double best = 0.0;
    for (int i = 0; i < n; ++i)
      if (std::fabs(double(mz[i]) - pk.first) <= 0.01) best = std::max(best, double(it[i]));
    if (best > 0.0) dot += std::sqrt(pk.second) * std::sqrt(best);
  }
  return float(dot / std::sqrt(na * nb + 1e-12));
}

// ---- TFM input (as tfm_train: FPNet prep_peaks: mz <= prec+2, I/Imax >= 1e-3,
// top maxPeaks by intensity kept in m/z order, sqrt(I/Imax))
constexpr float TFM_MODE[10] = {1, -1, 1, 1, 1, -1, -1, 1, 1, 1};
void tfmPrep(const Head& h, const float* mz, const float* it, int maxP, std::vector<float>& om, std::vector<float>& oi) {
  om.clear(); oi.clear();
  float mx = 0.f;
  for (int i = 0; i < h.n; ++i) if (mz[i] > 0.f && mz[i] <= h.precursor + 2.f) mx = std::max(mx, it[i]);
  if (mx > 0.f) {
    std::vector<int> idx;
    for (int i = 0; i < h.n; ++i)
      if (mz[i] > 0.f && mz[i] <= h.precursor + 2.f && it[i] / mx >= 1e-3f) idx.push_back(i);
    if (int(idx.size()) > maxP) {
      std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return it[a] > it[b]; });
      idx.resize(maxP);
      std::sort(idx.begin(), idx.end());
    }
    for (int i : idx) { om.push_back(mz[i]); oi.push_back(std::sqrt(it[i] / mx)); }
  }
  if (om.empty()) { om.push_back(0.f); oi.push_back(0.f); }
}
void tfmTag(int tag, int& ad, int& in, float& mode) {
  int a = -1, s = -1;
  if (tag >= 0) {
    bool unkInstr = tag % 16 == 15;
    a = unkInstr ? tag / 16 + 1 : tag / 16;
    s = unkInstr ? -1 : tag % 16;
  }
  ad = (a >= 0 && a < 10) ? a : 10;
  in = (s >= 0 && s < 5) ? s : 5;
  mode = (a >= 0 && a < 10) ? TFM_MODE[a] : 0.f;
}

std::vector<float> zscore(const std::vector<float>& v) {
  double m = 0, s = 0;
  for (float x : v) m += x;
  m /= double(std::max<size_t>(1, v.size()));
  for (float x : v) s += (x - m) * (x - m);
  s = std::sqrt(s / double(std::max<size_t>(1, v.size()))) + 1e-6;
  std::vector<float> z(v.size());
  for (size_t i = 0; i < v.size(); ++i) z[i] = float((v[i] - m) / s);
  return z;
}

// v38pc: the per-molecule (query-independent) part of a candidate's scoring, computed once per run per
// pool index (poolcache=1) instead of once per query whose window contains it; scored per query from these lists.
struct CandFeat {
  bool ok = false;
  std::string key;                                  // canon() stereo-blind key (only when keys are needed)
  int nAtoms = 0;
  std::vector<Frag> fr;                             // physics fragment table (enumerate)
  std::vector<forward_model_nn::FragFeatures> nf;   // MLP fragments
  std::vector<uint16_t> revBits, cnnBits, ecfp;     // REV2 (or ECFP revNB) bits, CNN vocab bits, ECFP r2 2048
  std::vector<int32_t> fpI, ftI;                    // fp3 features found in the pcfp / FT vocabularies (feature order)
};
// v38pc: PubChem window entry (pccache=1): parse ok + the fp3 features found in the pcfp vocabulary (feature order)
struct PcFeat {
  bool ok = false;
  std::vector<uint16_t> fpI;
};
// insert-only sharded map; element addresses stay valid (unordered_map nodes), so get/put return plain pointers
template <class V> struct ShardCache {
  static constexpr size_t NS = 64;
  std::mutex mu[NS];
  std::unordered_map<int64_t, V> m[NS];
  std::atomic<long> hits{0}, miss{0}, n{0};
  const V* get(int64_t k) {
    const size_t s = size_t(k) % NS;
    std::lock_guard<std::mutex> l(mu[s]);
    auto i = m[s].find(k);
    if (i == m[s].end()) { ++miss; return nullptr; }
    ++hits;
    return &i->second;
  }
  const V* put(int64_t k, V&& v) {
    const size_t s = size_t(k) % NS;
    std::lock_guard<std::mutex> l(mu[s]);
    auto r = m[s].emplace(k, std::move(v));
    if (r.second) ++n;
    return &r.first->second;
  }
  size_t size() {
    size_t n = 0;
    for (size_t s = 0; s < NS; ++s) { std::lock_guard<std::mutex> l(mu[s]); n += m[s].size(); }
    return n;
  }
};
// v38pc profile: thread-seconds per stage (summed over threads)
enum { PR_ANA, PR_ATT, PR_POOLPARSE, PR_GEN, PR_F_PARSE, PR_F_FRAG, PR_F_NF, PR_F_REV, PR_F_FP3, PR_F_CNN, PR_F_ECFP,
       PR_S_PHYS, PR_S_MLP, PR_S_BITS, PR_S_LIB, PR_PC_WIN, PR_PC_REL, PR_PC_FULL, PR_TFM, PR_N };
const char* PR_NAME[PR_N] = {"analog search", "attention retrieval", "pool window parse+key (+feat on miss)", "generation",
                             "feat: SMILES parse+canon (gen/pool)", "feat: physics fragments", "feat: MLP fragments",
                             "feat: REV2 bits", "feat: fp3 features (pcfp+FT)", "feat: CNN bits", "feat: ECFP 2048",
                             "score: physics predict+loglik", "score: MLP", "score: bit dot products", "score: libSim+analog",
                             "pubchem window parse+fp3 score", "pubchem rel (top non-pool key)", "pubchem full scoring",
                             "tfm columns"};
inline double secSince(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

}  // namespace

int main(int argc, char** argv) {
  const auto tMain0 = std::chrono::steady_clock::now();
  if (argc < 7) {
    std::fprintf(stderr, "usage: phys_ana3 fit|eval <dir> <qry> <pool_stem> <ppm> <params> [out.csv] ... [fuse_w.txt] [fpnet_scores.txt] [wf]\n");
    return 1;
  }
  std::string mode = argv[1], dir = argv[2], qs = argv[3], poolStem = argv[4];
  float ppm = float(std::atof(argv[5]));
  std::string paramPath = argv[6];
  std::string outPath = argc > 7 ? argv[7] : "";
  // eval only: a forward-MLP model to fuse with, z(phys) + w z(mlp cosine)
  std::string mlpPath = argc > 8 ? argv[8] : "";
  float fuseW = argc > 9 ? float(std::atof(argv[9])) : 0.0f;
  bool writeSmiles = argc > 10 && std::string(argv[10]) == "smiles";  // submission format
  bool haveMlp = !mlpPath.empty();
  // eval only: a reverse (spectrum -> ECFP) model, fused as + wr z(rev)
  std::string revPath = argc > 11 ? argv[11] : "";
  float fuseR = argc > 12 ? float(std::atof(argv[12])) : 0.0f;
  bool haveRev = !revPath.empty();
  // eval only: a spectral library (stem); a candidate with reference spectra
  // there gets its best similarity to the query as a continuous score,
  // + wl * libSim (instead of a hard "library top-1 first" rule)
  std::string libStem = argc > 13 ? argv[13] : "";
  float fuseL = argc > 14 ? float(std::atof(argv[14])) : 0.0f;
  float fuseLPow = argc > 15 ? float(std::atof(argv[15])) : 8.0f;
  float libPow = fuseLPow;
  float fuseA = argc > 16 ? float(std::atof(argv[16])) : 0.0f;  // analog-propagation weight
  float anaW = fuseA;
  // [anaVariant] [anaPow] [K]: which analog feature the output uses
  // (default = phys_rank's binned sim^3), and the stage-1 shortlist size
  int anaVar = argc > 18 ? std::atoi(argv[18]) : 0;
  int anaPowI = argc > 19 ? std::atoi(argv[19]) - 2 : 1;
  int shortK = argc > 20 ? std::atoi(argv[20]) : 2000;
  // [weights.txt]: 8 learned fusion weights (fuse_fit) replacing w/wr/wl/wa
  std::string weightsPath = argc > 21 ? argv[21] : "";
  // phys_fp: [fpnet_scores.txt] [wf=1]: "<molecule_id>\t<pool name>\t<score>" lines (fpnet_pool),
  // added to the final fused score as + wf * z(fpnet) (per-query z over its candidates;
  // a candidate without a score gets the query's mean, i.e. z = 0)
  std::string fpPath = argc > 22 ? argv[22] : "";
  float fuseF = argc > 23 ? float(std::atof(argv[23])) : 1.0f;
  // generation options (see header)
  std::map<std::string, std::string> opt;
  if (argc > 24) {
    std::stringstream ss(argv[24]);
    std::string kv;
    while (std::getline(ss, kv, ',')) {
      size_t e = kv.find('=');
      if (e != std::string::npos) opt[kv.substr(0, e)] = kv.substr(e + 1);
    }
  }
  auto optI = [&](const char* k, int d) { auto i = opt.find(k); return i == opt.end() ? d : std::atoi(i->second.c_str()); };
  const bool doGen = optI("gen", 0) != 0, rmTruth = optI("rm", 0) != 0, twoStep = optI("two", 1) != 0;
  const bool doGrid = optI("grid", 1) != 0;
  const bool libCap = optI("libcap", 0) != 0;  // v27: cap the library similarity at 1
  const int genCap = optI("cap", 1000), genNan = optI("nan", 80), genVar = optI("gvar", 4);
  const unsigned nThr = unsigned(optI("thr", int(std::max(1u, std::thread::hardware_concurrency()))));
  std::vector<double> gw(NG, 0.0);
  if (opt.count("gw")) {
    std::stringstream ss(opt["gw"]);
    std::string v;
    for (int k = 0; k < NG && std::getline(ss, v, ':'); ++k) gw[k] = std::atof(v.c_str());
  }
  std::unordered_map<std::string, float> opPrior;
  float opPriorDefault = 0.f;
  if (opt.count("prior")) {
    std::ifstream pf(dir + "/" + opt["prior"]);
    std::string k;
    float v;
    while (pf >> k >> v) {
      if (k == "default") opPriorDefault = v;
      else opPrior[k] = v;
    }
    std::printf("op priors: %zu (default %g)\n", opPrior.size(), opPriorDefault);
  }
  // a chain's prior: sum of its ops' single-op priors
  auto chainPrior = [&](const std::string& ch) {
    float s = 0.f;
    std::stringstream ss(ch);
    std::string o;
    while (std::getline(ss, o, '/')) {
      auto f = opPrior.find(o);
      s += f == opPrior.end() ? opPriorDefault : f->second;
    }
    return s;
  };
  auto optF = [&](const char* k, double d) { auto i = opt.find(k); return i == opt.end() ? d : std::atof(i->second.c_str()); };
  const bool doPc = optI("pc", 0) != 0, pcRecall = optI("pcrecall", 0) != 0;
  const int pcPre = optI("pcpre", 300), pcN = optI("pcn", 50), pcK = optI("pck", 2);
  const float pcPP = float(optF("pcpp", 20.0)), pcPop = float(optF("pcpop", 2.0));
  const float pcGate = float(optF("gate", 0.9)), pcGMarg = float(optF("gmarg", 1e9));
  // pcmode=score: instead of fixed slots, a pick enters where its score
  // (fused without popularity + pcsw * pop + pcbias) beats the main list's
  // next score, never above position pcmin
  const bool pcScoreMode = !(opt.count("pcmode") && opt["pcmode"] == "slot");
  const float pcSW = float(optF("pcsw", 1.0)), pcBias = float(optF("pcbias", -7.0));
  const int pcMin = optI("pcmin", 2);
  auto parseSlots = [&](const char* k, const char* d) {
    std::vector<int> r;
    std::stringstream ss(opt.count(k) ? opt[k] : std::string(d));
    std::string v;
    while (std::getline(ss, v, ':')) if (!v.empty()) r.push_back(std::atoi(v.c_str()));
    return r;
  };
  std::vector<int> pcSlots = parseSlots("slots", "2:4:6:8:10");
  // v28pc: PubChem channel v2 (pcv=2) - list ranked by z_window(REV2) + pcw * pop,
  // fixed slots, promotion of the top pick on confidence (see header)
  const int pcV = optI("pcv", 1);
  const float pcW = float(optF("pcw", 0.15));           // popularity weight on the window-z scale
  const bool pcRankFused = opt.count("pcrank") && opt["pcrank"] == "fused";
  std::vector<int> pcSlotsLo = parseSlots("slotslo", "4:8:12:16:20"), pcSlotsHi = parseSlots("slotshi", "2:4:6:8:10");
  const float pcRelHi = float(optF("relhi", 1e9));      // relz above: the dense slots
  const float pcRelP = float(optF("relp", 1e9));        // relz above: promote the top pick to rank 1
  const float pcSP = float(optF("sp", 1e9));            // ... or S above sp and the pick's pop >= popp
  const float pcPopP = float(optF("popp", 5.0));
  const int pcNoProm = optI("noprom", 0);
  // v33pc: Boltzmann merge of the two populations instead of the fixed slots (alloc=boltz)
  const bool pcBoltz = opt.count("alloc") && opt["alloc"] == "boltz";
  const double bT1 = optF("bT1", 2.0), bT2 = optF("bT2", 1.0), bPi = optF("bpi", 0.2), bA = optF("ba", 0.0);
  const int bK1 = optI("bK1", 60);
  // v30pc: the REV2 score form for PubChem window candidates (pcrev) and, separately,
  // for pool candidates (poolrev): 0 logit sum (v28pc), 1 continuous Tanimoto, 2 calibrated
  auto revModeOf = [&](const char* k) {
    auto i = opt.find(k);
    if (i == opt.end() || i->second == "logit") return 0;
    if (i->second == "tan" || i->second == "tanimoto") return 1;
    if (i->second == "cal") return 2;
    std::fprintf(stderr, "unknown %s=%s (logit|tan|cal), using logit\n", k, i->second.c_str());
    return 0;
  };
  const int pcRevMode = revModeOf("pcrev"), poolRevMode = revModeOf("poolrev");
  const char* revModeName[] = {"logit", "tanimoto", "calibrated"};
  // v32pc: own fp3 reverse model for the PubChem window candidates
  const std::string pcFpPath = opt.count("pcfp") ? opt["pcfp"] : std::string();
  const bool pcFp = !pcFpPath.empty() && doPc;
  const int pcFpTop = optI("pcfptop", 0);
  // v34pc: attention retrieval engine for the PubChem window (needs pcfp and a library)
  const bool attReq = optI("attn", 0) != 0 && pcFp;  // attOn (below, after haveLib) = attReq && a library
  const int attK = std::max(1, optI("attk", 10)), attPkN = optI("attpk", 40), attMp = optI("attmp", 4),
            attSim = std::min(3, std::max(0, optI("attsim", 3)));
  const float attT = float(optF("attT", 0.1)), attW = float(optF("attw", 1.0)), attTol = float(optF("atttol", 0.01));
  // attpool=w: also add w * z(att) (z over the query's pool + generated candidates) to the main fusion (default 0)
  const float attPoolW = float(optF("attpool", 0.0));
  // v36pc: cnnmodel=file (CNN1, cpp/cnn) -> continuous Tanimoto of the mean CNN p over the candidate's bits in the
  // model's own vocabulary; cnnpool=w adds w * z(cnn) over the query's candidates to the main fusion (default 0);
  // cnnshuf=k scores query q with the CNN p of query q+k (control); cfdump=file dumps every candidate
  const std::string cnnPath = opt.count("cnnmodel") ? opt["cnnmodel"] : std::string();
  const float cnnPoolW = float(optF("cnnpool", 0.0));
  const int cnnShuf = optI("cnnshuf", 0);
  const int cnnInst = optI("cnninst", -1);  // v36: >= 0 -> CNN term only for queries with a spectrum of this instrument (0 timsTOF)
  // v37pc: ftmodel=file (TFM1, fp3 vocabulary in the file) -> continuous Tanimoto of p = sigmoid(mean logit) over the
  // candidate's fp3 bits; ftpool=w adds w * z(ft) to the main fusion (default 0); cnninst gates it like the CNN;
  // ftshuf=k query q scored with query q+k's p (control); ftdump=file per-query mean logits (fpnet _mol.bin layout,
  // id = query index, names in file.txt); ftbudget=s: if the inference takes longer than s seconds it is abandoned
  // and the run continues without the term (0 = no limit)
  const std::string ftPath = opt.count("ftmodel") ? opt["ftmodel"] : std::string();
  const float ftPoolW = float(optF("ftpool", 0.0));
  const int ftShuf = optI("ftshuf", 0);
  const double ftBudget = optF("ftbudget", 0.0);
  // v28pc: pool popularity (ppop=w): among the top ppn of the final list, the pool
  // candidates' places are re-filled in the order z(final score) + w * pop
  // (z over those pool candidates); generated candidates keep their places
  const float ppW = float(optF("ppop", 0.0));
  const int ppN = optI("ppn", 60);
  // ppgate=x: skip the re-order when the best library similarity of the main candidates
  // is >= x (a library identity match: popularity should not override it)
  const float ppGate = float(optF("ppgate", 1e9));
  std::vector<float> poolPop;
  if (ppW != 0.f || opt.count("ppopf")) {
    std::string pf = opt.count("ppopf") ? opt["ppopf"] : std::string("pool730_pop.bin");
    std::string pth = (pf.size() > 1 && (pf[1] == ':' || pf[0] == '/')) ? pf : dir + "/" + pf;
    std::ifstream f(pth, std::ios::binary | std::ios::ate);
    if (f) {
      std::streamsize n = f.tellg();
      f.seekg(0);
      poolPop.resize(size_t(n) / 4);
      f.read(reinterpret_cast<char*>(poolPop.data()), std::streamsize(poolPop.size() * 4));
    }
    std::printf("pool popularity: %s, %zu values, weight %g over the top %d (pool candidates only), only when best libSim < %g%s\n",
                pth.c_str(), poolPop.size(), ppW, ppN, ppGate, poolPop.empty() ? " - FILE MISSING, POOL POP DISABLED" : "");
  }
  const std::string pcDir = opt.count("pcdir") ? opt["pcdir"] : dir;
  std::vector<float> pcMass, pcPopV;
  std::vector<uint64_t> pcOff;
  if (doPc) {
    auto rd = [&](const std::string& p, auto& v) {
      std::vector<char> b = slurp(pcDir + "/" + p);
      v.resize(b.size() / sizeof(v[0]));
      std::memcpy(v.data(), b.data(), v.size() * sizeof(v[0]));
    };
    rd("pc_mass.bin", pcMass);
    rd("pc_off.bin", pcOff);
    rd("pc_pop.bin", pcPopV);
    if (pcV == 2) {
      std::printf("pubchem v2: %zu compounds; list = z_window(rev2) + %g pop (%s), slots lo", pcMass.size(), pcW,
                  pcRankFused ? "top by it, then ordered by z(hand fusion) + pop" : "rev2 only");
      for (int s : pcSlotsLo) std::printf(":%d", s);
      std::printf(" hi");
      for (int s : pcSlotsHi) std::printf(":%d", s);
      std::printf(" (relz > %g); promote if relz > %g or (S > %g and pop >= %g)%s; gate libSim < %g\n", pcRelHi, pcRelP,
                  pcSP, pcPopP, pcNoProm ? " DISABLED" : "", pcGate);
      if (pcBoltz)
        std::printf("pubchem v33: alloc=boltz (replaces slots/promotion): T1 %g T2 %g pi %g ba %g K1 %d\n", bT1, bT2, bPi, bA, bK1);
    } else
      std::printf("pubchem: %zu compounds; prefilter %d (pop %g), full %d, insert %d", pcMass.size(), pcPre, pcPP, pcN, pcK);
    if (pcV == 2) {
    } else if (pcScoreMode) {
      std::printf(" by score (pop %g, bias %g, from position %d)", pcSW, pcBias, pcMin);
    } else {
      std::printf(" at slots");
      for (int s : pcSlots) std::printf(" %d", s);
    }
    if (pcV != 2) std::printf(", gate libSim < %g, margin < %g, list pop %g\n", pcGate, pcGMarg, pcPop);
  }
  if (doGen) std::printf("generation: cap %d, analogs %d (variant %d), two-step %d, remove truth %d, threads %u\n",
                         genCap, genNan, genVar, int(twoStep), int(rmTruth), nThr);
  if (anaVar < 0 || anaVar >= NVAR) anaVar = 0;
  if (anaPowI < 0 || anaPowI > 2) anaPowI = 1;
  bool haveLib = !libStem.empty();
  const bool attOn = attReq && haveLib;
  if (optI("attn", 0) != 0 && !attOn) std::printf("attn ignored (needs pc=1, pcfp and a library)\n");
  forward_model_nn::MLP mlp;
  if (haveMlp) mlp = forward_model_nn::MLP::load(dir + "/" + mlpPath);

  // spectra
  std::vector<Head> head;
  std::vector<float> mz, it;
  {
    std::vector<char> hb = slurp(dir + "/" + qs + "_head.bin");
    head.resize(hb.size() / sizeof(Head));
    std::memcpy(head.data(), hb.data(), hb.size());
    std::vector<char> pb = slurp(dir + "/" + qs + "_peaks.bin");
    const float* p = reinterpret_cast<const float*>(pb.data());
    for (Head& h : head) {
      h.off = int32_t(mz.size());
      mz.insert(mz.end(), p, p + h.n); p += h.n;
      it.insert(it.end(), p, p + h.n); p += h.n;
    }
  }
  std::vector<std::string> qnames, qsmiles;
  {
    std::ifstream f(dir + "/" + qs + "_mols.txt");
    std::string line;
    while (std::getline(f, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      size_t t = line.find('\t');
      qnames.push_back(line.substr(0, t));
      qsmiles.push_back(t == std::string::npos ? "" : line.substr(t + 1));
    }
  }
  // pool
  std::vector<float> pmass;
  std::vector<int32_t> pid;
  std::vector<std::string> pname, psmi;
  std::vector<uint8_t> pHasSpec;  // by pool index: a train structure (has reference spectra)
  {
    std::vector<char> hb = slurp(dir + "/" + poolStem + "_head.bin");
    size_t n = hb.size() / 12;
    pmass.resize(n); pid.resize(n); pHasSpec.assign(n, 0);
    for (size_t i = 0; i < n; ++i) {
      int32_t hs = 0;
      std::memcpy(&pmass[i], hb.data() + 12 * i, 4);
      std::memcpy(&pid[i], hb.data() + 12 * i + 4, 4);
      std::memcpy(&hs, hb.data() + 12 * i + 8, 4);
      if (pid[i] >= 0 && size_t(pid[i]) < n) pHasSpec[pid[i]] = uint8_t(hs != 0);
    }
    std::ifstream f(dir + "/" + poolStem + "_mols.txt");
    std::string line;
    while (std::getline(f, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      size_t t = line.find('\t');
      pname.push_back(line.substr(0, t));
      psmi.push_back(t == std::string::npos ? "" : line.substr(t + 1));
    }
  }
  std::unordered_map<std::string, int> poolIdx;
  for (size_t i = 0; i < pname.size(); ++i) poolIdx.emplace(pname[i], int(i));

  std::unordered_map<int, std::vector<int>> specOf;
  // fit only: [instrument] keeps just that instrument's spectra (0 timsTOF,
  // 1 Orbitrap, ...), to compare per-instrument fits
  int instrFilter = (mode == "fit" && argc > 8) ? std::atoi(argv[8]) : -1;
  for (size_t s = 0; s < head.size(); ++s) {
    if (instrFilter >= 0) {
      int32_t t = head[s].tag;
      int ins = (t < 0 || t % 16 == 15) ? -1 : t % 16;
      if (ins != instrFilter) continue;
    }
    specOf[head[s].mol].push_back(int(s));
  }
  if (instrFilter >= 0) std::printf("instrument filter %d: %zu molecules\n", instrFilter, specOf.size());
  std::vector<Query> Q;
  for (auto& kv : specOf) {
    Query q;
    q.name = qnames[kv.first];
    q.tsmi = qsmiles[kv.first];
    q.spectra = kv.second;
    Q.push_back(std::move(q));
  }
  std::sort(Q.begin(), Q.end(), [](auto& a, auto& b) { return a.name < b.name; });

  // Spectral library: spectra grouped by molecule, keyed by InChIKey14 name.
  constexpr int NBIN = 300000;  // 0.01 Da bins up to 3000 m/z
  std::vector<int64_t> binStart;
  std::vector<int32_t> postSpec;
  std::vector<float> postW;
  std::vector<int64_t> nlStart;
  std::vector<int32_t> nlSpec;
  std::vector<float> nlW, lnrm;
  std::vector<std::string> lnames, lsmiles;
  std::vector<Head> lhead;
  std::vector<float> lmz, lit;
  std::unordered_map<std::string, std::vector<int>> libSpecOf;
  if (haveLib) {
    std::vector<char> hb = slurp(dir + "/" + libStem + "_head.bin");
    lhead.resize(hb.size() / sizeof(Head));
    std::memcpy(lhead.data(), hb.data(), hb.size());
    std::vector<char> pb = slurp(dir + "/" + libStem + "_peaks.bin");
    const float* p = reinterpret_cast<const float*>(pb.data());
    size_t total = 0;
    for (const Head& h : lhead) total += size_t(h.n);
    lmz.reserve(total); lit.reserve(total);
    for (Head& h : lhead) {
      h.off = int32_t(lmz.size());
      lmz.insert(lmz.end(), p, p + h.n); p += h.n;
      lit.insert(lit.end(), p, p + h.n); p += h.n;
    }
    std::ifstream f(dir + "/" + libStem + "_mols.txt");
    std::string line;
    while (std::getline(f, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      size_t t = line.find('\t');
      lnames.push_back(line.substr(0, t));
      lsmiles.push_back(t == std::string::npos ? "" : line.substr(t + 1));
    }
    for (size_t s = 0; s < lhead.size(); ++s) libSpecOf[lnames[lhead[s].mol]].push_back(int(s));
    std::printf("library %s: %zu spectra, %zu molecules\n", libStem.c_str(), lhead.size(), libSpecOf.size());

    // Inverted index over 0.01 Da bins for the analog search: posting =
    // (spectrum, sqrt(I)/||sqrt(I)||), so summing query x library weights
    // over shared bins approximates the sqrt-intensity cosine of every
    // library spectrum at once.
    binStart.assign(NBIN + 1, 0);
    for (const Head& h : lhead)
      for (int j = 0; j < h.n; ++j) {
        int b = int(lmz[h.off + j] * 100.f);
        if (b >= 0 && b < NBIN) ++binStart[b + 1];
      }
    for (int b = 0; b < NBIN; ++b) binStart[b + 1] += binStart[b];
    postSpec.resize(binStart[NBIN]); postW.resize(binStart[NBIN]);
    std::vector<int64_t> fill(binStart.begin(), binStart.end() - 1);
    for (size_t s = 0; s < lhead.size(); ++s) {
      const Head& h = lhead[s];
      double nrm = 0;
      for (int j = 0; j < h.n; ++j) nrm += std::max(0.f, lit[h.off + j]);
      nrm = std::sqrt(nrm) + 1e-12;
      for (int j = 0; j < h.n; ++j) {
        int b = int(lmz[h.off + j] * 100.f);
        if (b < 0 || b >= NBIN) continue;
        postSpec[fill[b]] = int32_t(s);
        postW[fill[b]++] = float(std::sqrt(std::max(0.f, lit[h.off + j])) / nrm);
      }
    }
    std::printf("analog index: %zu postings\n", postSpec.size());
    // Neutral-loss index: bin of precursor - m/z (fragments below the
    // precursor only, including the precursor peak itself at loss ~0)
    lnrm.resize(lhead.size());
    nlStart.assign(NBIN + 1, 0);
    for (const Head& h : lhead)
      for (int j = 0; j < h.n; ++j) {
        int b = int((h.precursor - lmz[h.off + j]) * 100.f + 0.5f);  // 0.5 so the precursor peak lands in bin 0
        if (b >= 0 && b < NBIN) ++nlStart[b + 1];
      }
    for (int b = 0; b < NBIN; ++b) nlStart[b + 1] += nlStart[b];
    nlSpec.resize(nlStart[NBIN]); nlW.resize(nlStart[NBIN]);
    std::vector<int64_t> fill2(nlStart.begin(), nlStart.end() - 1);
    for (size_t s = 0; s < lhead.size(); ++s) {
      const Head& h = lhead[s];
      double nrm = 0;
      for (int j = 0; j < h.n; ++j) nrm += std::max(0.f, lit[h.off + j]);
      nrm = std::sqrt(nrm) + 1e-12;
      lnrm[s] = float(nrm);
      for (int j = 0; j < h.n; ++j) {
        int b = int((h.precursor - lmz[h.off + j]) * 100.f + 0.5f);
        if (b < 0 || b >= NBIN) continue;
        nlSpec[fill2[b]] = int32_t(s);
        nlW[fill2[b]++] = float(std::sqrt(std::max(0.f, lit[h.off + j])) / nrm);
      }
    }
    std::printf("neutral-loss index: %zu postings\n", nlSpec.size());
  }
  // v34pc: attention retrieval index (fragment m/z and neutral loss, attPrep peaks)
  AttIndex attFx, attLx;
  std::vector<int8_t> attPol;
  if (attOn) {
    auto ta0 = std::chrono::steady_clock::now();
    const size_t nLS = lhead.size();
    std::vector<std::vector<AttPk>> lp(nLS);
    attPol.assign(nLS, 0);
    {
      std::vector<std::thread> tt;
      for (unsigned t = 0; t < nThr; ++t)
        tt.emplace_back([&, t]() {
          for (size_t s = t; s < nLS; s += nThr) {
            const Head& h = lhead[s];
            if (h.mol >= 0) attPrep(h.precursor, lmz.data() + h.off, lit.data() + h.off, h.n, attPkN, lp[s]);
            int a = adductIdx(h.tag);
            attPol[s] = int8_t(a < 0 || a >= 10 ? 0 : (ADDUCT_NEG[a] ? -1 : 1));
          }
        });
      for (auto& x : tt) x.join();
    }
    auto build = [&](AttIndex& ix, bool loss) {
      ix.bw = attTol; ix.nb = int(2500.f / attTol) + 2;
      ix.off.assign(size_t(ix.nb) + 1, 0);
      auto key = [&](size_t s, const AttPk& k) {
        return loss ? (lhead[s].precursor > 0.f ? lhead[s].precursor - k.mz : -1.f) : k.mz;
      };
      for (size_t s = 0; s < nLS; ++s)
        for (const AttPk& k : lp[s]) { float v = key(s, k); if (v > 0.f && v < 2500.f) ++ix.off[size_t(v / ix.bw) + 1]; }
      for (int b = 0; b < ix.nb; ++b) ix.off[b + 1] += ix.off[b];
      ix.p.resize(ix.off[ix.nb]);
      std::vector<uint64_t> cur(ix.off.begin(), ix.off.end() - 1);
      for (size_t s = 0; s < nLS; ++s)
        for (const AttPk& k : lp[s]) {
          float v = key(s, k);
          if (v > 0.f && v < 2500.f) ix.p[cur[size_t(v / ix.bw)]++] = {uint32_t(s), v, k.s, k.e};
        }
    };
    build(attFx, false);
    build(attLx, true);
    double ta = std::chrono::duration<double>(std::chrono::steady_clock::now() - ta0).count();
    std::printf("attn: index %zu fragment + %zu loss postings (%.1fs); sim %d, k %d, T %g, w %g, peaks %d, tol %g, matched-peak scale %d\n",
                attFx.p.size(), attLx.p.size(), ta, attSim, attK, attT, attW, attPkN, attTol, attMp);
  }

  // Reverse-model bit logits per query (single-threaded: the model owns its buffers).
  int revNB = 0;
  // REV2 model files (magic "REV2") carry their own input layout and an fp2
  // vocabulary defining the output bits; candidate bits then come from revVocab.
  bool rev2Mode = false;
  fp2::Vocab revVocab;
  if (haveRev) {
    int32_t magic = 0;
    {
      std::ifstream mf(dir + "/" + revPath, std::ios::binary);
      mf.read(reinterpret_cast<char*>(&magic), 4);
    }
    rev2Mode = magic == 0x32564552;
  }
  if (haveRev && rev2Mode) {
    rev2::Model rm;
    if (!rm.load(dir + "/" + revPath)) { std::fprintf(stderr, "cannot load %s\n", revPath.c_str()); return 1; }
    if (rm.vocab.empty() || int(rm.vocab.size()) != rm.NB) {
      std::fprintf(stderr, "%s: REV2 model without embedded vocabulary (nVocab %zu, NB %d)\n", revPath.c_str(), rm.vocab.size(), rm.NB);
      return 1;
    }
    revNB = rm.NB;
    revVocab.rE = rm.rE; revVocab.rF = rm.rF; revVocab.hashes = rm.vocab; revVocab.build();
    // logit of the mean p, clipped to [0.01, 0.99] (REV2_NOCLIP=1: [1e-6, 1-1e-6] as v1)
    const char* nc = std::getenv("REV2_NOCLIP");
    const float lo = (nc && *nc == '1') ? 1e-6f : 0.01f;
    std::vector<float> Pb;
    for (Query& q : Q) {
      std::vector<std::vector<std::pair<int, float>>> rows(q.spectra.size());
      for (size_t i = 0; i < q.spectra.size(); ++i) {
        const Head& h = head[q.spectra[i]];
        rev2::Head rh{h.precursor, h.mol, 0, h.n, h.tag};
        rev2::featurize(rm.fc, rh, mz.data() + h.off, it.data() + h.off, rows[i]);
      }
      int B = int(rows.size());
      rm.reserve(B);
      rm.setBatch(rows, false);
      rm.forward(B, false, 0.f, 0);
      rev2::download(rm.P, Pb, int64_t(B) * revNB);
      q.logit.assign(revNB, 0.f);
      for (int b = 0; b < revNB; ++b) {
        float p = 0.f;
        for (int i = 0; i < B; ++i) p += Pb[size_t(i) * revNB + b];
        p = std::min(1.f - lo, std::max(lo, p / float(B)));
        q.logit[b] = std::log(p / (1.f - p));
      }
      if (pcRevMode || poolRevMode) {
        // v30pc: as cpp/rev3/win_eval.cpp - mean p over the query's spectra (unclipped) for the
        // continuous Tanimoto; calibrated z = calA * logit(clip 1e-6) + calC, clipped to +-13.8
        const bool hasCal = int(rm.calA.size()) == revNB;
        q.revP.assign(revNB, 0.f);
        q.revCalU.assign(revNB, 0.f);
        q.revSumP = 0.0;
        for (int b = 0; b < revNB; ++b) {
          float p = 0.f;
          for (int i = 0; i < B; ++i) p += Pb[size_t(i) * revNB + b];
          p /= float(B);
          q.revP[b] = p;
          q.revSumP += p;
          float pi = std::min(1.f - 1e-6f, std::max(1e-6f, p));
          float lg = std::log(pi / (1.f - pi));
          float z = hasCal ? rm.calA[b] * lg + rm.calC[b] : lg;
          q.revCalU[b] = std::min(13.8f, std::max(-13.8f, z));
        }
      }
    }
    std::printf("reverse model %s: REV2, %d vocab bits (ECFP r%d + FCFP r%d), clip %g\n", revPath.c_str(), revNB, rm.rE, rm.rF, lo);
    if (pcRevMode || poolRevMode)
      std::printf("reverse model score form: PubChem window %s, pool %s (Platt calibration in the model: %d)\n",
                  revModeName[pcRevMode], revModeName[poolRevMode], int(int(rm.calA.size()) == revNB));
  } else if (haveRev) {
    rev::Model rm;
    if (!rm.load(dir + "/" + revPath)) { std::fprintf(stderr, "cannot load %s\n", revPath.c_str()); return 1; }
    revNB = rm.NB;
    std::vector<float> Pb;
    for (Query& q : Q) {
      std::vector<std::vector<std::pair<int, float>>> rows(q.spectra.size());
      for (size_t i = 0; i < q.spectra.size(); ++i) {
        const Head& h = head[q.spectra[i]];
        rev::Head rh{h.precursor, h.mol, 0, h.n, h.tag};
        rev::featurize(rh, mz.data() + h.off, it.data() + h.off, rows[i]);
      }
      int B = int(rows.size());
      rm.reserve(B);
      rm.setBatch(rows, false);
      rm.forward(B, false, 0.f, 0);
      rev::download(rm.P, Pb, int64_t(B) * revNB);
      q.logit.assign(revNB, 0.f);
      for (int b = 0; b < revNB; ++b) {
        float p = 0.f;
        for (int i = 0; i < B; ++i) p += Pb[size_t(i) * revNB + b];
        p = std::min(1.f - 1e-6f, std::max(1e-6f, p / float(B)));
        q.logit[b] = std::log(p / (1.f - p));
      }
      if (pcRevMode || poolRevMode) {  // v1 reverse model: no calibration (cal = logit)
        q.revP.assign(revNB, 0.f);
        q.revSumP = 0.0;
        for (int b = 0; b < revNB; ++b) {
          float p = 0.f;
          for (int i = 0; i < B; ++i) p += Pb[size_t(i) * revNB + b];
          q.revP[b] = p / float(B);
          q.revSumP += q.revP[b];
        }
        q.revCalU = q.logit;
      }
    }
    std::printf("reverse model %s: %d bits\n", revPath.c_str(), revNB);
  }

  // ---- v36pc: CNN reverse model (CPU inference, mean p over the query's spectra)
  fp2::Vocab cnnVocab;
  const bool cnnOn = !cnnPath.empty();
  if (cnnOn) {
    auto tc0 = std::chrono::steady_clock::now();
    std::string pth = (cnnPath.size() > 1 && (cnnPath[1] == ':' || cnnPath[0] == '/')) ? cnnPath : dir + "/" + cnnPath;
    cnn::Model cm;
    if (!cnn::Model::isCnnFile(pth) || !cm.load(pth)) { std::fprintf(stderr, "cannot load cnnmodel %s\n", pth.c_str()); return 1; }
    if (cm.vocab.empty() || int(cm.vocab.size()) != cm.NB) { std::fprintf(stderr, "cnnmodel without vocabulary\n"); return 1; }
    cnnVocab.rE = cm.rE; cnnVocab.rF = cm.rF; cnnVocab.hashes = cm.vocab; cnnVocab.build();
    std::vector<cnn::Sample> ss;
    std::vector<cnn::real> P;
    size_t nSp = 0;
    for (Query& q : Q) {
      const int B = int(q.spectra.size());
      if (B == 0) continue;
      if (cnnInst >= 0 && std::none_of(q.spectra.begin(), q.spectra.end(), [&](int s) { return head[s].tag >= 0 && head[s].tag % 16 == cnnInst; })) continue;  // cnninst: no p -> no CNN term
      ss.assign(B, cnn::Sample());
      for (int i = 0; i < B; ++i) {
        const Head& h = head[q.spectra[i]];
        rev2::Head rh{h.precursor, h.mol, 0, h.n, h.tag};
        cnn::featurize(cm.cfg, rh, mz.data() + h.off, it.data() + h.off, ss[i]);
      }
      cm.setBatch(ss, false);
      cm.forward(B, false, 0.f, 0);
      rev2::download(cm.P, P, int64_t(B) * cm.NB);
      q.cnnP.assign(cm.NB, 0.f);
      q.cnnSumP = 0.0;
      for (int b = 0; b < cm.NB; ++b) {
        float p = 0.f;
        for (int i = 0; i < B; ++i) p += float(P[size_t(i) * cm.NB + b]);
        q.cnnP[b] = p / float(B);
        q.cnnSumP += q.cnnP[b];
      }
      nSp += size_t(B);
    }
    if (cnnShuf != 0 && !Q.empty()) {
      std::vector<std::vector<float>> P2(Q.size());
      std::vector<double> S2(Q.size());
      for (size_t qi = 0; qi < Q.size(); ++qi) { size_t s = (qi + size_t(cnnShuf)) % Q.size(); P2[qi] = Q[s].cnnP; S2[qi] = Q[s].cnnSumP; }
      for (size_t qi = 0; qi < Q.size(); ++qi) { Q[qi].cnnP.swap(P2[qi]); Q[qi].cnnSumP = S2[qi]; }
    }
    std::printf("cnn model %s: %d bits (vocab r%d/%d), %zu spectra in %.1fs, cnnpool %g, cnnshuf %d\n", pth.c_str(), cm.NB,
                cm.rE, cm.rF, nSp, std::chrono::duration<double>(std::chrono::steady_clock::now() - tc0).count(), cnnPoolW, cnnShuf);
    if (cnnInst >= 0) std::printf("cnninst %d: CNN term on %zu of %zu queries\n", cnnInst, size_t(std::count_if(Q.begin(), Q.end(), [](const Query& q) { return !q.cnnP.empty(); })), Q.size());
  }

  // ---- v37pc: fine-tuned pretrained transformer (TFM1, fp3c16k head), CPU; per query p = sigmoid(mean logit)
  // (= pt_train mollogits + pool37ft.py: spectra with peaks, peaks/tag prepared as tfm_train)
  std::unordered_map<uint64_t, int> ftIndex;
  const bool ftOn = !ftPath.empty();
  if (ftOn) {
    parallel_cap() = nThr;
    auto tf0 = std::chrono::steady_clock::now();
    std::string pth = (ftPath.size() > 1 && (ftPath[1] == ':' || ftPath[0] == '/')) ? ftPath : dir + "/" + ftPath;
    tfm::Model fm;
    if (!fm.load(pth, false)) { std::fprintf(stderr, "cannot load ftmodel %s\n", pth.c_str()); return 1; }
    if (fm.vocab.empty() || int(fm.vocab.size()) != fm.c.NB || fm.rE < 300) {
      std::fprintf(stderr, "ftmodel %s: not an fp3-vocabulary model (nVocab %zu, NB %d, rE %d)\n", pth.c_str(), fm.vocab.size(), fm.c.NB, fm.rE);
      return 1;
    }
    if (fm.rF != fp3::FP3_VERSION) std::fprintf(stderr, "ftmodel: fp3 version %d in the model, %d here\n", fm.rF, fp3::FP3_VERSION);
    for (size_t i = 0; i < fm.vocab.size(); ++i) ftIndex.emplace(fm.vocab[i], int(i));
    const int NB = fm.c.NB;
    std::vector<std::pair<int, int>> sp;  // (query, spectrum) for the queries that get the term
    for (size_t qi = 0; qi < Q.size(); ++qi) {
      const Query& q = Q[qi];
      if (cnnInst >= 0 && std::none_of(q.spectra.begin(), q.spectra.end(), [&](int s) { return head[s].tag >= 0 && head[s].tag % 16 == cnnInst; })) continue;  // cnninst gates FT too
      for (int s : q.spectra) if (head[s].n > 0) sp.emplace_back(int(qi), s);
    }
    std::vector<std::vector<float>> M(sp.size()), I(sp.size());
    for (size_t i = 0; i < sp.size(); ++i) {
      const Head& h = head[sp[i].second];
      tfmPrep(h, mz.data() + h.off, it.data() + h.off, fm.c.maxPeaks, M[i], I[i]);
    }
    std::vector<int> ord(sp.size());
    for (size_t i = 0; i < ord.size(); ++i) ord[i] = int(i);
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return M[a].size() < M[b].size(); });
    std::vector<std::vector<double>> zsum(Q.size());
    std::vector<int> zcnt(Q.size(), 0);
    tfm::HostBatch hb;
    std::vector<float> z;
    const size_t BS = 32;
    bool ftTimeout = false;
    size_t nDone = 0;
    for (size_t s0 = 0; s0 < ord.size(); s0 += BS) {
      if (ftBudget > 0 && std::chrono::duration<double>(std::chrono::steady_clock::now() - tf0).count() > ftBudget) { ftTimeout = true; break; }
      size_t s1 = std::min(ord.size(), s0 + BS);
      int B = int(s1 - s0), mx = 1;
      for (size_t i = s0; i < s1; ++i) mx = std::max(mx, int(M[ord[i]].size()));
      hb.B = B; hb.N = 1 + mx;
      hb.mz.assign(size_t(B) * mx, 0.f); hb.it.assign(size_t(B) * mx, 0.f);
      hb.prec.resize(B); hb.mode.resize(B); hb.np.resize(B); hb.ad.resize(B); hb.in.resize(B);
      for (int b = 0; b < B; ++b) {
        int i = ord[s0 + b];
        const Head& h = head[sp[i].second];
        std::copy(M[i].begin(), M[i].end(), hb.mz.begin() + size_t(b) * mx);
        std::copy(I[i].begin(), I[i].end(), hb.it.begin() + size_t(b) * mx);
        hb.np[b] = int(M[i].size());
        hb.prec[b] = h.precursor;
        tfmTag(h.tag, hb.ad[b], hb.in[b], hb.mode[b]);
      }
      fm.setBatch(hb);
      fm.forward(false, 0.f, 0);
      z.resize(size_t(B) * NB);
      tfm::download(fm.Z, z.data(), int64_t(z.size()));
      for (int b = 0; b < B; ++b) {
        const int qi = sp[ord[s0 + b]].first;
        auto& v = zsum[qi];
        if (v.empty()) v.assign(NB, 0.0);
        for (int j = 0; j < NB; ++j) v[j] += double(z[size_t(b) * NB + j]);
        ++zcnt[qi];
      }
      nDone = s1;
    }
    if (!ftTimeout) {
      std::ofstream fd, fdn;
      if (opt.count("ftdump")) { fd.open(dir + "/" + opt["ftdump"], std::ios::binary); fdn.open(dir + "/" + opt["ftdump"] + ".txt"); }
      int32_t nq = 0, nb32 = NB;
      for (int c : zcnt) nq += c > 0;
      if (fd.is_open()) { fd.write((const char*)&nq, 4); fd.write((const char*)&nb32, 4); }
      std::vector<float> row(NB);
      for (size_t qi = 0; qi < Q.size(); ++qi) {
        if (!zcnt[qi]) continue;
        Query& q = Q[qi];
        q.ftP.assign(NB, 0.f);
        q.ftSumP = 0.0;
        for (int j = 0; j < NB; ++j) {
          row[j] = float(zsum[qi][j] / zcnt[qi]);  // mean logit (as mollogits, float)
          double x = std::max(-30.0, std::min(30.0, double(row[j])));
          q.ftP[j] = float(1.0 / (1.0 + std::exp(-x)));
          q.ftSumP += double(q.ftP[j]);
        }
        if (fd.is_open()) { int32_t id = int32_t(qi); fd.write((const char*)&id, 4); fd.write((const char*)row.data(), 4 * size_t(NB)); fdn << qi << '\t' << q.name << '\n'; }
      }
      if (ftShuf != 0 && !Q.empty()) {
        std::vector<std::vector<float>> P2(Q.size());
        std::vector<double> S2(Q.size());
        for (size_t qi = 0; qi < Q.size(); ++qi) { size_t s = (qi + size_t(ftShuf)) % Q.size(); P2[qi] = Q[s].ftP; S2[qi] = Q[s].ftSumP; }
        for (size_t qi = 0; qi < Q.size(); ++qi) { Q[qi].ftP.swap(P2[qi]); Q[qi].ftSumP = S2[qi]; }
      }
    }
    double tf = std::chrono::duration<double>(std::chrono::steady_clock::now() - tf0).count();
    if (ftTimeout) {
      std::printf("FT TIMEOUT: ftmodel inference abandoned after %.1fs (%zu of %zu spectra, budget %.0fs) - no FT term\n", tf, nDone, sp.size(), ftBudget);
      ftIndex.clear();
    } else {
      std::printf("ft model %s: d%d L%d H%d HH%d, %d fp3 bits (fp3 v%d), %zu spectra in %.1fs (%.1f ms/spectrum, %u threads), ftpool %g, ftshuf %d\n",
                  pth.c_str(), fm.c.d, fm.c.L, fm.c.H, fm.c.HH, NB, fm.rF, sp.size(), tf, 1000.0 * tf / std::max<size_t>(1, sp.size()), nThr, ftPoolW, ftShuf);
      std::printf("ft term on %zu of %zu queries%s\n", size_t(std::count_if(Q.begin(), Q.end(), [](const Query& q) { return !q.ftP.empty(); })), Q.size(), cnnInst >= 0 ? " (cnninst gate)" : "");
    }
    std::fflush(stdout);
  }

  // ---- v32pc: fp3 reverse model (same file format as REV2, fp3 vocabulary) for the PubChem window
  std::unordered_map<uint64_t, int> fpIndex;
  if (pcFp && haveRev) {
    auto tf0 = std::chrono::steady_clock::now();
    std::string pth = (pcFpPath.size() > 1 && (pcFpPath[1] == ':' || pcFpPath[0] == '/')) ? pcFpPath : dir + "/" + pcFpPath;
    rev2::Model fm;
    if (!fm.load(pth)) { std::fprintf(stderr, "cannot load pcfp %s\n", pth.c_str()); return 1; }
    if (fm.vocab.empty() || int(fm.vocab.size()) != fm.NB || fm.rE < 300) {
      std::fprintf(stderr, "pcfp %s: not an fp3-vocabulary model (nVocab %zu, NB %d, rE %d)\n", pth.c_str(), fm.vocab.size(), fm.NB, fm.rE);
      return 1;
    }
    if (fm.rF != fp3::FP3_VERSION) std::fprintf(stderr, "pcfp: fp3 version %d in the model, %d here\n", fm.rF, fp3::FP3_VERSION);
    for (size_t i = 0; i < fm.vocab.size(); ++i) fpIndex.emplace(fm.vocab[i], int(i));
    const int fNB = fm.NB;
    std::vector<float> Pb;
    for (Query& q : Q) {
      std::vector<std::vector<std::pair<int, float>>> rows(q.spectra.size());
      for (size_t i = 0; i < q.spectra.size(); ++i) {
        const Head& h = head[q.spectra[i]];
        rev2::Head rh{h.precursor, h.mol, 0, h.n, h.tag};
        rev2::featurize(fm.fc, rh, mz.data() + h.off, it.data() + h.off, rows[i]);
      }
      int B = int(rows.size());
      q.fpP.assign(fNB, 0.f);
      q.fpSumP = 0.0;
      if (B == 0) continue;
      fm.reserve(B);
      fm.setBatch(rows, false);
      fm.forward(B, false, 0.f, 0);
      rev2::download(fm.P, Pb, int64_t(B) * fNB);
      for (int b = 0; b < fNB; ++b) {
        float p = 0.f;
        for (int i = 0; i < B; ++i) p += Pb[size_t(i) * fNB + b];
        q.fpP[b] = p / float(B);
        q.fpSumP += q.fpP[b];
      }
    }
    double tf = std::chrono::duration<double>(std::chrono::steady_clock::now() - tf0).count();
    std::printf("pcfp %s: fp3 reverse model, %d bits (fp3 v%d); load+forward %.1fs (%.1f ms/query); window top %d (0 = all)\n",
                pth.c_str(), fNB, fm.rF, tf, 1000.0 * tf / std::max<size_t>(1, Q.size()), pcFpTop);
  } else if (!pcFpPath.empty()) {
    std::printf("pcfp ignored (needs pc=1 and a reverse model)\n");
  }

  // ---- own Transformer (TFM1): molecule probabilities per query, CPU
  std::vector<std::string> tfmPaths;
  if (opt.count("tfm")) {
    std::stringstream ss(opt["tfm"]);
    std::string v;
    while (std::getline(ss, v, '+')) if (!v.empty()) tfmPaths.push_back(v);
  }
  const int tfmNM = int(tfmPaths.size());
  const int tfmNP = tfmNM + (tfmNM > 1 ? 1 : 0);  // probability vectors per query (models + ensemble)
  const int tfmNC = 2 * tfmNP;
  float tfmW = opt.count("tw") ? float(std::atof(opt["tw"].c_str())) : 0.f;
  int tfmV = optI("tv", tfmNM > 1 ? 2 * tfmNM : 0);
  const bool tfmShuf = optI("tshuf", 0) != 0;
  if (opt.count("tsign")) tfmW *= float(std::atof(opt["tsign"].c_str()));
  fp2::Vocab tfmVocab;
  if (tfmNM) {
    parallel_cap() = nThr;
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::pair<int, int>> sp;  // (query, spectrum)
    for (size_t qi = 0; qi < Q.size(); ++qi)
      for (int s : Q[qi].spectra) sp.emplace_back(int(qi), s);
    for (Query& q : Q) q.tP.assign(tfmNP, {});
    long long nTok = 0;
    for (int k = 0; k < tfmNM; ++k) {
      tfm::Model tm;
      if (!tm.load(dir + "/" + tfmPaths[k], false)) { std::fprintf(stderr, "cannot load %s\n", tfmPaths[k].c_str()); return 1; }
      if (tm.vocab.empty() || int(tm.vocab.size()) != tm.c.NB) {
        std::fprintf(stderr, "%s: TFM model without embedded vocabulary\n", tfmPaths[k].c_str());
        return 1;
      }
      if (k == 0) {
        tfmVocab.rE = tm.rE; tfmVocab.rF = tm.rF; tfmVocab.hashes = tm.vocab; tfmVocab.build();
      } else if (tm.vocab != tfmVocab.hashes || tm.rE != tfmVocab.rE || tm.rF != tfmVocab.rF) {
        std::fprintf(stderr, "%s: vocabulary differs from %s\n", tfmPaths[k].c_str(), tfmPaths[0].c_str());
        return 1;
      }
      const int NB = tm.c.NB;
      // order by peak count (less padding), batches of 64 spectra
      std::vector<std::vector<float>> M(sp.size()), I(sp.size());
      for (size_t i = 0; i < sp.size(); ++i) {
        const Head& h = head[sp[i].second];
        tfmPrep(h, mz.data() + h.off, it.data() + h.off, tm.c.maxPeaks, M[i], I[i]);
      }
      std::vector<int> ord(sp.size());
      for (size_t i = 0; i < ord.size(); ++i) ord[i] = int(i);
      std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return M[a].size() < M[b].size(); });
      for (Query& q : Q) q.tP[k].assign(NB, 0.f);
      tfm::HostBatch hb;
      std::vector<float> z;
      const size_t BS = 64;
      for (size_t s0 = 0; s0 < ord.size(); s0 += BS) {
        size_t s1 = std::min(ord.size(), s0 + BS);
        int B = int(s1 - s0), mx = 1;
        for (size_t i = s0; i < s1; ++i) mx = std::max(mx, int(M[ord[i]].size()));
        hb.B = B; hb.N = 1 + mx;
        hb.mz.assign(size_t(B) * mx, 0.f); hb.it.assign(size_t(B) * mx, 0.f);
        hb.prec.resize(B); hb.mode.resize(B); hb.np.resize(B); hb.ad.resize(B); hb.in.resize(B);
        for (int b = 0; b < B; ++b) {
          int i = ord[s0 + b];
          const Head& h = head[sp[i].second];
          std::copy(M[i].begin(), M[i].end(), hb.mz.begin() + size_t(b) * mx);
          std::copy(I[i].begin(), I[i].end(), hb.it.begin() + size_t(b) * mx);
          hb.np[b] = int(M[i].size());
          hb.prec[b] = h.precursor;
          tfmTag(h.tag, hb.ad[b], hb.in[b], hb.mode[b]);
          nTok += 1 + hb.np[b];
        }
        tm.setBatch(hb);
        tm.forward(false, 0.f, 0);
        z.resize(size_t(B) * NB);
        tfm::download(tm.Z, z.data(), int64_t(z.size()));
        for (int b = 0; b < B; ++b) {
          Query& q = Q[sp[ord[s0 + b]].first];
          const float inv = 1.f / float(q.spectra.size());
          for (int j = 0; j < NB; ++j) q.tP[k][j] += inv / (1.f + std::exp(-z[size_t(b) * NB + j]));
        }
      }
      std::printf("TFM model %s: d%d L%d H%d HH%d, %d vocab bits (ECFP r%d + FCFP r%d)\n", tfmPaths[k].c_str(), tm.c.d,
                  tm.c.L, tm.c.H, tm.c.HH, NB, tm.rE, tm.rF);
    }
    if (tfmNM > 1)
      for (Query& q : Q) {
        q.tP[tfmNM].assign(q.tP[0].size(), 0.f);
        for (int k = 0; k < tfmNM; ++k)
          for (size_t j = 0; j < q.tP[k].size(); ++j) q.tP[tfmNM][j] += q.tP[k][j] / float(tfmNM);
      }
    double tt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("TFM inference: %zu spectra, %d model(s), %.1f s wall with %u threads = %.1f ms/spectrum/model; "
                "mean %.1f tokens; fusion + %g z(col %d)%s\n",
                sp.size(), tfmNM, tt, nThr, 1000.0 * tt / std::max<size_t>(1, sp.size()) / tfmNM,
                double(nTok) / std::max<size_t>(1, sp.size() * tfmNM), tfmW, tfmV, tfmShuf ? " SHUFFLED (control)" : "");
    std::fflush(stdout);
  }

  // candidates + fragment tables (the expensive part, done once)
  RDKit::PeriodicTable::getTable();
  std::mutex timeMu;
  // v38pc caches (see the header); the tfm columns and fit mode need the molecules -> no pool cache there
  const bool poolCacheOn = optI("poolcache", 0) != 0 && tfmNM == 0 && mode != "fit";
  const bool pcCacheOn = optI("pccache", 1) != 0 && pcFp && pcFpTop <= 0 && !pcRecall && fpIndex.size() <= 65536;
  ShardCache<CandFeat> poolCache;
  ShardCache<PcFeat> pcCache;
  const long pcCacheMax = optI("pccachemax", 3000000);  // entries (~0.9 KB of process memory each, measured); beyond: computed, not stored
  std::atomic<long> nPcFeatSum(0), nPcFeatMol(0);
  double prof[PR_N] = {0};
  std::atomic<long> nPoolLook(0), nPcLook(0), nPcSeenDup(0), nPcFull(0);
  std::printf("v38: poolcache %d, pccache %d\n", int(poolCacheOn), int(pcCacheOn));
  double tStage1 = 0, tAnaTotal = 0;
  double tGen = 0, tPc = 0;
  std::atomic<long> nRmDup(0), nTruthDup(0), nOutWin(0);
  const std::string detPath = opt.count("det") ? opt["det"] : "";
  const bool keepFrags = mode == "fit";
  Params P0;
  if (!keepFrags && !P0.load(dir + "/" + paramPath)) std::printf("(params %s absent: defaults)\n", paramPath.c_str());
  size_t nAnaSpec = 0;
  auto tEnum0 = std::chrono::steady_clock::now();
  std::atomic<size_t> next(0);
  std::vector<std::thread> th;
  for (unsigned t = 0; t < nThr; ++t)
    th.emplace_back([&]() {
      std::vector<float> acc(haveLib ? lhead.size() : 0, 0.f), acc2(haveLib ? lhead.size() : 0, 0.f);
      // v34pc attention accumulators (per library spectrum; c* = current query peak)
      const size_t attNS = attOn ? lhead.size() : 0, attNM = attOn ? lnames.size() : 0;
      std::vector<float> aF(attNS, 0.f), aE(attNS, 0.f), aMF(attNS, 0.f), aME(attNS, 0.f);
      std::vector<float> cF(attNS, 0.f), cE(attNS, 0.f), cMF(attNS, 0.f), cME(attNS, 0.f);
      std::vector<uint16_t> aNM(attNS, 0);
      std::vector<char> aSeen(attNS, 0), cSeen(attNS, 0);
      std::vector<uint32_t> aTouched, cTouched;
      std::vector<float> aBest(attNM, 0.f);
      std::vector<int> aMols;
      std::vector<AttPk> aQp;
      std::vector<uint8_t> mark(haveLib ? lhead.size() : 0, 0);
      std::vector<int32_t> touched;
      std::vector<float> qw;
      std::vector<std::pair<float, int32_t>> short1;
      std::vector<PairT> pairs;
      std::vector<uint8_t> usedQ, usedL;
      std::vector<std::pair<double, double>> predBuf;
      double pr[PR_N] = {0};
      // stereo-blind key (= the per-query canon below)
      auto canonT = [](const RDKit::ROMol& m) -> std::string {
        std::string k = gen::key(m);
        std::unique_ptr<RDKit::RWMol> r;
        try { r.reset(RDKit::SmilesToMol(k)); } catch (...) { r.reset(); }
        return r ? gen::key(*r) : k;
      };
      // v38pc: the query-independent work on one structure (same calls as the v37 per-candidate code)
      auto featOf = [&](const RDKit::ROMol& m, bool wantKey, CandFeat& F) {
        auto t = std::chrono::steady_clock::now();
        F.ok = true;
        if (wantKey) { F.key = canonT(m); pr[PR_F_PARSE] += secSince(t); t = std::chrono::steady_clock::now(); }
        F.nAtoms = int(m.getNumAtoms());
        F.fr = enumerate(m);
        pr[PR_F_FRAG] += secSince(t); t = std::chrono::steady_clock::now();
        if (haveMlp) { F.nf = forward_model_nn::enumerateFragments(m); pr[PR_F_NF] += secSince(t); t = std::chrono::steady_clock::now(); }
        if (haveRev) {
          F.revBits = rev2Mode ? revVocab.bits(m) : ecfp::compute(m, 2, revNB);
          pr[PR_F_REV] += secSince(t); t = std::chrono::steady_clock::now();
        }
        if ((haveRev && pcFp) || ftOn) {
          std::vector<uint64_t> f;
          try { f = fp3::features(m); } catch (...) { f.clear(); }
          if (haveRev && pcFp)
            for (uint64_t h : f) { auto itf = fpIndex.find(h); if (itf != fpIndex.end()) F.fpI.push_back(itf->second); }
          if (ftOn)
            for (uint64_t h : f) { auto itf = ftIndex.find(h); if (itf != ftIndex.end()) F.ftI.push_back(itf->second); }
          pr[PR_F_FP3] += secSince(t); t = std::chrono::steady_clock::now();
        }
        if (cnnOn) { F.cnnBits = cnnVocab.bits(m); pr[PR_F_CNN] += secSince(t); t = std::chrono::steady_clock::now(); }
        if (haveLib) { F.ecfp = ecfp::compute(m, 2, 2048); pr[PR_F_ECFP] += secSince(t); }
      };
      for (;;) {
        size_t qi = next++;
        if (qi >= Q.size()) {
          std::lock_guard<std::mutex> lk(timeMu);
          for (int k = 0; k < PR_N; ++k) prof[k] += pr[k];
          break;
        }
        Query& q = Q[qi];
        // Spectral analogs: library molecules whose spectra look like this
        // query's (any structure, precursor within 200 Da), top 80 by
        // similarity, with their fingerprints.
        // analogs[v] = (sim, fingerprint) of the top 80 molecules for variant v
        std::vector<std::pair<float, const std::vector<uint16_t>*>> analogs[NVAR];
        std::unordered_map<int, std::vector<uint16_t>> fpCache;
        std::vector<std::pair<float, int>> genAn;  // (sim, library molecule) for generation
        if (haveLib) {
          auto t0 = std::chrono::steady_clock::now();
          double t1s = 0;
          std::unordered_map<int, float> molBest[NVAR];
          for (int s : q.spectra) {
            auto ta = std::chrono::steady_clock::now();
            const Head& h = head[s];
            double nrm = 0;
            for (int i = 0; i < h.n; ++i) nrm += std::max(0.f, it[h.off + i]);
            nrm = std::sqrt(nrm) + 1e-12;
            touched.clear();
            qw.resize(h.n);
            double qnNP2 = 0;
            for (int i = 0; i < h.n; ++i) {
              float wq = float(std::sqrt(std::max(0.f, it[h.off + i])) / nrm);
              qw[i] = wq;
              if (mz[h.off + i] <= h.precursor - 1.f) qnNP2 += double(wq) * wq;
              // fragment index: the peak's own bin and its nearer neighbour
              float x = mz[h.off + i] * 100.f;
              int b = int(x);
              int b2 = (x - float(b)) < 0.5f ? b - 1 : b + 1;
              for (int bb : {b, b2}) {
                if (bb < 0 || bb >= NBIN) continue;
                for (int64_t k = binStart[bb]; k < binStart[bb + 1]; ++k) {
                  int32_t ls = postSpec[k];
                  if (!mark[ls]) { mark[ls] = 1; touched.push_back(ls); }
                  acc[ls] += wq * postW[k];
                }
              }
              // neutral-loss index
              float y = (h.precursor - mz[h.off + i]) * 100.f + 0.5f;
              if (y < 0.f) continue;
              int c = int(y);
              int c2 = (y - float(c)) < 0.5f ? c - 1 : c + 1;
              for (int bb : {c, c2}) {
                if (bb < 0 || bb >= NBIN) continue;
                for (int64_t k = nlStart[bb]; k < nlStart[bb + 1]; ++k) {
                  int32_t ls = nlSpec[k];
                  if (!mark[ls]) { mark[ls] = 1; touched.push_back(ls); }
                  acc2[ls] += wq * nlW[k];
                }
              }
            }
            short1.clear();
            for (int32_t ls : touched) {
              float a1 = acc[ls], a2 = acc2[ls];
              acc[ls] = 0.f; acc2[ls] = 0.f; mark[ls] = 0;
              if (std::fabs(lhead[ls].precursor - h.precursor) > 200.f) continue;
              float sim = std::min(1.f, a1);
              if (sim >= 0.3f) { auto& mb = molBest[0][lhead[ls].mol]; mb = std::max(mb, sim); }
              if (a1 + a2 >= 0.2f) short1.emplace_back(a1 + a2, ls);
            }
            if (int(short1.size()) > shortK) {
              std::nth_element(short1.begin(), short1.begin() + shortK, short1.end(),
                               [](auto& a, auto& b) { return a.first > b.first; });
              short1.resize(shortK);
            }
            auto tb = std::chrono::steady_clock::now();
            t1s += std::chrono::duration<double>(tb - ta).count();
            // stage 2: exact (modified) cosine on the shortlist
            float qnNP = float(std::sqrt(qnNP2));
            for (auto& sl : short1) {
              int32_t ls = sl.second;
              const Head& lh = lhead[ls];
              McosOut o;
              modcos(mz.data() + h.off, qw.data(), h.n, h.precursor, qnNP, lmz.data() + lh.off,
                     lit.data() + lh.off, lh.n, lh.precursor, lnrm[ls], 0.01f, pairs, usedQ, usedL, o);
              float v[NVAR] = {0.f, o.cos, o.mcos, o.mcosNP, o.nMatch >= 6 ? o.mcos : 0.f,
                              o.nMatchNP >= 4 ? o.mcosNP : 0.f, o.nMatchNP >= 6 ? o.mcosNP : 0.f,
                              o.nMatchNP >= 8 ? o.mcosNP : 0.f, o.nMatchU >= 6 ? o.cos : 0.f};
              for (int vi = 1; vi < NVAR; ++vi)
                if (v[vi] >= 0.3f) { auto& mb = molBest[vi][lh.mol]; mb = std::max(mb, v[vi]); }
            }
          }
          for (int vi = 0; vi < NVAR; ++vi) {
            std::vector<std::pair<float, int>> top;
            for (auto& kv : molBest[vi]) top.emplace_back(kv.second, kv.first);
            std::sort(top.begin(), top.end(), [](auto& a, auto& b) { return a.first > b.first; });
            if (vi == genVar)
              for (size_t k = 0; k < top.size() && int(k) < genNan; ++k) genAn.push_back(top[k]);
            if (top.size() > 80) top.resize(80);
            for (auto& tp : top) {
              auto fc = fpCache.find(tp.second);
              if (fc == fpCache.end()) {
                std::unique_ptr<RDKit::RWMol> am;
                try { am.reset(RDKit::SmilesToMol(lsmiles[tp.second])); } catch (...) { am.reset(); }
                fc = fpCache.emplace(tp.second, am ? ecfp::compute(*am, 2, 2048) : std::vector<uint16_t>()).first;
              }
              if (fc->second.empty()) continue;
              analogs[vi].emplace_back(tp.first, &fc->second);
            }
          }
          double tt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
          pr[PR_ANA] += tt;
          {
            std::lock_guard<std::mutex> lk(timeMu);
            tStage1 += t1s; tAnaTotal += tt; nAnaSpec += q.spectra.size();
          }
        }
        // v34pc: attention retrieval -> predicted fp3 bit probabilities q.attP
        const auto tAtt0 = std::chrono::steady_clock::now();
        if (attOn) {
          aMols.clear();
          for (int s : q.spectra) {
            const Head& h = head[s];
            attPrep(h.precursor, mz.data() + h.off, it.data() + h.off, h.n, attPkN, aQp);
            int a = adductIdx(h.tag);
            const int pol = a < 0 || a >= 10 ? 0 : (ADDUCT_NEG[a] ? -1 : 1);
            aTouched.clear();
            auto scan = [&](const AttIndex& ix, float v, const AttPk& k, bool loss) {
              if (v <= 0.f) return;
              int b0 = int(v / ix.bw);
              for (int b = std::max(0, b0 - 1); b <= std::min(ix.nb - 1, b0 + 1); ++b)
                for (uint64_t j = ix.off[b]; j < ix.off[b + 1]; ++j) {
                  const AttPost& P = ix.p[j];
                  if (std::fabs(P.v - v) > attTol) continue;
                  const uint32_t sp = P.spec;
                  if (pol != 0 && attPol[sp] != 0 && attPol[sp] != pol) continue;
                  if (loss && !(lhead[sp].precursor > 0.f && std::fabs(h.precursor - lhead[sp].precursor) > attTol)) continue;
                  if (!cSeen[sp]) { cSeen[sp] = 1; cTouched.push_back(sp); }
                  float vs = k.s * P.s, ve = attGEnt(k.e, P.e);
                  if (!loss) { cF[sp] = std::max(cF[sp], vs); cE[sp] = std::max(cE[sp], ve); }
                  cMF[sp] = std::max(cMF[sp], vs); cME[sp] = std::max(cME[sp], ve);
                }
            };
            for (const AttPk& k : aQp) {
              cTouched.clear();
              scan(attFx, k.mz, k, false);
              if (h.precursor > 0.f) scan(attLx, h.precursor - k.mz, k, true);
              for (uint32_t sp : cTouched) {
                if (!aSeen[sp]) { aSeen[sp] = 1; aTouched.push_back(sp); }
                aF[sp] += cF[sp]; aE[sp] += cE[sp]; aMF[sp] += cMF[sp]; aME[sp] += cME[sp];
                ++aNM[sp];
                cF[sp] = cE[sp] = cMF[sp] = cME[sp] = 0.f; cSeen[sp] = 0;
              }
            }
            for (uint32_t sp : aTouched) {
              float pen = attMp > 0 ? std::min(1.f, float(aNM[sp]) / float(attMp)) : 1.f;
              float v = attSim == 0 ? aF[sp] : attSim == 1 ? aE[sp] : attSim == 2 ? aMF[sp] : aME[sp];
              v = std::max(1e-9f, std::min(1.f, v) * pen);
              int lm = lhead[sp].mol;
              if (lm >= 0 && size_t(lm) < attNM) {
                if (aBest[lm] == 0.f) aMols.push_back(lm);
                aBest[lm] = std::max(aBest[lm], v);
              }
              aF[sp] = aE[sp] = aMF[sp] = aME[sp] = 0.f; aNM[sp] = 0; aSeen[sp] = 0;
            }
          }
          std::vector<std::pair<float, int>> nb;
          nb.reserve(aMols.size());
          for (int lm : aMols) { nb.emplace_back(aBest[lm], lm); aBest[lm] = 0.f; }
          // ties broken by library molecule index (deterministic)
          auto cmpN = [](const std::pair<float, int>& x, const std::pair<float, int>& y) {
            return x.first != y.first ? x.first > y.first : x.second < y.second;
          };
          const size_t K = std::min(nb.size(), size_t(attK));
          std::partial_sort(nb.begin(), nb.begin() + K, nb.end(), cmpN);
          nb.resize(K);
          if (!nb.empty()) {
            q.attP.assign(fpIndex.size(), 0.f);
            double ws = 0;
            std::vector<double> w(nb.size());
            for (size_t j = 0; j < nb.size(); ++j) { w[j] = std::exp((double(nb[j].first) - nb[0].first) / attT); ws += w[j]; }
            std::vector<int> bits;
            for (size_t j = 0; j < nb.size(); ++j) {
              std::unique_ptr<RDKit::RWMol> am;
              try { am.reset(RDKit::SmilesToMol(lsmiles[nb[j].second])); } catch (...) { am.reset(); }
              if (!am) continue;
              std::vector<uint64_t> f;
              try { f = fp3::features(*am); } catch (...) { f.clear(); }
              bits.clear();
              for (uint64_t hh : f) { auto itf = fpIndex.find(hh); if (itf != fpIndex.end()) bits.push_back(itf->second); }
              std::sort(bits.begin(), bits.end());
              bits.erase(std::unique(bits.begin(), bits.end()), bits.end());
              const float wj = float(w[j] / ws);
              for (int b : bits) q.attP[b] += wj;
            }
            q.attSumP = 0.0;
            for (float x : q.attP) q.attSumP += x;
            q.attN = int(nb.size()); q.attTop = nb[0].first;
          }
        }
        pr[PR_ATT] += secSince(tAtt0);
        // neutral-mass windows of this query (one per spectrum / adduct)
        std::vector<std::pair<double, double>> win;
        std::set<int> cs;
        for (int s : q.spectra) {
          const Head& h = head[s];
          int a = adductIdx(h.tag);
          float shift = (a >= 0 && a < 10) ? ADDUCT_SHIFT[a] : 1.007276f;
          float neutral = h.precursor - shift;
          if (a == 8 || a == 9) neutral *= 0.5f;
          float w = neutral * ppm * 1e-6f;
          win.emplace_back(double(neutral) - w, double(neutral) + w);
          size_t lo = std::lower_bound(pmass.begin(), pmass.end(), neutral - w) - pmass.begin();
          size_t hi = std::upper_bound(pmass.begin(), pmass.end(), neutral + w) - pmass.begin();
          for (size_t k = lo; k < hi; ++k) cs.insert(pid[k]);
        }
        // stereo-blind key: non-isomeric canonical SMILES, re-parsed once so
        // pool, truth and generated structures go through the same path
        auto canon = [](const RDKit::ROMol& m) -> std::string {
          std::string k = gen::key(m);
          std::unique_ptr<RDKit::RWMol> r;
          try { r.reset(RDKit::SmilesToMol(k)); } catch (...) { r.reset(); }
          return r ? gen::key(*r) : k;
        };
        const bool needKeys = doGen || rmTruth || doPc;
        std::string tkey;
        if (needKeys && !q.tsmi.empty()) {
          std::unique_ptr<RDKit::RWMol> tm;
          try { tm.reset(RDKit::SmilesToMol(q.tsmi)); } catch (...) { tm.reset(); }
          if (tm) tkey = canon(*tm);
        }
        int truthPool = -1;
        {
          auto tp = poolIdx.find(q.name);
          if (tp != poolIdx.end()) truthPool = tp->second;
        }
        // v38pc: (molecule or null when it came from the cache, candidate code, per-molecule features)
        struct ClItem { std::unique_ptr<RDKit::RWMol> mol; int code; const CandFeat* F; };
        std::vector<ClItem> cl;
        std::deque<CandFeat> ownF;  // features of this query's uncached candidates
        std::unordered_map<std::string, int> poolKeys;
        bool truthInPool = false;
        const auto tPool0 = std::chrono::steady_clock::now();
        for (int c : cs) {
          if (rmTruth && c == truthPool) continue;
          std::unique_ptr<RDKit::RWMol> m;
          const CandFeat* F = nullptr;
          if (poolCacheOn) {
            ++nPoolLook;
            F = poolCache.get(c);
            if (!F) {
              CandFeat nf;
              auto tp = std::chrono::steady_clock::now();
              try { m.reset(RDKit::SmilesToMol(psmi[c])); } catch (...) { m.reset(); }
              pr[PR_F_PARSE] += secSince(tp);
              if (m) featOf(*m, needKeys, nf);
              m.reset();
              F = poolCache.put(c, std::move(nf));
            }
            if (!F->ok) continue;
          } else {
            auto tp = std::chrono::steady_clock::now();
            try { m.reset(RDKit::SmilesToMol(psmi[c])); } catch (...) { continue; }
            pr[PR_F_PARSE] += secSince(tp);
            if (!m) continue;
            ownF.emplace_back();
            featOf(*m, needKeys, ownF.back());
            F = &ownF.back();
          }
          if (needKeys) {
            const std::string& k = F->key;
            if (rmTruth && !tkey.empty() && k == tkey) { ++nRmDup; continue; }  // stereo-blind duplicate of the answer
            poolKeys.emplace(k, c);
          }
          if (c == truthPool) truthInPool = true;
          cl.push_back(ClItem{std::move(m), c, F});
        }
        pr[PR_POOLPARSE] += secSince(tPool0);
        std::vector<std::array<float, NG>> gfeat(cl.size(), std::array<float, NG>{0, 0, 0, 0, 0, 0});
        if (doGen && !genAn.empty()) {
          auto tg0 = std::chrono::steady_clock::now();
          std::unordered_map<std::string, gen::Prod> prod;
          std::vector<std::vector<int>> chains;
          for (size_t r = 0; r < genAn.size(); ++r) {
            std::unique_ptr<RDKit::RWMol> am;
            try {
              std::unique_ptr<RDKit::RWMol> a0(RDKit::SmilesToMol(lsmiles[genAn[r].second]));
              if (a0) am.reset(RDKit::SmilesToMol(gen::key(*a0)));  // drop stereo
            } catch (...) { am.reset(); }
            if (!am) continue;
            gen::chainsFor(gen::molMass(*am), win, chains, twoStep);
            if (chains.empty()) continue;
            try {
              gen::generateFrom(*am, int(r), genAn[r].first, chains, prod, 2000);
            } catch (...) {
            }
          }
          // canonical re-key, merge, pool dedupe, mass check
          std::unordered_map<std::string, gen::Prod*> byKey;
          for (auto& kv : prod) {
            std::string k = canon(*kv.second.mol);
            auto ins = byKey.emplace(k, &kv.second);
            if (!ins.second) {
              gen::Prod& a = *ins.first->second;
              gen::Prod& b = kv.second;
              if (b.sim > a.sim || (b.sim == a.sim && b.nSib < a.nSib)) { a.sim = b.sim; a.nSib = b.nSib; }
              a.rank = std::min(a.rank, b.rank);
              a.steps = std::min(a.steps, b.steps);
              for (int s : b.sup) if (std::find(a.sup.begin(), a.sup.end(), s) == a.sup.end()) a.sup.push_back(s);
              for (auto& s : b.chains) if (std::find(a.chains.begin(), a.chains.end(), s) == a.chains.end()) a.chains.push_back(s);
            }
          }
          q.nGenRaw = int(byKey.size());
          q.truthGenRaw = !tkey.empty() && byKey.count(tkey) > 0;
          std::vector<std::pair<const std::string*, gen::Prod*>> items;
          for (auto& kv : byKey) {
            if (poolKeys.count(kv.first)) continue;
            if (truthInPool && kv.first == tkey) { ++nTruthDup; continue; }  // the answer's own pool entry stays
            double pm = gen::molMass(*kv.second->mol);
            bool inw = false;
            for (auto& w : win) if (pm >= w.first - 1e-4 && pm <= w.second + 1e-4) inw = true;
            if (!inw) { ++nOutWin; continue; }
            items.emplace_back(&kv.first, kv.second);
          }
          std::sort(items.begin(), items.end(), [](auto& a, auto& b) {
            if (a.second->sim != b.second->sim) return a.second->sim > b.second->sim;
            if (a.second->steps != b.second->steps) return a.second->steps < b.second->steps;
            if (a.second->rank != b.second->rank) return a.second->rank < b.second->rank;
            return *a.first < *b.first;
          });
          if (int(items.size()) > genCap) items.resize(genCap);
          std::ostringstream det;
          for (auto& itm : items) {
            std::unique_ptr<RDKit::RWMol> gm;
            try { gm.reset(RDKit::SmilesToMol(*itm.first)); } catch (...) { gm.reset(); }
            if (!gm) continue;
            const gen::Prod& p = *itm.second;
            int gi = int(q.gSmi.size());
            q.gSmi.push_back(*itm.first);
            float pr = -1e9f;
            std::string chs;
            for (auto& ch : p.chains) { pr = std::max(pr, chainPrior(ch)); chs += (chs.empty() ? "" : ";") + ch; }
            gfeat.push_back({1.f, p.sim * p.sim * p.sim, p.steps == 2 ? 1.f : 0.f, std::log1p(float(p.sup.size())),
                             std::log1p(float(p.nSib)), opPrior.empty() ? 0.f : pr});
            ownF.emplace_back();
            featOf(*gm, false, ownF.back());
            cl.push_back(ClItem{std::move(gm), -(gi + 1), &ownF.back()});
            bool isT = !tkey.empty() && *itm.first == tkey;
            if (isT) q.truthGenKept = true;
            if (!detPath.empty())
              det << q.name << "\t" << *itm.first << "\t" << int(isT) << "\t" << p.sim << "\t" << p.rank << "\t"
                  << p.steps << "\t" << p.sup.size() << "\t" << p.nSib << "\t" << chs << "\n";
          }
          q.det = det.str();
          double tg = std::chrono::duration<double>(std::chrono::steady_clock::now() - tg0).count();
          pr[PR_GEN] += tg;
          std::lock_guard<std::mutex> lk(timeMu);
          tGen += tg;
        }
        // physics / forward-MLP / reverse-model scores of one structure
        auto physOf = [&](const RDKit::ROMol& m) {
          auto fr = enumerate(m);
          float eps = std::exp(P0.logEps), sc = 0.f;
          for (int s : q.spectra) {
            const Head& h = head[s];
            predict(fr, P0, adductIdx(h.tag), int(m.getNumAtoms()), predBuf);
            sc += loglik(predBuf, mz.data() + h.off, it.data() + h.off, h.n, 0.01f, eps, h.precursor);
          }
          return sc / float(q.spectra.size());
        };
        auto mlpOf = [&](const RDKit::ROMol& m) {
          auto nf = forward_model_nn::enumerateFragments(m);
          float best = 0.f;
          for (int s : q.spectra) {
            const Head& h = head[s];
            forward_model_nn::Context ctx;
            int a = adductIdx(h.tag);
            ctx.adductIdx = a < forward_model_nn::Context::kNumAdducts ? a : -1;
            ctx.instrumentIdx = (h.tag < 0 || h.tag % 16 == 15) ? -1 : h.tag % 16;
            auto pred = forward_model_nn::scoreFragmentsWithContext(nf, mlp, ctx);
            best = std::max(best, mlpCosine(pred, mz.data() + h.off, it.data() + h.off, h.n));
          }
          return best;
        };
        auto revOf = [&](const RDKit::ROMol& m) {
          float r = 0.f;
          if (rev2Mode) {
            for (uint16_t b : revVocab.bits(m)) r += q.logit[b];
          } else {
            for (uint16_t b : ecfp::compute(m, 2, revNB)) r += q.logit[b];
          }
          return r;
        };
        // v30pc: REV2 score in form mode (0 = revOf; 1 = continuous Tanimoto
        // sum_bits p / (n + sum_all p - sum_bits p); 2 = sum of calibrated weights), as win_eval.cpp
        auto revOfM = [&](const RDKit::ROMol& m, int mode) {
          if (mode == 0) return revOf(m);
          std::vector<uint16_t> bb = rev2Mode ? revVocab.bits(m) : ecfp::compute(m, 2, revNB);
          if (mode == 2) {
            float s = 0.f;
            for (uint16_t b : bb) s += q.revCalU[b];
            return s;
          }
          double in = 0;
          for (uint16_t b : bb) in += q.revP[b];
          return float(in / (double(bb.size()) + q.revSumP - in + 1e-9));
        };
        // v32pc: continuous Tanimoto of the fp3 model's p over the candidate's fp3 vocabulary bits
        // v34pc: the same pass also gives the attention score (lastAtt) when q.attP is set
        float lastAtt = 0.f;
        const bool qAtt = attOn && !q.attP.empty();
        auto fp3Of = [&](const RDKit::ROMol& m) {
          std::vector<uint64_t> f;
          try { f = fp3::features(m); } catch (...) { f.clear(); }
          double in = 0, inA = 0;
          size_t nb = 0;
          for (uint64_t h : f) {
            auto itf = fpIndex.find(h);
            if (itf != fpIndex.end()) { in += q.fpP[itf->second]; ++nb; if (qAtt) inA += q.attP[itf->second]; }
          }
          lastAtt = qAtt ? float(inA / (double(nb) + q.attSumP - inA + 1e-9)) : 0.f;
          return float(in / (double(nb) + q.fpSumP - in + 1e-9));
        };
        // max_a Tanimoto(cand, a) * sim_a^(2,3,4) over the analogs of variant vi
        auto anaOf = [&](const std::vector<uint16_t>& cb, int vi, float ap[3]) {
          ap[0] = ap[1] = ap[2] = 0.f;
          for (auto& an : analogs[vi]) {
            const auto& af = *an.second;
            size_t i = 0, j = 0, inter = 0;
            while (i < cb.size() && j < af.size()) {
              if (cb[i] == af[j]) { ++inter; ++i; ++j; }
              else if (cb[i] < af[j]) ++i;
              else ++j;
            }
            size_t uni = cb.size() + af.size() - inter;
            float tan = uni ? float(inter) / float(uni) : 0.f;
            float s2 = an.first * an.first;
            ap[0] = std::max(ap[0], tan * s2);
            ap[1] = std::max(ap[1], tan * s2 * an.first);
            ap[2] = std::max(ap[2], tan * s2 * s2);
          }
        };
        // v38pc: the same scores from the per-molecule lists (identical operations in identical order)
        auto revOfF = [&](const std::vector<uint16_t>& bb, int mode) {
          if (mode == 0) {
            float r = 0.f;
            for (uint16_t b : bb) r += q.logit[b];
            return r;
          }
          if (mode == 2) {
            float s = 0.f;
            for (uint16_t b : bb) s += q.revCalU[b];
            return s;
          }
          double in = 0;
          for (uint16_t b : bb) in += q.revP[b];
          return float(in / (double(bb.size()) + q.revSumP - in + 1e-9));
        };
        for (size_t ci = 0; ci < cl.size(); ++ci) {
          const CandFeat& F = *cl[ci].F;
          const int c = cl[ci].code;
          q.cand.push_back(c);
          q.gF.push_back(gfeat[ci]);
          auto ts = std::chrono::steady_clock::now();
          if (keepFrags) {
            q.frags.push_back(F.fr);
          } else {
            // physics score now; the fragment table is not kept (memory)
            float eps = std::exp(P0.logEps), sc = 0.f;
            for (int s : q.spectra) {
              const Head& h = head[s];
              predict(F.fr, P0, adductIdx(h.tag), F.nAtoms, predBuf);
              sc += loglik(predBuf, mz.data() + h.off, it.data() + h.off, h.n, 0.01f, eps, h.precursor);
            }
            q.phys.push_back(sc / float(q.spectra.size()));
          }
          pr[PR_S_PHYS] += secSince(ts); ts = std::chrono::steady_clock::now();
          q.candN.push_back(F.nAtoms);
          if (haveMlp) {
            float best = 0.f;
            for (int s : q.spectra) {
              const Head& h = head[s];
              forward_model_nn::Context ctx;
              int a = adductIdx(h.tag);
              ctx.adductIdx = a < forward_model_nn::Context::kNumAdducts ? a : -1;
              ctx.instrumentIdx = (h.tag < 0 || h.tag % 16 == 15) ? -1 : h.tag % 16;
              auto pred = forward_model_nn::scoreFragmentsWithContext(F.nf, mlp, ctx);
              best = std::max(best, mlpCosine(pred, mz.data() + h.off, it.data() + h.off, h.n));
            }
            q.mlpCos.push_back(best);
          }
          pr[PR_S_MLP] += secSince(ts); ts = std::chrono::steady_clock::now();
          if (haveRev) {
            q.revLL.push_back(revOfF(F.revBits, poolRevMode));
            if (pcFp) {
              double in = 0, inA = 0;
              size_t nb = 0;
              for (int32_t b : F.fpI) { in += q.fpP[b]; ++nb; if (qAtt) inA += q.attP[b]; }
              q.revPc.push_back(float(in / (double(nb) + q.fpSumP - in + 1e-9)));
              if (qAtt) q.attPc.push_back(float(inA / (double(nb) + q.attSumP - inA + 1e-9)));
            }
            else if (doPc && pcRevMode != poolRevMode) q.revPc.push_back(revOfF(F.revBits, pcRevMode));
          }
          if (cnnOn && !q.cnnP.empty()) {
            double in = 0;
            const std::vector<uint16_t>& bb = F.cnnBits;
            for (uint16_t b : bb) in += q.cnnP[b];
            q.cnnPc.push_back(float(in / (double(bb.size()) + q.cnnSumP - in + 1e-9)));
          }
          if (ftOn && !q.ftP.empty()) {  // v37pc: continuous Tanimoto over the candidate's fp3 bits in the FT vocabulary
            double in = 0;
            size_t nb = 0;
            for (int32_t b : F.ftI) { in += q.ftP[b]; ++nb; }
            q.ftPc.push_back(float(in / (double(nb) + q.ftSumP - in + 1e-9)));
          }
          pr[PR_S_BITS] += secSince(ts); ts = std::chrono::steady_clock::now();
          if (haveLib) {
            // sqrt-intensity cosine, each query peak to its best library peak within 0.01 Da
            float best = 0.f;
            auto ls = c >= 0 ? libSpecOf.find(pname[c]) : libSpecOf.end();
            if (ls != libSpecOf.end())
              for (int s : q.spectra) {
                const Head& h = head[s];
                double nq = 0;
                for (int i = 0; i < h.n; ++i) nq += it[h.off + i];
                for (int r : ls->second) {
                  const Head& lh = lhead[r];
                  double nl = 0, dot = 0;
                  for (int j = 0; j < lh.n; ++j) nl += lit[lh.off + j];
                  int j0 = 0;
                  for (int i = 0; i < h.n; ++i) {
                    double mm = mz[h.off + i], bestI = 0;
                    while (j0 < lh.n && lmz[lh.off + j0] < mm - 0.01) ++j0;
                    for (int j = j0; j < lh.n && lmz[lh.off + j] <= mm + 0.01; ++j)
                      bestI = std::max(bestI, double(lit[lh.off + j]));
                    dot += std::sqrt(std::max(0.f, it[h.off + i]) * bestI);
                  }
                  if (nq > 0 && nl > 0) best = std::max(best, float(dot / std::sqrt(nq * nl)));
                }
              }
            q.libSim.push_back(libCap ? std::min(best, 1.f) : best);
            // analog propagation: a candidate structurally close to a
            // library molecule whose spectrum matches is likely right
            const std::vector<uint16_t>& cb = F.ecfp;
            for (int vi = 0; vi < NVAR; ++vi) {
              float ap[3];
              anaOf(cb, vi, ap);
              for (int pi = 0; pi < 3; ++pi) q.anaV[vi][pi].push_back(ap[pi]);
            }
            q.anaP.push_back(q.anaV[anaVar][anaPowI].back());
          }
          pr[PR_S_LIB] += secSince(ts);
        }
        // ---- PubChem channel ------------------------------------------------
        if (doPc && haveRev && !keepFrags) {
          auto tp0 = std::chrono::steady_clock::now();
          for (float l : q.libSim) q.libMax = std::max(q.libMax, l);
          if (!genAn.empty()) q.anaTop = genAn[0].first;
          // the truth's formula signature (recall study)
          auto sigOf = [](const RDKit::ROMol& m) {
            std::map<int, int> cnt;
            int chg = 0;
            for (const auto a : m.atoms()) { ++cnt[a->getAtomicNum()]; cnt[1] += int(a->getTotalNumHs(false)); chg += a->getFormalCharge(); }
            std::string s;
            for (auto& kv : cnt) s += std::to_string(kv.first) + ":" + std::to_string(kv.second) + ",";
            return s + "q" + std::to_string(chg);
          };
          std::string tsig;
          if (pcRecall && !tkey.empty()) {
            std::unique_ptr<RDKit::RWMol> tm;
            try { tm.reset(RDKit::SmilesToMol(tkey)); } catch (...) { tm.reset(); }
            if (tm) tsig = sigOf(*tm);
          }
          // window ranges in the mass-sorted table (merged; one contiguous read each)
          std::vector<std::pair<size_t, size_t>> rng;
          for (auto& w : win) {
            size_t lo = std::lower_bound(pcMass.begin(), pcMass.end(), float(w.first)) - pcMass.begin();
            size_t hi = std::upper_bound(pcMass.begin(), pcMass.end(), float(w.second)) - pcMass.begin();
            if (hi > lo) rng.emplace_back(lo, hi);
          }
          std::sort(rng.begin(), rng.end());
          std::vector<std::pair<size_t, size_t>> mr;
          for (auto& r : rng) {
            if (!mr.empty() && r.first <= mr.back().second) mr.back().second = std::max(mr.back().second, r.second);
            else mr.push_back(r);
          }
          struct W { size_t k; float rev, pop, pre; int32_t cid; std::string smi; bool truth; float att = 0.f; };
          std::vector<W> all;
          std::ifstream fs(pcDir + "/pc_smi.bin", std::ios::binary);
          std::ifstream fc(pcDir + "/pc_cid.bin", std::ios::binary);
          std::string buf;
          std::vector<int32_t> cidb;
          for (auto& r : mr) {
            uint64_t o0 = pcOff[r.first], o1 = pcOff[r.second];
            buf.resize(size_t(o1 - o0));
            fs.seekg(std::streamoff(o0));
            fs.read(buf.data(), std::streamsize(o1 - o0));
            cidb.resize(r.second - r.first);
            if (fc) {  // optional (the Kaggle dataset has no pc_cid.bin): else the table index
              fc.seekg(std::streamoff(r.first * 4));
              fc.read(reinterpret_cast<char*>(cidb.data()), std::streamsize(cidb.size() * 4));
            } else {
              for (size_t k = r.first; k < r.second; ++k) cidb[k - r.first] = int32_t(k);
            }
            for (size_t k = r.first; k < r.second; ++k) {
              std::string smi = buf.substr(size_t(pcOff[k] - o0), size_t(pcOff[k + 1] - pcOff[k]));
              if (pcCacheOn) {  // v38pc: parse + fp3 features once per PubChem index per run
                ++nPcLook;
                const PcFeat* e = pcCache.get(int64_t(k));
                PcFeat pcTmp;
                if (!e) {
                  PcFeat v;
                  std::unique_ptr<RDKit::RWMol> m;
                  try { m.reset(RDKit::SmilesToMol(smi)); } catch (...) { m.reset(); }
                  if (m) {
                    v.ok = true;
                    std::vector<uint64_t> f;
                    try { f = fp3::features(*m); } catch (...) { f.clear(); }
                    for (uint64_t h : f) { auto itf = fpIndex.find(h); if (itf != fpIndex.end()) v.fpI.push_back(uint16_t(itf->second)); }
                  }
                  nPcFeatSum += long(v.fpI.size()); ++nPcFeatMol;
                  if (long(pcCache.n) < pcCacheMax) e = pcCache.put(int64_t(k), std::move(v));
                  else { pcTmp = std::move(v); e = &pcTmp; }
                }
                if (!e->ok) continue;
                double in = 0, inA = 0;
                size_t nb = 0;
                for (uint16_t b : e->fpI) { in += q.fpP[b]; ++nb; if (qAtt) inA += q.attP[b]; }
                W w{k, float(in / (double(nb) + q.fpSumP - in + 1e-9)), pcPopV.empty() ? 0.f : pcPopV[k], 0.f, cidb[k - r.first], std::move(smi), false};
                if (qAtt) w.att = float(inA / (double(nb) + q.attSumP - inA + 1e-9));
                w.pre = w.rev + pcPP * w.pop;
                all.push_back(std::move(w));
                continue;
              }
              std::unique_ptr<RDKit::RWMol> m;
              try { m.reset(RDKit::SmilesToMol(smi)); } catch (...) { m.reset(); }
              if (!m) continue;
              W w{k, (pcFp && pcFpTop <= 0) ? fp3Of(*m) : revOfM(*m, pcRevMode), pcPopV.empty() ? 0.f : pcPopV[k], 0.f, cidb[k - r.first], std::move(smi), false};
              if (qAtt && pcFpTop <= 0) w.att = lastAtt;
              w.pre = w.rev + pcPP * w.pop;
              if (!tsig.empty() && sigOf(*m) == tsig && canon(*m) == tkey) { w.truth = true; q.pcTruthInWin = true; }
              all.push_back(std::move(w));
            }
          }
          q.pcWin = int(all.size());
          pr[PR_PC_WIN] += secSince(tp0);
          const auto tRel0 = std::chrono::steady_clock::now();
          if (pcFp && pcFpTop > 0 && !all.empty()) {
            // v32pc pcfptop: fp3 only for the top N by the REV2 pcrev score, the rest just below them
            std::vector<int> ordF(all.size());
            for (size_t i = 0; i < ordF.size(); ++i) ordF[i] = int(i);
            const size_t nTop = std::min(all.size(), size_t(pcFpTop));
            std::partial_sort(ordF.begin(), ordF.begin() + nTop, ordF.end(),
                              [&](int a, int b) { return all[a].rev != all[b].rev ? all[a].rev > all[b].rev : all[a].k < all[b].k; });
            float lo = 1e9f, loA = 1e9f;
            std::vector<char> done(all.size(), 0);
            for (size_t t = 0; t < nTop; ++t) {
              W& w = all[ordF[t]];
              std::unique_ptr<RDKit::RWMol> m;
              try { m.reset(RDKit::SmilesToMol(w.smi)); } catch (...) { m.reset(); }
              w.rev = m ? fp3Of(*m) : 0.f;
              lo = std::min(lo, w.rev);
              if (qAtt) { w.att = m ? lastAtt : 0.f; loA = std::min(loA, w.att); }
              done[ordF[t]] = 1;
            }
            for (size_t i = 0; i < all.size(); ++i) if (!done[i]) { all[i].rev = lo - 1e-6f; if (qAtt) all[i].att = loA - 1e-6f; }
          }
          if (qAtt && !all.empty()) {
            double mR = 0, sR = 0, mA = 0, sA = 0;
            for (auto& w : all) { mR += w.rev; mA += w.att; }
            mR /= double(all.size()); mA /= double(all.size());
            for (auto& w : all) { sR += (w.rev - mR) * (w.rev - mR); sA += (w.att - mA) * (w.att - mA); }
            sR = std::sqrt(sR / double(all.size())) + 1e-9; sA = std::sqrt(sA / double(all.size())) + 1e-9;
            for (auto& w : all) w.rev = float((w.rev - mR) / sR + attW * (w.att - mA) / sA);
            for (size_t i = 0; i < q.revPc.size() && i < q.attPc.size(); ++i)
              q.revPc[i] = float((q.revPc[i] - mR) / sR + attW * (q.attPc[i] - mA) / sA);
          }
          if (pcFp)
            for (auto& w : all) w.pre = w.rev + pcPP * w.pop;
          // the main list's REV2 scores in the PubChem form (relz, z for the fused PubChem list)
          const std::vector<float>& poolPc = (!pcFp && pcRevMode == poolRevMode) ? q.revLL : q.revPc;
          // stereo-blind keys already in the main list (pool window + generated)
          std::unordered_set<std::string> seen;
          for (auto& kv : poolKeys) seen.insert(kv.first);
          for (auto& g : q.gSmi) seen.insert(g);
          if (pcV == 2 && !all.empty()) {
            double s1 = 0, s2 = 0;
            for (auto& w : all) s1 += w.rev;
            s1 /= double(all.size());
            for (auto& w : all) s2 += (w.rev - s1) * (w.rev - s1);
            q.pcMean = float(s1);
            q.pcSd = float(std::sqrt(s2 / double(all.size())) + 1e-6);
            for (auto& w : all) w.pre = (w.rev - q.pcMean) / q.pcSd + pcW * w.pop;
            // rel = best REV2 score of a PubChem structure NOT in the main list - best pool candidate's
            std::vector<int> ordR(all.size());
            for (size_t i = 0; i < ordR.size(); ++i) ordR[i] = int(i);
            std::sort(ordR.begin(), ordR.end(), [&](int a, int b) { return all[a].rev > all[b].rev; });
            q.pcRevTopNP = all[ordR[0]].rev;
            for (size_t t = 0; t < ordR.size() && t < 5000; ++t) {
              std::unique_ptr<RDKit::RWMol> m;
              try { m.reset(RDKit::SmilesToMol(all[ordR[t]].smi)); } catch (...) { m.reset(); }
              if (!m) continue;
              if (!seen.count(canon(*m))) { q.pcRevTopNP = all[ordR[t]].rev; break; }
            }
            q.pcHavePool = false;
            for (size_t i = 0; i < q.cand.size(); ++i)
              if (q.cand[i] >= 0 && (!q.pcHavePool || poolPc[i] > q.poolRevBest)) { q.poolRevBest = poolPc[i]; q.pcHavePool = true; }
            q.pcRelz = q.pcHavePool ? (q.pcRevTopNP - q.poolRevBest) / q.pcSd : 1e9f;
          }
          std::sort(all.begin(), all.end(), [](const W& a, const W& b) { return a.pre != b.pre ? a.pre > b.pre : a.k < b.k; });
          for (size_t i = 0; i < all.size(); ++i) if (all[i].truth) { q.pcTruthWin = int(i); break; }
          if (opt.count("pcall")) {
            std::ostringstream os;
            for (auto& w : all) os << q.name << "\t" << w.cid << "\t" << w.rev << "\t" << w.pop << "\t" << int(w.truth) << "\n";
            q.pcAll = os.str();
          }
          if (int(all.size()) > pcPre) all.resize(pcPre);
          // stereo-blind dedupe against the main list and each other (seen, above)
          // main-list z statistics: PubChem candidates go on the same scale
          auto msd = [](const std::vector<float>& v, double& m, double& s) {
            m = 0; s = 0;
            for (float x : v) m += x;
            m /= double(std::max<size_t>(1, v.size()));
            for (float x : v) s += (x - m) * (x - m);
            s = std::sqrt(s / double(std::max<size_t>(1, v.size()))) + 1e-6;
          };
          double mp, sp, mm = 0, sm = 1, mrv, srv;
          msd(q.phys, mp, sp);
          if (haveMlp) msd(q.mlpCos, mm, sm);
          msd(poolPc, mrv, srv);
          const bool ownStats = q.cand.size() < 2;
          pr[PR_PC_REL] += secSince(tRel0);
          const auto tFull0 = std::chrono::steady_clock::now();
          std::vector<Query::PcCand> got;
          int pre = 0;
          for (auto& w : all) {
            ++pre;
            if (int(got.size()) >= pcN) break;
            std::unique_ptr<RDKit::RWMol> m;
            try { m.reset(RDKit::SmilesToMol(w.smi)); } catch (...) { m.reset(); }
            if (!m) continue;
            std::string k = canon(*m);
            ++nPcFull;
            if (!seen.insert(k).second) { ++nPcSeenDup; continue; }
            bool isT = !tkey.empty() && k == tkey;
            if (isT && truthInPool) continue;  // the answer's pool entry (canonicalized differently) stays the only copy
            if (isT) { q.pcTruthKept = true; q.pcTruthPre = pre - 1; }
            const bool full = pcV != 2 || pcRankFused;  // pcv=2 rev-only: no physics / MLP / analog features needed
            Query::PcCand pcand{w.cid, w.smi, isT, w.rev, w.pop, full ? physOf(*m) : 0.f, full && haveMlp ? mlpOf(*m) : 0.f, 0.f, 0.f};
            if (haveLib && full) {
              auto cb = ecfp::compute(*m, 2, 2048);
              float ap[3];
              anaOf(cb, anaVar, ap);
              pcand.ana = ap[anaPowI];
            }
            got.push_back(std::move(pcand));
          }
          if (ownStats && !got.empty()) {
            std::vector<float> a, b, c;
            for (auto& g : got) { a.push_back(g.phys); b.push_back(g.mlp); c.push_back(g.rev); }
            msd(a, mp, sp); if (haveMlp) msd(b, mm, sm); msd(c, mrv, srv);
          }
          for (auto& g : got) {
            double zp = (g.phys - mp) / sp, zm = haveMlp ? (g.mlp - mm) / sm : 0.0, zr = (g.rev - mrv) / srv;
            // the hand fusion of the main list (libSim = 0: not a library molecule) + popularity
            g.fused = float(zp + fuseW * zm + fuseR * zr + fuseA * g.ana + pcPop * g.pop);
            g.zp = float(zp); g.zm = float(zm); g.zr = float(zr);
          }
          if (pcV == 2 && !got.empty()) {
            if (pcRankFused) {
              // hand fusion without popularity, z over the listed PubChem candidates, + pcw pop
              std::vector<float> h;
              for (auto& g : got) h.push_back(g.fused - pcPop * g.pop);
              double hm, hs;
              msd(h, hm, hs);
              for (size_t i = 0; i < got.size(); ++i) got[i].fused = float((h[i] - hm) / hs + pcW * got[i].pop);
            } else {
              for (auto& g : got) g.fused = (g.rev - q.pcMean) / q.pcSd + pcW * g.pop;
            }
          }
          std::stable_sort(got.begin(), got.end(), [](auto& a, auto& b) { return a.fused > b.fused; });
          if (pcV == 2 && !got.empty()) {
            q.pcS = (got[0].rev - q.pcMean) / q.pcSd + pcW * got[0].pop;  // the public S: z(fp score) + w pop of the top pick
            q.pcTopPop = got[0].pop;
          }
          q.pc = std::move(got);
          pr[PR_PC_FULL] += secSince(tFull0);
          double tpc = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count();
          std::lock_guard<std::mutex> lk(timeMu);
          tPc += tpc;
        }
        // the truth: its pool entry, else (answer outside the pool window or
        // removed) a generated candidate with the same stereo-blind key
        for (size_t i = 0; i < q.cand.size(); ++i) if (truthPool >= 0 && q.cand[i] == truthPool) q.truth = int(i);
        if (q.truth < 0 && !tkey.empty())
          for (size_t i = 0; i < q.cand.size(); ++i)
            if (q.cand[i] < 0 && q.gSmi[-q.cand[i] - 1] == tkey) { q.truth = int(i); break; }
        if (tfmNM) {
          // per query: logit(p) and sum p for each probability vector
          std::vector<std::vector<float>> U(tfmNP);
          std::vector<double> sumP(tfmNP, 0.0);
          for (int k = 0; k < tfmNP; ++k) {
            const auto& p = q.tP[k];
            U[k].resize(p.size());
            for (size_t j = 0; j < p.size(); ++j) {
              float pc = std::min(1.f - 1e-6f, std::max(1e-6f, p[j]));
              U[k][j] = std::log(pc / (1.f - pc));
              sumP[k] += p[j];
            }
          }
          q.tcol.assign(q.cand.size(), std::vector<float>(tfmNC, 0.f));
          for (size_t ci = 0; ci < cl.size(); ++ci) {
            std::vector<uint16_t> bits = tfmVocab.bits(*cl[ci].mol);
            for (int k = 0; k < tfmNP; ++k) {
              double sl = 0, inter = 0;
              for (uint16_t b : bits) { sl += U[k][b]; inter += q.tP[k][b]; }
              q.tcol[ci][2 * k] = float(sl);
              q.tcol[ci][2 * k + 1] = float(inter / (double(bits.size()) + sumP[k] - inter + 1e-9));
            }
          }
          std::vector<std::vector<float>>().swap(q.tP);  // free (4096 floats per vector)
        }
        if (qi % 100 == 0) std::printf("  enumerated %zu / %zu\n", qi, Q.size());
      }
    });
  for (auto& t : th) t.join();
  const auto tEnumEnd = std::chrono::steady_clock::now();
  {
    double tot = 0;
    for (int k = 0; k < PR_N; ++k) std::printf("v38 profile: %-40s %9.1f thread-s\n", PR_NAME[k], prof[k]);
    for (int k : {PR_ANA, PR_ATT, PR_POOLPARSE, PR_GEN, PR_S_PHYS, PR_S_MLP, PR_S_BITS, PR_S_LIB, PR_PC_WIN, PR_PC_REL, PR_PC_FULL}) tot += prof[k];
    std::printf("v38 profile: enumeration wall %.1f s (%u threads), timed stages %.1f thread-s; setup before enumeration %.1f s\n",
                std::chrono::duration<double>(tEnumEnd - tEnum0).count(), nThr, tot,
                std::chrono::duration<double>(tEnum0 - tMain0).count());
    std::printf("v38 redundancy: pool lookups %ld, distinct %zu (hits %ld); pubchem window lookups %ld, distinct %zu (hits %ld); "
                "pubchem full-scoring candidates %ld, of which stereo-blind duplicates of the main list / each other %ld; "
                "fp3c vocabulary features per parsed PubChem molecule %.1f\n",
                long(nPoolLook), poolCache.size(), long(poolCache.hits), long(nPcLook), pcCache.size(), long(pcCache.hits),
                long(nPcFull), long(nPcSeenDup), double(nPcFeatSum) / std::max(1L, long(nPcFeatMol)));
  }
  if (haveLib)
    std::printf("analog search: %zu query spectra, stage1 %.1f thread-s, total (incl. stage2 + fingerprints) %.1f thread-s"
                " = %.1f ms/spectrum; enumeration wall %.1f s; K=%d\n",
                nAnaSpec, tStage1, tAnaTotal, 1000.0 * tAnaTotal / std::max<size_t>(1, nAnaSpec),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - tEnum0).count(), shortK);
  size_t withTruth = 0, totC = 0;
  for (auto& q : Q) { withTruth += q.truth >= 0; totC += q.cand.size(); }
  std::printf("%zu queries, %.1f candidates each, truth present in %zu\n", Q.size(),
              double(totC) / Q.size(), withTruth);
  if (doGen || rmTruth) {
    size_t nG = 0, nRaw = 0, tRaw = 0, tKept = 0, tGenTruth = 0, qWithGen = 0;
    std::vector<int> per;
    for (auto& q : Q) {
      nG += q.gSmi.size(); nRaw += q.nGenRaw; tRaw += q.truthGenRaw; tKept += q.truthGenKept;
      per.push_back(int(q.gSmi.size())); qWithGen += !q.gSmi.empty();
      if (q.truth >= 0 && q.cand[q.truth] < 0) ++tGenTruth;
    }
    std::sort(per.begin(), per.end());
    std::printf("generation: %.1f thread-s (%.1f ms/query); distinct products %.1f/query; kept %.1f/query (median %d, "
                "queries with any %zu); truth among products %zu (%.3f), kept %zu (%.3f); truth = generated candidate in %zu; "
                "dropped: answer's stereo-blind duplicates in pool %ld, generated copies of an in-pool answer %ld, outside window %ld\n",
                tGen, 1000.0 * tGen / std::max<size_t>(1, Q.size()), double(nRaw) / Q.size(), double(nG) / Q.size(),
                per.empty() ? 0 : per[per.size() / 2], qWithGen, tRaw, double(tRaw) / Q.size(), tKept,
                double(tKept) / Q.size(), tGenTruth, long(nRmDup), long(nTruthDup), long(nOutWin));
    if (!detPath.empty()) {
      std::ofstream df(dir + "/" + detPath);
      for (auto& q : Q) df << q.det;
    }
  }

  if (doPc) {
    size_t nWin = 0, tIn = 0, tKept = 0, nFull = 0, tTop[5] = {0, 0, 0, 0, 0};
    std::vector<int> wins;
    for (auto& q : Q) {
      nWin += q.pcWin; wins.push_back(q.pcWin); tIn += q.pcTruthInWin; tKept += q.pcTruthKept; nFull += q.pc.size();
      if (q.pcTruthWin >= 0) {
        int r = q.pcTruthWin;
        tTop[0] += r < 25; tTop[1] += r < 100; tTop[2] += r < 300; tTop[3] += r < 1000; tTop[4] += 1;
      }
    }
    std::sort(wins.begin(), wins.end());
    std::printf("pubchem: %.1f thread-s (%.1f ms/query); window %.0f/query (median %d, max %d); truth in window %zu (%.3f)"
                " [rank by rev+pop: <25 %zu, <100 %zu, <300 %zu, <1000 %zu]; truth among fully scored %zu (%.3f); full %.1f/query\n",
                tPc, 1000.0 * tPc / std::max<size_t>(1, Q.size()), double(nWin) / Q.size(), wins.empty() ? 0 : wins[wins.size() / 2],
                wins.empty() ? 0 : wins.back(), tIn, double(tIn) / Q.size(), tTop[0], tTop[1], tTop[2], tTop[3], tKept,
                double(tKept) / Q.size(), double(nFull) / Q.size());
    if (opt.count("pcall")) {
      std::ofstream f(dir + "/" + opt["pcall"]);
      for (auto& q : Q) f << q.pcAll;
    }
  }

  // MRR@25 over all queries (truth absent = 0), with ties broken pessimistically
  auto evaluate = [&](const Params& P, std::vector<std::vector<float>>* scoresOut) {
    std::vector<double> rr(Q.size(), 0.0);
    if (scoresOut) scoresOut->assign(Q.size(), {});
    std::atomic<size_t> nx(0);
    std::vector<std::thread> tt;
    for (unsigned t = 0; t < nThr; ++t)
      tt.emplace_back([&]() {
        std::vector<std::pair<double, double>> pred;
        for (;;) {
          size_t qi = nx++;
          if (qi >= Q.size()) break;
          const Query& q = Q[qi];
          std::vector<float> sc(q.cand.size(), 0.f);
          float eps = std::exp(P.logEps);
          if (!keepFrags) sc = q.phys;  // eval: scored with the loaded params during enumeration
          for (size_t c = 0; c < q.cand.size() && keepFrags; ++c) {
            for (int s : q.spectra) {
              const Head& h = head[s];
              predict(q.frags[c], P, adductIdx(h.tag), q.candN[c], pred);
              sc[c] += loglik(pred, mz.data() + h.off, it.data() + h.off, h.n, 0.01f, eps, h.precursor);
            }
            sc[c] /= float(q.spectra.size());
          }
          if (q.truth >= 0) {
            int better = 0;
            for (size_t c = 0; c < sc.size(); ++c)
              if (int(c) != q.truth && sc[c] >= sc[q.truth]) ++better;
            if (better < 25) rr[qi] = 1.0 / (better + 1);
          }
          if (scoresOut) (*scoresOut)[qi] = std::move(sc);
        }
      });
    for (auto& t : tt) t.join();
    double s = 0;
    for (double x : rr) s += x;
    return s / Q.size();
  };

  Params P;
  if (mode == "fit") {
    // fit <...> <params_out> [params_init]: start from an earlier fit
    if (!outPath.empty() && P.load(dir + "/" + outPath)) std::printf("start from %s\n", outPath.c_str());
    double best = evaluate(P, nullptr);
    std::printf("initial MRR %.4f\n", best);
    // coordinate search: each parameter tried at +-step, step halves when a
    // full sweep finds nothing; E[INTACT] and the k weights included.
    float step = 1.0f;
    for (int sweep = 0; sweep < 12 && step > 0.05f; ++sweep) {
      bool improved = false;
      for (int i = 0; i < Params::N; ++i) {
        float s = (i == NCLS - 1 || i == NCLS) ? step * 0.1f : step;  // betas live on a smaller scale
        for (float d : {s, -s}) {
          Params T = P;
          *T.at(i) += d;
          if ((i == NCLS - 1 || i == NCLS) && *T.at(i) < 0.f) continue;
          double v = evaluate(T, nullptr);
          if (v > best + 1e-5) { best = v; P = T; improved = true; std::printf("  p%d -> %.3f  MRR %.4f\n", i, *P.at(i), best); break; }
        }
      }
      std::printf("sweep %d step %.3f MRR %.4f\n", sweep, step, best);
      std::fflush(stdout);
      if (!improved) step *= 0.5f;
    }
    P.save(dir + "/" + paramPath);
    std::printf("saved %s\n", paramPath.c_str());
    // ablations at the fitted point
    Params A = P; A.betaPos = 0.f; A.betaNeg = 0.f;
    std::printf("ablation: no charge retention %.4f\n", evaluate(A, nullptr));
    A = P; for (int i = 1; i < NCLS; ++i) A.E[i] = 1.0f;
    std::printf("ablation: flat bond classes %.4f\n", evaluate(A, nullptr));
    A = P; A.dofSlope = 0.f;
    std::printf("ablation: no size effect (dofSlope 0) %.4f\n", evaluate(A, nullptr));
    A = P; A.E[CGLYC] = A.E[ARYLC];
    std::printf("ablation: C-glycoside = aryl C-C %.4f\n", evaluate(A, nullptr));
    int nTruthCg = 0;
    for (auto& q : Q)
      if (q.truth >= 0)
        for (auto& fr : q.frags[q.truth])
          if (fr.cls == CGLYC) { ++nTruthCg; break; }
    std::printf("true structures with a C-glycosidic bond: %d / %zu\n", nTruthCg, Q.size());
    std::printf("fitted:");
    const char* cn[NCLS] = {"intact", "acyl", "glycosidic", "C-het", "aryl-C", "C-C", "ring", "ring-ar", "other", "C-glycosidic"};
    for (int i = 0; i < NCLS; ++i) std::printf(" %s=%.3f", cn[i], P.E[i]);
    std::printf(" E2=%.3f dofSlope=%.3f\n", P.E2, P.dofSlope);
    return 0;
  }

  if (!P.load(dir + "/" + paramPath)) std::printf("(params %s absent: defaults)\n", paramPath.c_str());
  std::vector<std::vector<float>> scores;
  double m = evaluate(P, &scores);
  auto inLibOf = [&](int c) { return haveLib && c >= 0 && libSpecOf.count(pname[c]) > 0; };
  // generated candidates: G<k> (the truth's own name when it is the answer, so mrr.py scores the CSV)
  auto candName = [&](const Query& q, size_t i) -> std::string {
    int c = q.cand[i];
    if (c >= 0) return pname[c];
    if (int(i) == q.truth) return q.name;
    return "G" + std::to_string(-c - 1);
  };
  auto candSmiles = [&](const Query& q, size_t i) -> std::string {
    int c = q.cand[i];
    return c >= 0 ? psmi[c] : q.gSmi[-c - 1];
  };
  // TFM fusion column, z-scored per query (tshuf=1: permuted within the query)
  std::vector<std::vector<float>> tfmZ(Q.size());
  if (tfmNM && tfmV >= 0 && tfmV < tfmNC) {
    std::mt19937 rng(12345);
    for (size_t qi = 0; qi < Q.size(); ++qi) {
      std::vector<float> v(Q[qi].cand.size());
      for (size_t i = 0; i < v.size(); ++i) v[i] = Q[qi].tcol[i][tfmV];
      if (tfmShuf) std::shuffle(v.begin(), v.end(), rng);
      tfmZ[qi] = zscore(v);
    }
  }
  auto tfmTerm = [&](size_t qi, size_t i) { return tfmZ[qi].empty() ? 0.f : tfmW * tfmZ[qi][i]; };
  auto genTerm = [&](const Query& q, size_t i, const std::vector<double>& g) {
    double t = 0;
    for (int k = 0; k < NG; ++k) t += g[k] * q.gF[i][k];
    return float(t);
  };
  // [features.txt]: per-candidate features for fuse_fit (learned weights).
  // "Q <name> <truth index or -1> <n>" then n lines
  // "phys mlp rev libSim anaP inLib name" (raw values; fuse_fit normalizes).
  std::string featPath = argc > 17 ? argv[17] : "";
  if (!featPath.empty()) {
    std::ofstream ff(dir + "/" + featPath);
    for (size_t qi = 0; qi < Q.size(); ++qi) {
      const Query& q = Q[qi];
      ff << "Q " << q.name << " " << q.truth << " " << q.cand.size() << "\n";
      for (size_t i = 0; i < q.cand.size(); ++i) {
        ff << scores[qi][i] << " " << (haveMlp ? q.mlpCos[i] : 0.f) << " "
           << (haveRev ? q.revLL[i] : 0.f) << " " << (haveLib ? q.libSim[i] : 0.f) << " "
           << (haveLib ? q.anaP[i] : 0.f) << " "
           // "has reference spectra in THIS library", not the pool's train
           // flag: a held-out answer is a train structure whose spectra were
           // removed, so the pool flag marks it (1, with library sim 0) - a
           // signature the hidden test does not share. Fitted on that flag
           // the weights learned the holdout's construction (cqry 0.48 -> 0.77).
           << int(inLibOf(q.cand[i])) << " " << (q.cand[i] >= 0 ? pname[q.cand[i]] : "GEN" + std::to_string(-q.cand[i] - 1))
           << " " << q.gF[i][0] << " " << q.gF[i][1] << " " << q.gF[i][2] << " " << q.gF[i][3] << " " << q.gF[i][4]
           << " " << q.gF[i][5];
        for (int k = 0; k < tfmNC; ++k) ff << " " << q.tcol[i][k];
        ff << " " << candSmiles(q, i) << "\n";
      }
    }
    std::printf("wrote features %s\n", featPath.c_str());
  }
  std::printf("MRR@25 %.4f\n", m);
  if (keepFrags) {
    Params A = P; A.betaPos = 0.f; A.betaNeg = 0.f;
    std::printf("ablation: no charge retention %.4f\n", evaluate(A, nullptr));
  }
  if (haveMlp || haveRev || haveLib) {
    // z(phys) + w z(mlp) + wr z(rev) + wl libSim; w < 0 means the MLP alone
    auto fused = [&](float w, float wr, float wl, std::vector<std::vector<float>>* out) {
      double tot = 0;
      // results go to a separate buffer: `out` is the caller's `scores`,
      // which this loop also READS as the physics term - clearing it first
      // (as v16-v19 did) left the output sorted on freed memory
      std::vector<std::vector<float>> res(out ? Q.size() : 0);
      for (size_t qi = 0; qi < Q.size(); ++qi) {
        auto zp = zscore(scores[qi]);
        std::vector<float> zm = haveMlp ? zscore(Q[qi].mlpCos) : std::vector<float>(zp.size(), 0.f);
        std::vector<float> zr = haveRev ? zscore(Q[qi].revLL) : std::vector<float>(zp.size(), 0.f);
        std::vector<float> f(zp.size());
        // v34pc attpool: attention score of the pool candidates in the main fusion
        const bool qAp = attPoolW != 0.f && Q[qi].attPc.size() == zp.size();
        std::vector<float> za = qAp ? zscore(Q[qi].attPc) : std::vector<float>();
        const bool qCp = cnnPoolW != 0.f && Q[qi].cnnPc.size() == zp.size();
        std::vector<float> zc = qCp ? zscore(Q[qi].cnnPc) : std::vector<float>();
        const bool qFp = ftPoolW != 0.f && Q[qi].ftPc.size() == zp.size();  // v37pc ftpool
        std::vector<float> zf = qFp ? zscore(Q[qi].ftPc) : std::vector<float>();
        for (size_t i = 0; i < f.size(); ++i) {
          f[i] = w < 0.f ? zm[i] : zp[i] + w * zm[i] + wr * zr[i];
          if (qAp) f[i] += attPoolW * za[i];
          if (qCp) f[i] += cnnPoolW * zc[i];
          if (qFp) f[i] += ftPoolW * zf[i];
          // a sharp power: only a near-identical library spectrum (the same
          // molecule) should count - a linear term promotes every analog
          // whose spectrum is merely similar, which costs the de novo case
          if (haveLib) f[i] += wl * std::pow(Q[qi].libSim[i], libPow) + anaW * Q[qi].anaP[i];
          f[i] += genTerm(Q[qi], i, gw) + tfmTerm(qi, i);
        }
        int t = Q[qi].truth;
        if (t >= 0) {
          int better = 0;
          for (size_t i = 0; i < f.size(); ++i) if (int(i) != t && f[i] >= f[t]) ++better;
          if (better < 25) tot += 1.0 / (better + 1);
        }
        if (out) res[qi] = f;
      }
      if (out) *out = std::move(res);
      return tot / Q.size();
    };
    if (doGrid)
    for (float wr : {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 5.f}) {
      if (!haveRev && wr > 0.f) break;
      for (float w : {-1.f, 0.f, 0.25f, 0.5f, 1.f}) {
        if (!haveMlp && w != 0.f) continue;
        std::printf("fusion w=%-5g wr=%-5g MRR %.4f\n", w, wr, fused(w, wr, 0.f, nullptr));
      }
    }
    if (haveLib && doGrid) {
      libPow = 32.f;
      for (float wa : {0.f, 1.f, 2.f, 4.f, 8.f, 16.f}) {
        anaW = wa;
        for (float wl : {0.f, 8.f})
          std::printf("fusion w=0.25  wr=1  pow=32 wl=%-3g wa=%-4g MRR %.4f\n", wl, wa, fused(0.25f, 1.f, wl, nullptr));
      }
      // analog variants: one line per (variant, pow, wl), MRR over the wa grid
      const float WA[] = {0.f, 4.f, 8.f, 12.f, 16.f, 24.f, 32.f, 48.f};
      std::printf("ana grid (w=0.25 wr=1 libpow=32) wa:");
      for (float wa : WA) std::printf(" %6g", wa);
      std::printf("\n");
      for (int vi = 0; vi < NVAR; ++vi)
        for (int pi = 0; pi < 3; ++pi)
          for (float wl : {0.f, 8.f}) {
            for (auto& q : Q) q.anaP.swap(q.anaV[vi][pi]);
            std::printf("ana %-12s p=%d wl=%-2g", VARNAME[vi], pi + 2, wl);
            for (float wa : WA) { anaW = wa; std::printf(" %.4f", fused(0.25f, 1.f, wl, nullptr)); }
            std::printf("\n");
            for (auto& q : Q) q.anaP.swap(q.anaV[vi][pi]);
          }
      std::fflush(stdout);
    }
    libPow = fuseLPow;
    anaW = fuseA;
    if (!weightsPath.empty()) {
      // learned weights (fuse_fit, conditional logit) over the same features
      // fuse_fit reads: z(phys) z(mlp) z(rev) lib^32 lib^8 lib analog inLib
      std::ifstream wf(dir + "/" + weightsPath);
      double b[8] = {0};
      for (double& x : b) wf >> x;
      // optional: NG generation-feature weights after the 8 base weights
      {
        std::vector<double> g(NG, 0.0);
        int k = 0;
        while (k < NG && (wf >> g[k])) ++k;
        if (k == NG) { gw = g; std::printf("learned generation weights read (%d)\n", k); }
      }
      double tot = 0;
      std::vector<std::vector<float>> res(Q.size());
      for (size_t qi = 0; qi < Q.size(); ++qi) {
        const Query& q = Q[qi];
        auto zp = zscore(scores[qi]);
        std::vector<float> zm = haveMlp ? zscore(q.mlpCos) : std::vector<float>(zp.size(), 0.f);
        std::vector<float> zr = haveRev ? zscore(q.revLL) : std::vector<float>(zp.size(), 0.f);
        res[qi].resize(zp.size());
        for (size_t i = 0; i < zp.size(); ++i) {
          float l = haveLib ? std::max(0.f, q.libSim[i]) : 0.f;
          float inLib = inLibOf(q.cand[i]) ? 1.f : 0.f;
          res[qi][i] = float(b[0] * zp[i] + b[1] * zm[i] + b[2] * zr[i] + b[3] * std::pow(l, 32.f) +
                             b[4] * std::pow(l, 8.f) + b[5] * l + b[6] * (haveLib ? q.anaP[i] : 0.f) +
                             b[7] * inLib) + genTerm(q, i, gw) + tfmTerm(qi, i);
        }
        if (q.truth >= 0) {
          int better = 0;
          for (size_t i = 0; i < res[qi].size(); ++i)
            if (int(i) != q.truth && res[qi][i] >= res[qi][q.truth]) ++better;
          if (better < 25) tot += 1.0 / (better + 1);
        }
      }
      scores = std::move(res);
      std::printf("output uses learned weights %s: MRR@25 %.4f\n", weightsPath.c_str(), tot / Q.size());
    } else if (fuseW > 0.f || fuseR > 0.f || fuseL > 0.f || fuseA > 0.f) {
      fused(fuseW, fuseR, fuseL, &scores);
      std::printf("output uses fusion w=%g wr=%g wl=%g\n", fuseW, fuseR, fuseL);
    }
  }
  // score-mode insertion compares PubChem picks (which have no FPNet score)
  // with the main list's scores BEFORE the FPNet term
  const std::vector<std::vector<float>> scoresNoFp = scores;
  if (!fpPath.empty() && fuseF != 0.f) {
    std::unordered_map<std::string, std::unordered_map<std::string, float>> fps;
    {
      std::ifstream f(dir + "/" + fpPath);
      if (!f) { std::fprintf(stderr, "cannot read %s\n", (dir + "/" + fpPath).c_str()); return 1; }
      std::string line;
      size_t nl = 0;
      while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t t1 = line.find('\t'); if (t1 == std::string::npos) continue;
        size_t t2 = line.find('\t', t1 + 1); if (t2 == std::string::npos) continue;
        float v = float(std::atof(line.c_str() + t2 + 1));
        auto& m = fps[line.substr(0, t1)];
        auto ins = m.emplace(line.substr(t1 + 1, t2 - t1 - 1), v);
        if (!ins.second) ins.first->second = std::max(ins.first->second, v);
        ++nl;
      }
      std::printf("fpnet scores: %zu lines, %zu queries\n", nl, fps.size());
    }
    double tot = 0;
    size_t nHave = 0, nMiss = 0, nQ0 = 0;
    for (size_t qi = 0; qi < Q.size(); ++qi) {
      const Query& q = Q[qi];
      if (scores[qi].size() != q.cand.size()) continue;
      auto fq = fps.find(q.name);
      std::vector<float> v(q.cand.size());
      std::vector<char> have(q.cand.size(), 0);
      double m = 0; int k = 0;
      if (fq != fps.end())
        for (size_t i = 0; i < q.cand.size(); ++i) {
          if (q.cand[i] < 0) continue;  // generated: no FPNet score (query mean)
          auto c = fq->second.find(pname[q.cand[i]]);
          if (c != fq->second.end()) { v[i] = c->second; have[i] = 1; m += c->second; ++k; }
        }
      nHave += k; nMiss += q.cand.size() - k;
      if (k == 0) { ++nQ0; continue; }
      m /= k;
      for (size_t i = 0; i < v.size(); ++i) if (!have[i]) v[i] = float(m);
      auto zf = zscore(v);
      for (size_t i = 0; i < v.size(); ++i) scores[qi][i] += fuseF * zf[i];
      if (q.truth >= 0) {
        int better = 0;
        for (size_t i = 0; i < v.size(); ++i)
          if (int(i) != q.truth && scores[qi][i] >= scores[qi][q.truth]) ++better;
        if (better < 25) tot += 1.0 / (better + 1);
      }
    }
    std::printf("output + %g z(fpnet): candidates with score %zu, without %zu, queries without any %zu, MRR@25 %.4f\n",
                fuseF, nHave, nMiss, nQ0, tot / std::max<size_t>(1, Q.size()));
  }
  // final order per query (+ the PubChem slot insertion)
  std::vector<std::vector<std::string>> plain(Q.size()), inserted(Q.size());
  size_t nGated = 0, nPpMoved = 0, nProm = 0, nHi = 0;
  // ppdet=file: per query the final order's top max(ppN,25) before the pool-popularity step
  // ("name rank score isPool pop candname") - offline grids over ppop / PubChem options
  std::ofstream ppdet;
  if (opt.count("ppdet")) ppdet.open(dir + "/" + opt["ppdet"]);
  // v36pc cfdump=file: every candidate "name idx score isPool pop candname cnn att fp3" (offline pool-fusion grids)
  std::ofstream cfdump;
  if (opt.count("cfdump")) cfdump.open(dir + "/" + opt["cfdump"]);
  for (size_t qi = 0; qi < Q.size(); ++qi) {
    Query& q = Q[qi];
    std::vector<int> o(q.cand.size());
    for (size_t i = 0; i < o.size(); ++i) o[i] = int(i);
    std::sort(o.begin(), o.end(), [&](int a, int b) { return scores[qi][a] > scores[qi][b]; });
    std::vector<float> sTop;  // v33pc alloc=boltz: the final scores by rank, before the pool-popularity re-fill
    if (pcBoltz && scores[qi].size() == q.cand.size())
      for (size_t r = 0; r < o.size() && int(r) < bK1; ++r) sTop.push_back(scores[qi][o[r]]);
    if (scores[qi].size() == q.cand.size()) {
      q.top1 = o.size() > 0 ? scores[qi][o[0]] : 0.f;
      q.top2 = o.size() > 1 ? scores[qi][o[1]] : q.top1 - 1e3f;
    }
    if (cfdump.is_open() && scores[qi].size() == q.cand.size())
      for (size_t i = 0; i < q.cand.size(); ++i) {
        int c = q.cand[i];
        cfdump << q.name << '\t' << i << '\t' << scores[qi][i] << '\t' << int(c >= 0) << '\t'
               << (c >= 0 && size_t(c) < poolPop.size() ? poolPop[c] : 0.f) << '\t' << candName(q, i) << '\t'
               << (i < q.cnnPc.size() ? q.cnnPc[i] : 0.f) << '\t' << (i < q.attPc.size() ? q.attPc[i] : 0.f) << '\t'
               << (i < q.revPc.size() ? q.revPc[i] : 0.f);
        if (ftOn) cfdump << '\t' << (i < q.ftPc.size() ? q.ftPc[i] : 0.f);  // v37pc: 10th column ft (only with ftmodel)
        cfdump << '\n';
      }
    if (ppdet.is_open() && scores[qi].size() == q.cand.size())
      for (size_t r = 0; r < o.size() && int(r) < std::max(ppN, 25); ++r) {
        int c = q.cand[o[r]];
        ppdet << q.name << "\t" << r << "\t" << scores[qi][o[r]] << "\t" << int(c >= 0) << "\t"
              << (c >= 0 && size_t(c) < poolPop.size() ? poolPop[c] : 0.f) << "\t" << candName(q, o[r]) << "\n";
      }
    // pool popularity: re-fill the pool candidates' places within the top ppN
    float libBest = 0.f;
    for (float l : q.libSim) libBest = std::max(libBest, l);
    if (ppW != 0.f && !poolPop.empty() && scores[qi].size() == q.cand.size() && libBest < ppGate) {
      std::vector<int> posP, idxP;
      for (size_t r = 0; r < o.size() && int(r) < ppN; ++r)
        if (q.cand[o[r]] >= 0) { posP.push_back(int(r)); idxP.push_back(o[r]); }
      if (idxP.size() > 1) {
        double m = 0, s = 0;
        for (int i : idxP) m += scores[qi][i];
        m /= double(idxP.size());
        for (int i : idxP) s += (scores[qi][i] - m) * (scores[qi][i] - m);
        s = std::sqrt(s / double(idxP.size())) + 1e-6;
        std::vector<std::pair<double, int>> key;
        for (int i : idxP) {
          int c = q.cand[i];
          float pp = size_t(c) < poolPop.size() ? poolPop[c] : 0.f;
          key.emplace_back((scores[qi][i] - m) / s + ppW * pp, i);
        }
        std::stable_sort(key.begin(), key.end(), [](auto& a, auto& b) { return a.first > b.first; });
        for (size_t j = 0; j < posP.size(); ++j) {
          if (o[posP[j]] != key[j].second) ++nPpMoved;
          o[posP[j]] = key[j].second;
        }
      }
    }
    for (size_t i = 0; i < o.size() && i < 25; ++i) {
      plain[qi].push_back(writeSmiles ? candSmiles(q, o[i]) : candName(q, o[i]));
      if (scoresNoFp[qi].size() == q.cand.size()) q.topS.push_back(scoresNoFp[qi][o[i]]);
    }
    if (!doPc) continue;
    if (pcV == 2) {
      // gate, fixed slots (dense when relz > relhi), promotion of the top pick
      auto& L = inserted[qi];
      if (!(q.libMax < pcGate) || q.pc.empty()) { L = plain[qi]; continue; }
      ++nGated;
      if (pcBoltz) {
        if (sTop.empty()) { L = plain[qi]; continue; }
        auto lse = [](const std::vector<double>& v) {
          double m = v[0];
          for (double x : v) m = std::max(m, x);
          double t = 0;
          for (double x : v) t += std::exp(x - m);
          return m + std::log(t);
        };
        std::vector<double> e1, e2;
        for (float x : sTop) e1.push_back(double(x) / bT1);
        for (auto& c : q.pc) e2.push_back(double(c.fused) / bT2);
        const double l1 = lse(e1), l2 = lse(e2);
        const double r = std::fabs(q.pcRelz) < 1e8f ? double(q.pcRelz) : 0.0;
        const double lp = std::log(bPi / (1.0 - bPi)) + bA * r;
        std::vector<std::pair<double, int>> it;  // (log prob, index: >= 0 main rank, < 0 PubChem -1-j)
        const size_t nM = std::min(plain[qi].size(), e1.size());
        for (size_t i = 0; i < nM; ++i) it.emplace_back(e1[i] - l1, int(i));
        for (size_t j = 0; j < e2.size(); ++j) it.emplace_back(e2[j] - l2 + lp, -1 - int(j));
        std::stable_sort(it.begin(), it.end(), [](auto& x, auto& y) { return x.first > y.first; });
        for (auto& [lpv, k] : it) {
          if (L.size() >= 25) break;
          if (k >= 0) L.push_back(plain[qi][size_t(k)]);
          else {
            const auto& c = q.pc[size_t(-1 - k)];
            L.push_back(writeSmiles ? c.smi : (c.truth ? q.name : "P" + std::to_string(c.cid)));
          }
        }
        continue;
      }
      const bool hi = q.pcRelz > pcRelHi;
      nHi += hi;
      const std::vector<int>& slv = hi ? pcSlotsHi : pcSlotsLo;
      std::set<int> sl(slv.begin(), slv.end());
      std::vector<std::string> pick;
      for (size_t i = 0; i < q.pc.size() && pick.size() < slv.size(); ++i)
        pick.push_back(writeSmiles ? q.pc[i].smi : (q.pc[i].truth ? q.name : "P" + std::to_string(q.pc[i].cid)));
      size_t a = 0, b = 0;
      for (int pos = 1; pos <= 25; ++pos) {
        if (sl.count(pos) && b < pick.size()) L.push_back(pick[b++]);
        else if (a < plain[qi].size()) L.push_back(plain[qi][a++]);
        else if (b < pick.size()) L.push_back(pick[b++]);
        else break;
      }
      const bool prom = !pcNoProm && (q.pcRelz > pcRelP || (q.pcS > pcSP && q.pcTopPop >= pcPopP));
      if (prom && !pick.empty()) {
        ++nProm;
        auto it = std::find(L.begin(), L.end(), pick[0]);
        if (it == L.end()) { L.insert(L.begin(), pick[0]); if (L.size() > 25) L.pop_back(); }
        else std::rotate(L.begin(), it, it + 1);
      }
      continue;
    }
    const bool open =q.libMax < pcGate && (o.size() < 2 || q.top1 - q.top2 < pcGMarg);
    std::vector<std::string> pick;
    std::vector<float> pickS;  // score-mode: the pick's score on the main list's scale
    if (open) {
      ++nGated;
      for (size_t i = 0; i < q.pc.size() && int(pick.size()) < pcK; ++i) {
        pick.push_back(writeSmiles ? q.pc[i].smi : (q.pc[i].truth ? q.name : "P" + std::to_string(q.pc[i].cid)));
        pickS.push_back(q.pc[i].fused - pcPop * q.pc[i].pop + pcSW * q.pc[i].pop + pcBias);
      }
    }
    std::set<int> sl(pcSlots.begin(), pcSlots.end());
    size_t a = 0, b = 0;
    auto& L = inserted[qi];
    for (int pos = 1; pos <= 25; ++pos) {
      bool take;
      if (pcScoreMode)  // merge by score, never above slot pcMin
        take = b < pick.size() && pos >= pcMin && (a >= q.topS.size() || pickS[b] > q.topS[a]);
      else
        take = sl.count(pos) && b < pick.size();
      if (take) L.push_back(pick[b++]);
      else if (a < plain[qi].size()) L.push_back(plain[qi][a++]);
      else if (b < pick.size() && !pcScoreMode) L.push_back(pick[b++]);
      else break;
    }
  }
  auto writeCsv = [&](const std::string& path, const std::vector<std::vector<std::string>>& lists) {
    std::ofstream out(dir + "/" + path);
    out << "molecule_id,smiles\n";
    for (size_t qi = 0; qi < Q.size(); ++qi) {
      out << Q[qi].name << ",";
      for (size_t i = 0; i < lists[qi].size(); ++i) out << (i ? ";" : "") << lists[qi][i];
      out << "\n";
    }
  };
  if (ppW != 0.f && !poolPop.empty())
    std::printf("pool popularity: weight %g, top %d re-ordered, %zu list places changed over %zu queries\n", ppW, ppN, nPpMoved, Q.size());
  // production (pc=1 without pcout): out.csv IS the inserted list
  if (!outPath.empty()) writeCsv(outPath, doPc && !opt.count("pcout") ? inserted : plain);
  if (doPc) {
    double mPlain = 0, mIns = 0;
    for (size_t qi = 0; qi < Q.size(); ++qi) {
      for (size_t i = 0; i < plain[qi].size(); ++i) if (plain[qi][i] == Q[qi].name) { mPlain += 1.0 / double(i + 1); break; }
      for (size_t i = 0; i < inserted[qi].size(); ++i) if (inserted[qi][i] == Q[qi].name) { mIns += 1.0 / double(i + 1); break; }
    }
    std::printf("pubchem slots: gate open for %zu / %zu queries; MRR@25 (eval names) plain %.4f -> inserted %.4f\n", nGated,
                Q.size(), mPlain / double(Q.size()), mIns / double(Q.size()));
    if (pcV == 2) std::printf("pubchem v2: dense slots %zu, promoted to rank 1 %zu / %zu queries\n", nHi, nProm, Q.size());
    if (opt.count("pcout")) writeCsv(opt["pcout"], inserted);
    if (opt.count("pcdet")) {
      std::ofstream f(dir + "/" + opt["pcdet"]);
      for (auto& q : Q) {
        f << "Q\t" << q.name << "\t" << q.libMax << "\t" << q.top1 << "\t" << q.top2 << "\t" << q.anaTop << "\t"
          << q.cand.size() << "\t" << q.truth << "\t" << q.pcWin << "\t" << int(q.pcTruthInWin) << "\t" << q.pcTruthWin
          << "\t" << q.pc.size();
        for (float s : q.topS) f << "\t" << s;
        f << "\n";
        if (pcV == 2)
          f << "V2\t" << q.pcMean << "\t" << q.pcSd << "\t" << q.pcRelz << "\t" << q.pcS << "\t" << q.pcTopPop << "\t"
            << q.pcRevTopNP << "\t" << q.poolRevBest << "\t" << int(q.pcHavePool) << "\n";
        for (auto& c : q.pc)
          f << c.cid << "\t" << int(c.truth) << "\t" << c.rev << "\t" << c.pop << "\t" << c.phys << "\t" << c.mlp << "\t"
            << c.ana << "\t" << c.fused << "\t" << c.zp << "\t" << c.zm << "\t" << c.zr << "\t" << c.smi << "\n";
      }
    }
  }
  std::printf("v38 profile: after enumeration (fusion, merge, output) %.1f s; total %.1f s\n", secSince(tEnumEnd), secSince(tMain0));
  return 0;
}
