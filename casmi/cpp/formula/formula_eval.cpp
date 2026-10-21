// formula_eval: "output side" of the subformula design (formula_PLAN.md) on the window yardstick.
//
// For each query and each of its top-N PubChem-window candidates by the current best window
// score (fp3 + attention; win_dump.cpp), the physics model (phys_core.h = phys_size / phys_v33pc
// enumerate(): intact, every non-ring single cut, pairs of labile cuts, two-bond ring openings;
// H shifts k = -2..2; class barriers, proton-affinity share and k weights from phys_size_fqry.txt)
// lists the candidate's fragment ions. Every fragment ion is a sub-formula of the candidate
// (exact monoisotopic mass of the cut sub-graph +- k H, ion electron), so ions with the same
// exact mass (1e-5 Da) are merged into one formula with the summed predicted weight W_F.
// Each query peak is then annotated with the candidate formulas within tolerance and per
// candidate the following are computed (mean over the query molecule's spectra):
//   expl_raw  : sum_i a_i [peak i explained] / sum_i a_i      (a = intensity)
//   expl_sqrt : same with a = sqrt(intensity)
//   expl_E    : sum_i a_i r(F_i) / sum_i a_i,  r = W/(W + W_med)  (energy-weighted: a peak explained
//               only by a high-barrier fragment counts little; W_med = median formula weight)
//   miss      : fraction of the predicted weight of the candidate's 10 strongest fragment formulas
//               (inside the observed m/z range, below the precursor) that has no peak  (penalty)
//   nexpl     : fraction of the 20 strongest peaks explained (unweighted count)
// for tolerances 5 ppm, 10 ppm, 20 ppm (floor 0.001 Da) and 0.01 Da, plus the control
//   phys_ll   : the physics model's own background-mixture log-likelihood (phys_v33pc physOf:
//               loglik(pred, tol 0.01 Da, eps = exp(logEps), range = precursor)).
// Peaks: intensity >= relmin x max, m/z < precursor - 0.5 (the precursor ion is explained by
// every candidate of the window). All features are also computed with the spectra of query
// q + shift (shuffled-query control).
//
// Evaluation (re-ranking of the top N): MRR@25 with pessimistic ties, MRR_in over queries whose
// truth is in the window, MRR_all over all queries (truth outside the top N = 0): feature alone,
// base = z_win(fp3) + z_win(attn) (as attn_eval), base + w z_N(feature) for a weight grid, and the
// same with the shuffled feature; paired difference vs base (mean, SE, queries up / down).
//
//   formula_eval dir=C:\prog\casmi qry=hcqry [top=formula\hcqry_top.bin] [N=200] [params=phys_size_fqry.txt]
//                [threads=4] [shift=7] [relmin=0.005] [feat=formula\<qry>_feat_N<N>.bin] [evalN=50,100,200]
//                [ws=0.25,0.5,1,2]
#include "phys_core.h"

#include <chrono>
#include <functional>
#include <map>
#include <numeric>
#include <sstream>

namespace {
std::map<std::string, std::string> A;
std::string get(const std::string& k, const std::string& d) { auto it = A.find(k); return it == A.end() ? d : it->second; }
std::vector<double> numList(const std::string& s) {
  std::vector<double> v; std::stringstream ss(s); std::string t;
  while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::atof(t.c_str()));
  return v;
}
struct Cand { uint32_t widx, uid; float zf, za, zash, pop; };
struct QTop { int mol, truthW, truthRank; uint32_t winSize; std::vector<Cand> c; int tpos = -1; };

constexpr int NTOL = 4;
const char* TOLN[NTOL] = {"5ppm", "10ppm", "20ppm", "0.01Da"};
inline double tolOf(int t, double mz) {
  if (t == 3) return 0.01;
  double ppm = t == 0 ? 5 : t == 1 ? 10 : 20;
  return std::max(0.001, mz * ppm * 1e-6);
}
// feature layout: per tol 5 features, then phys_ll
enum { F_RAW = 0, F_SQRT, F_E, F_MISS, F_NEXPL, F_PER_TOL };
constexpr int NF = NTOL * F_PER_TOL + 3;
const int F_LL = NTOL * F_PER_TOL, F_LL5 = F_LL + 1, F_LL3 = F_LL + 2;  // phys loglik at 0.01 / 0.005 / 0.003 Da
std::string featName(int f) {
  if (f == F_LL) return "phys_ll";
  if (f == F_LL5) return "phys_ll@0.005Da";
  if (f == F_LL3) return "phys_ll@0.003Da";
  const char* n[] = {"expl_raw", "expl_sqrt", "expl_E", "miss", "nexpl"};
  return std::string(n[f % F_PER_TOL]) + "@" + TOLN[f / F_PER_TOL];
}

struct Spec { float prec; int adduct; std::vector<float> mz, it; float minMz = 0; };

// features of one candidate (fragment list) against one spectrum
void featOne(const std::vector<Frag>& fr, int nHeavy, const Params& P, const Spec& s, const Head& h, const float* rawMz,
             const float* rawIt, std::vector<std::pair<double, double>>& pred, std::vector<std::pair<double, double>>& F,
             float* out) {
  predict(fr, P, s.adduct, nHeavy, pred);
  // control: the physics likelihood exactly as phys_v33pc physOf (all raw peaks)
  out[F_LL] = loglik(pred, rawMz, rawIt, h.n, 0.01f, std::exp(P.logEps), h.precursor);
  out[F_LL5] = loglik(pred, rawMz, rawIt, h.n, 0.005f, std::exp(P.logEps), h.precursor);
  out[F_LL3] = loglik(pred, rawMz, rawIt, h.n, 0.003f, std::exp(P.logEps), h.precursor);
  // merge into formulas (same exact mass within 1e-5 Da = same sub-formula and H shift)
  F.clear();
  for (auto& p : pred) {
    if (!F.empty() && p.first - F.back().first < 1e-5) F.back().second += p.second;
    else F.push_back(p);
  }
  std::vector<double> ws;
  ws.reserve(F.size());
  for (auto& f : F) ws.push_back(f.second);
  std::nth_element(ws.begin(), ws.begin() + ws.size() / 2, ws.end());
  double wmed = ws.empty() ? 1.0 : ws[ws.size() / 2];
  // strongest predicted formulas inside the observed range
  std::vector<int> strong;
  for (int i = 0; i < int(F.size()); ++i)
    if (F[i].first < s.prec - 0.5 && F[i].first >= s.minMz) strong.push_back(i);
  int nS = std::min<int>(10, int(strong.size()));
  std::partial_sort(strong.begin(), strong.begin() + nS, strong.end(), [&](int a, int b) { return F[a].second > F[b].second; });
  strong.resize(nS);
  // the 20 strongest peaks for the count feature
  std::vector<int> topPk(s.mz.size());
  std::iota(topPk.begin(), topPk.end(), 0);
  int nP = std::min<int>(20, int(topPk.size()));
  std::partial_sort(topPk.begin(), topPk.begin() + nP, topPk.end(), [&](int a, int b) { return s.it[a] > s.it[b]; });
  std::vector<char> isTop(s.mz.size(), 0);
  for (int j = 0; j < nP; ++j) isTop[topPk[j]] = 1;
  double sRaw = 0, sSqrt = 0;
  for (float x : s.it) { sRaw += x; sSqrt += std::sqrt(double(x)); }
  for (int t = 0; t < NTOL; ++t) {
    double eRaw = 0, eSqrt = 0, eE = 0;
    int nE = 0;
    for (size_t i = 0; i < s.mz.size(); ++i) {
      double m = s.mz[i], tol = tolOf(t, m);
      auto lo = std::lower_bound(F.begin(), F.end(), std::make_pair(m - tol, -1e300));
      double best = 0;
      for (auto p = lo; p != F.end() && p->first <= m + tol; ++p) best = std::max(best, p->second);
      if (best > 0) {
        eRaw += s.it[i]; eSqrt += std::sqrt(double(s.it[i])); eE += s.it[i] * best / (best + wmed);
        nE += isTop[i];
      }
    }
    double miss = 0, tot = 0;
    for (int j : strong) {
      double m = F[j].first, tol = tolOf(t, m);
      auto lo = std::lower_bound(s.mz.begin(), s.mz.end(), float(m - tol));
      bool seen = lo != s.mz.end() && *lo <= m + tol;
      tot += F[j].second;
      if (!seen) miss += F[j].second;
    }
    float* o = out + t * F_PER_TOL;
    o[F_RAW] = sRaw > 0 ? float(eRaw / sRaw) : 0.f;
    o[F_SQRT] = sSqrt > 0 ? float(eSqrt / sSqrt) : 0.f;
    o[F_E] = sRaw > 0 ? float(eE / sRaw) : 0.f;
    o[F_MISS] = tot > 0 ? float(miss / tot) : 0.f;
    o[F_NEXPL] = nP > 0 ? float(nE) / nP : 0.f;
  }
}

void zsc(std::vector<float>& v) {
  double m = 0, s = 0;
  for (float x : v) m += x;
  m /= double(std::max<size_t>(1, v.size()));
  for (float x : v) s += (x - m) * (x - m);
  s = std::sqrt(s / double(std::max<size_t>(1, v.size()))) + 1e-9;
  for (auto& x : v) x = float((x - m) / s);
}
}  // namespace

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    size_t e = s.find('=');
    if (e != std::string::npos) A[s.substr(0, e)] = s.substr(e + 1);
  }
  std::string dir = get("dir", "C:\\prog\\casmi"), qry = get("qry", "hcqry");
  int N = std::atoi(get("N", "200").c_str()), shift = std::atoi(get("shift", "7").c_str());
  unsigned nThr = unsigned(std::atoi(get("threads", "4").c_str()));
  float relMin = float(std::atof(get("relmin", "0.005").c_str()));
  auto t00 = std::chrono::steady_clock::now();
  auto el = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t00).count(); };

  // ---- top-N dump ----
  std::vector<QTop> Q;
  {
    auto b = slurp(dir + "/" + get("top", "formula/" + qry + "_top.bin"));
    const char* p = b.data();
    uint32_t nq; std::memcpy(&nq, p, 4); p += 4;
    for (uint32_t i = 0; i < nq; ++i) {
      QTop x; int32_t hd[3]; uint32_t ws, n;
      std::memcpy(hd, p, 12); std::memcpy(&ws, p + 12, 4); std::memcpy(&n, p + 16, 4); p += 20;
      x.mol = hd[0]; x.truthW = hd[1]; x.truthRank = hd[2]; x.winSize = ws;
      x.c.resize(n);
      for (uint32_t j = 0; j < n; ++j) {
        std::memcpy(&x.c[j].widx, p, 8); std::memcpy(&x.c[j].zf, p + 8, 16); p += 24;
      }
      if (int(x.c.size()) > N) x.c.resize(N);
      for (int j = 0; j < int(x.c.size()); ++j) if (int(x.c[j].widx) == x.truthW) x.tpos = j;
      Q.push_back(std::move(x));
    }
  }
  const int nQ = int(Q.size());
  int nIn = 0, nInN = 0;
  for (auto& q : Q) { nIn += q.truthW >= 0; nInN += q.tpos >= 0; }
  std::printf("%s: %d queries, truth in window %d, in top %d: %d (recall %.3f)\n", qry.c_str(), nQ, nIn, N, nInN,
              double(nInN) / std::max(1, nIn));

  // ---- query spectra ----
  std::vector<Head> head;
  std::vector<float> mzAll, itAll;
  {
    auto hb = slurp(dir + "/" + qry + "_head.bin");
    head.resize(hb.size() / sizeof(Head));
    std::memcpy(head.data(), hb.data(), head.size() * sizeof(Head));
    auto pb = slurp(dir + "/" + qry + "_peaks.bin");
    size_t total = 0;
    for (auto& h : head) total += size_t(h.n);
    mzAll.resize(total); itAll.resize(total);
    const float* p = reinterpret_cast<const float*>(pb.data());
    size_t cur = 0;
    for (auto& h : head) {
      std::memcpy(mzAll.data() + cur, p, size_t(h.n) * 4); p += h.n;
      std::memcpy(itAll.data() + cur, p, size_t(h.n) * 4); p += h.n;
      h.off = int32_t(cur); cur += size_t(h.n);
    }
  }
  int maxMol = 0;
  for (auto& h : head) maxMol = std::max(maxMol, h.mol + 1);
  std::vector<std::vector<int>> specOf(maxMol);
  for (size_t s = 0; s < head.size(); ++s) if (head[s].mol >= 0 && head[s].n > 0) specOf[head[s].mol].push_back(int(s));
  std::vector<Spec> SP(head.size());
  for (size_t s = 0; s < head.size(); ++s) {
    const Head& h = head[s];
    Spec& x = SP[s];
    x.prec = h.precursor; x.adduct = adductIdx(h.tag);
    float mx = 0;
    for (int i = 0; i < h.n; ++i) mx = std::max(mx, itAll[h.off + i]);
    std::vector<std::pair<float, float>> pk;
    for (int i = 0; i < h.n; ++i) {
      float m = mzAll[h.off + i], v = itAll[h.off + i];
      if (v > 0 && v >= relMin * mx && (h.precursor <= 0 || m < h.precursor - 0.5f)) pk.push_back({m, v});
    }
    std::sort(pk.begin(), pk.end());
    for (auto& p : pk) { x.mz.push_back(p.first); x.it.push_back(p.second); }
    x.minMz = pk.empty() ? 0.f : pk.front().first - 0.5f;
  }

  // ---- features (cached) ----
  std::string featPath = dir + "/" + get("feat", "formula/" + qry + "_feat_N" + std::to_string(N) + ".bin");
  // feat[q][cand][2][NF]: [0] = own spectra, [1] = spectra of query q + shift
  std::vector<std::vector<float>> feat(nQ);
  bool have = false;
  if (get("recompute", "0") != "1") {
    std::ifstream f(featPath, std::ios::binary);
    if (f) {
      uint32_t nq, nf; f.read((char*)&nq, 4); f.read((char*)&nf, 4);
      if (nq == uint32_t(nQ) && nf == uint32_t(NF)) {
        for (int qi = 0; qi < nQ; ++qi) {
          uint32_t n; f.read((char*)&n, 4);
          feat[qi].resize(size_t(n) * 2 * NF);
          f.read((char*)feat[qi].data(), feat[qi].size() * 4);
        }
        have = bool(f);
        if (have) std::printf("loaded features %s\n", featPath.c_str());
      }
    }
  }
  if (!have) {
    Params P0;
    if (!P0.load(dir + "/" + get("params", "phys_size_fqry.txt"))) { std::fprintf(stderr, "cannot load params\n"); return 1; }
    // SMILES of the needed window uids
    uint32_t maxU = 0;
    for (auto& q : Q) for (auto& c : q.c) maxU = std::max(maxU, c.uid);
    std::vector<char> need(size_t(maxU) + 1, 0);
    for (auto& q : Q) for (auto& c : q.c) need[c.uid] = 1;
    std::unordered_map<uint32_t, std::string> smi;
    {
      std::ifstream f(dir + "/w_" + qry + "_wmols.txt");
      std::string line;
      while (std::getline(f, line)) {
        size_t t = line.find('\t');
        if (t == std::string::npos) continue;
        uint32_t u = uint32_t(std::strtoul(line.c_str(), nullptr, 10));
        if (u <= maxU && need[u]) smi[u] = line.substr(t + 1);
      }
    }
    std::printf("SMILES for %zu candidates (%.0fs)\n", smi.size(), el());
    std::vector<std::pair<int, int>> tasks;
    for (int qi = 0; qi < nQ; ++qi) {
      feat[qi].assign(Q[qi].c.size() * 2 * NF, 0.f);
      for (int j = 0; j < int(Q[qi].c.size()); ++j) tasks.push_back({qi, j});
    }
    std::atomic<size_t> nx(0), done(0), bad(0);
    std::vector<std::thread> th;
    for (unsigned t = 0; t < nThr; ++t)
      th.emplace_back([&]() {
        std::vector<std::pair<double, double>> pred, F;
        std::vector<float> acc(NF);
        for (;;) {
          size_t k = nx++;
          if (k >= tasks.size()) break;
          int qi = tasks[k].first, j = tasks[k].second;
          float* out = feat[qi].data() + size_t(j) * 2 * NF;
          std::unique_ptr<RDKit::ROMol> m;
          auto it = smi.find(Q[qi].c[j].uid);
          if (it != smi.end()) {
            try { m.reset(RDKit::SmilesToMol(it->second)); } catch (...) { m.reset(); }
          }
          if (!m) {
            ++bad;
            for (int f = 0; f < 2 * NF; ++f) out[f] = std::nanf("");
            continue;
          }
          auto fr = enumerate(*m);
          int nHeavy = int(m->getNumAtoms());
          for (int side = 0; side < 2; ++side) {
            int mol = Q[side ? (qi + shift) % nQ : qi].mol;
            std::fill(acc.begin(), acc.end(), 0.f);
            int ns = 0;
            if (mol >= 0 && mol < maxMol)
              for (int s : specOf[mol]) {
                float o[NF];
                const Head& h = head[s];
                featOne(fr, nHeavy, P0, SP[s], h, mzAll.data() + h.off, itAll.data() + h.off, pred, F, o);
                for (int f = 0; f < NF; ++f) acc[f] += o[f];
                ++ns;
              }
            for (int f = 0; f < NF; ++f) out[side * NF + f] = ns ? acc[f] / ns : 0.f;
          }
          size_t d = ++done;
          if (d % 5000 == 0) { std::printf("  %zu / %zu candidates (%.0fs)\n", d, tasks.size(), el()); std::fflush(stdout); }
        }
      });
    for (auto& t : th) t.join();
    std::printf("features: %zu candidates, %zu unparsable (%.0fs, %u threads)\n", tasks.size(), size_t(bad), el(), nThr);
    std::ofstream f(featPath, std::ios::binary);
    uint32_t nq = uint32_t(nQ), nf = uint32_t(NF);
    f.write((char*)&nq, 4); f.write((char*)&nf, 4);
    for (int qi = 0; qi < nQ; ++qi) {
      uint32_t n = uint32_t(Q[qi].c.size());
      f.write((char*)&n, 4);
      f.write((char*)feat[qi].data(), feat[qi].size() * 4);
    }
  }

  // ---- evaluation ----
  auto ws = numList(get("ws", "0.25,0.5,1,2"));
  auto evalNs = numList(get("evalN", std::to_string(N)));
  // score of candidate j: generic combiner; returns per-query reciprocal ranks (MRR@25, pessimistic)
  using Comb = std::function<void(int qi, int n, std::vector<float>& sc)>;
  auto run = [&](int n, const Comb& cb) {
    std::vector<double> rr(nQ, 0.0);
    for (int qi = 0; qi < nQ; ++qi) {
      int tp = Q[qi].tpos;
      int nn = std::min<int>(n, int(Q[qi].c.size()));
      if (tp < 0 || tp >= nn) continue;
      std::vector<float> sc;
      cb(qi, nn, sc);
      int better = 0;
      for (int j = 0; j < nn; ++j) if (j != tp && !(sc[j] < sc[tp])) ++better;  // NaN at truth -> worst
      if (std::isnan(sc[tp])) better = nn;
      if (better < 25) rr[qi] = 1.0 / (better + 1);
    }
    return rr;
  };
  auto mean = [&](const std::vector<double>& rr) { return std::accumulate(rr.begin(), rr.end(), 0.0); };
  // z of a feature over the first n candidates (NaN -> min); sign: larger = better
  auto zfeat = [&](int qi, int n, int f, int side, float sign, std::vector<float>& z) {
    z.resize(n);
    float mn = 1e30f;
    for (int j = 0; j < n; ++j) {
      float v = feat[qi][size_t(j) * 2 * NF + side * NF + f];
      z[j] = std::isnan(v) ? std::nanf("") : sign * v;
      if (!std::isnan(z[j])) mn = std::min(mn, z[j]);
    }
    for (auto& x : z) if (std::isnan(x)) x = mn - 1e-3f;
    zsc(z);
  };
  // combined score feature set: "f" or "f-lam*miss"
  struct FS { std::string name; int f; int miss; double lam; };
  std::vector<FS> fsets;
  for (int f = 0; f < NF; ++f) fsets.push_back({featName(f), f, -1, 0});
  for (int t = 0; t < NTOL; ++t)
    for (double lam : {0.25, 0.5, 1.0}) {
      char b[80];
      std::snprintf(b, sizeof b, "expl_raw - %.2g miss @%s", lam, TOLN[t]);
      fsets.push_back({b, t * F_PER_TOL + F_RAW, t * F_PER_TOL + F_MISS, lam});
    }
  auto fsScore = [&](int qi, int n, const FS& fs, int side, std::vector<float>& z) {
    float sign = (fs.f < F_LL && fs.f % F_PER_TOL == F_MISS) ? -1.f : 1.f;
    if (fs.miss < 0) { zfeat(qi, n, fs.f, side, sign, z); return; }
    z.resize(n);
    for (int j = 0; j < n; ++j) {
      const float* o = feat[qi].data() + size_t(j) * 2 * NF + side * NF;
      z[j] = std::isnan(o[fs.f]) ? -1e3f : float(o[fs.f] - fs.lam * o[fs.miss]);
    }
    zsc(z);
  };
  for (double nd : evalNs) {
    int n = int(nd);
    int inN = 0;
    for (auto& q : Q) inN += q.tpos >= 0 && q.tpos < n;
    std::printf("\n=== %s, re-rank of the top %d by fp3+attn (truth in top %d: %d of %d in window) ===\n", qry.c_str(), n, n,
                inN, nIn);
    auto pr = [&](const std::string& lab, const std::vector<double>& rr, const std::vector<double>* base) {
      double s = mean(rr);
      std::printf("  %-46s MRR_in %.4f  MRR_all %.4f", lab.c_str(), s / std::max(1, nIn), s / nQ);
      if (base) {
        std::vector<double> d;
        int up = 0, dn = 0;
        for (int qi = 0; qi < nQ; ++qi) {
          if (Q[qi].truthW < 0) continue;
          d.push_back(rr[qi] - (*base)[qi]);
          up += rr[qi] > (*base)[qi] + 1e-12; dn += rr[qi] < (*base)[qi] - 1e-12;
        }
        double m = std::accumulate(d.begin(), d.end(), 0.0) / d.size(), v = 0;
        for (double x : d) v += (x - m) * (x - m);
        std::printf("  d %+.4f (SE %.4f) up %d down %d", m, std::sqrt(v / d.size() / d.size()), up, dn);
      }
      std::printf("\n");
      std::fflush(stdout);
    };
    auto base = run(n, [&](int qi, int nn, std::vector<float>& sc) {
      sc.resize(nn);
      for (int j = 0; j < nn; ++j) sc[j] = Q[qi].c[j].zf + Q[qi].c[j].za;
    });
    pr("base: z(fp3) + z(attn)", base, nullptr);
    pr("z(fp3) alone", run(n, [&](int qi, int nn, std::vector<float>& sc) {
         sc.resize(nn);
         for (int j = 0; j < nn; ++j) sc[j] = Q[qi].c[j].zf;
       }), &base);
    pr("base + 0.15 pop", run(n, [&](int qi, int nn, std::vector<float>& sc) {
         sc.resize(nn);
         for (int j = 0; j < nn; ++j) sc[j] = Q[qi].c[j].zf + Q[qi].c[j].za + 0.15f * Q[qi].c[j].pop;
       }), &base);
    for (auto& fs : fsets) {
      std::printf(" [%s]\n", fs.name.c_str());
      for (int side = 0; side < 2; ++side)
        pr(std::string(side ? "SHUF " : "") + "alone", run(n, [&](int qi, int nn, std::vector<float>& sc) {
             fsScore(qi, nn, fs, side, sc);
           }), &base);
      for (double w : ws)
        for (int side = 0; side < 2; ++side) {
          char lab[80];
          std::snprintf(lab, sizeof lab, "%sbase + %.2g z", side ? "SHUF " : "", w);
          pr(lab, run(n, [&](int qi, int nn, std::vector<float>& sc) {
               fsScore(qi, nn, fs, side, sc);
               for (int j = 0; j < nn; ++j) sc[j] = Q[qi].c[j].zf + Q[qi].c[j].za + float(w) * sc[j];
             }), &base);
        }
    }
    // formula + physics likelihood together: does the formula score add anything beyond the old physics term?
    std::printf(" [base + a z(phys_ll) + b z(formula feature)]\n");
    const int combF[] = {1 * F_PER_TOL + F_E, 2 * F_PER_TOL + F_E, 3 * F_PER_TOL + F_E, 1 * F_PER_TOL + F_RAW,
                         2 * F_PER_TOL + F_RAW, 1 * F_PER_TOL + F_NEXPL, F_LL3};
    for (int cf : combF)
      for (double a : {0.0, 0.125, 0.25, 0.5})
        for (double b : {0.0, 0.125, 0.25, 0.5}) {
          if (b == 0.0 && cf != combF[0]) continue;
          for (int side = 0; side < 2; ++side) {
            if (side && b == 0.0) continue;
            char lab[96];
            std::snprintf(lab, sizeof lab, "%sbase + %.3g ll + %.3g %s", side ? "SHUF-f " : "", a, b, featName(cf).c_str());
            pr(lab, run(n, [&](int qi, int nn, std::vector<float>& sc) {
                 std::vector<float> za, zb;
                 zfeat(qi, nn, F_LL, 0, 1.f, za);
                 zfeat(qi, nn, cf, side, 1.f, zb);
                 sc.resize(nn);
                 for (int j = 0; j < nn; ++j) sc[j] = Q[qi].c[j].zf + Q[qi].c[j].za + float(a) * za[j] + float(b) * zb[j];
               }), &base);
          }
        }
    // wider grid with the paired difference against the physics term alone at the same weight a:
    // "does the formula score add anything once phys_ll is in?"
    std::printf(" [grid2: base + a z(phys_ll) + b z(f); dLL = vs base + a z(phys_ll)]\n");
    auto dstat = [&](const std::vector<double>& rr, const std::vector<double>& ref, double& m, double& se, int& up, int& dn) {
      std::vector<double> d;
      up = dn = 0;
      for (int qi = 0; qi < nQ; ++qi) {
        if (Q[qi].truthW < 0) continue;
        d.push_back(rr[qi] - ref[qi]);
        up += rr[qi] > ref[qi] + 1e-12; dn += rr[qi] < ref[qi] - 1e-12;
      }
      m = std::accumulate(d.begin(), d.end(), 0.0) / d.size();
      double v = 0;
      for (double x : d) v += (x - m) * (x - m);
      se = std::sqrt(v / d.size() / d.size());
    };
    std::vector<FS> g2 = {{featName(2 * F_PER_TOL + F_E), 2 * F_PER_TOL + F_E, -1, 0},
                          {featName(1 * F_PER_TOL + F_E), 1 * F_PER_TOL + F_E, -1, 0},
                          {featName(2 * F_PER_TOL + F_RAW), 2 * F_PER_TOL + F_RAW, -1, 0},
                          {featName(1 * F_PER_TOL + F_NEXPL), 1 * F_PER_TOL + F_NEXPL, -1, 0},
                          {"expl_raw - 0.5 miss @20ppm", 2 * F_PER_TOL + F_RAW, 2 * F_PER_TOL + F_MISS, 0.5}};
    for (double a : {0.25, 0.5, 0.75, 1.0, 1.5}) {
      auto rrA = run(n, [&](int qi, int nn, std::vector<float>& sc) {
        zfeat(qi, nn, F_LL, 0, 1.f, sc);
        for (int j = 0; j < nn; ++j) sc[j] = Q[qi].c[j].zf + Q[qi].c[j].za + float(a) * sc[j];
      });
      double m, se; int up, dn;
      dstat(rrA, base, m, se, up, dn);
      std::printf("  g2 a=%.3g ll only                                 MRR_in %.4f MRR_all %.4f  d %+.4f (%.4f)\n", a,
                  mean(rrA) / std::max(1, nIn), mean(rrA) / nQ, m, se);
      for (auto& fs : g2)
        for (double b : {0.125, 0.25, 0.5, 1.0})
          for (int side = 0; side < 2; ++side) {
            auto rr = run(n, [&](int qi, int nn, std::vector<float>& sc) {
              std::vector<float> za;
              zfeat(qi, nn, F_LL, 0, 1.f, za);
              fsScore(qi, nn, fs, side, sc);
              for (int j = 0; j < nn; ++j) sc[j] = Q[qi].c[j].zf + Q[qi].c[j].za + float(a) * za[j] + float(b) * sc[j];
            });
            double m2, se2; int up2, dn2;
            dstat(rr, base, m, se, up, dn);
            dstat(rr, rrA, m2, se2, up2, dn2);
            std::printf("  g2 a=%.3g %s+ %.3g %-30s MRR_in %.4f MRR_all %.4f  d %+.4f (%.4f)  dLL %+.4f (%.4f) up %d down %d\n", a,
                        side ? "SHUF " : "", b, fs.name.c_str(), mean(rr) / std::max(1, nIn), mean(rr) / nQ, m, se, m2, se2,
                        up2, dn2);
          }
    }
    std::fflush(stdout);
  }
  // correlation of the formula features with the physics likelihood inside the top N (per query, mean)
  std::printf("\nmean within-query Pearson r with phys_ll (own spectra, top %d):\n", N);
  for (int f = 0; f < NF; ++f) {
    if (f == F_LL) continue;
    double sr = 0; int cnt = 0;
    for (int qi = 0; qi < nQ; ++qi) {
      int n = int(Q[qi].c.size());
      if (n < 5) continue;
      std::vector<float> a, b;
      zfeat(qi, n, f, 0, 1.f, a); zfeat(qi, n, F_LL, 0, 1.f, b);
      double r = 0;
      for (int j = 0; j < n; ++j) r += double(a[j]) * b[j];
      sr += r / n; ++cnt;
    }
    std::printf("  %-20s r %.3f\n", featName(f).c_str(), sr / std::max(1, cnt));
  }
  std::printf("all done (%.0fs)\n", el());
  return 0;
}
