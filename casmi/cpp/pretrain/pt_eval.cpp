// pt_eval: attn_eval (cpp/attn/attn_eval.cpp, attn_STATUS.md) with pretrained-embedding neighbours
// (pretrain_STATUS.md). Changes vs attn_eval:
//   embnbr=<file>  (pt_train nbr output) replaces neighbour slots 0..2 by the embedding-cosine neighbours
//                  (0 = embG global token, 1 = embM mean of peak tokens, 2 = embGM average of both cosines);
//                  slot 3 stays the shift-aware entropy (ment) neighbours of the attn cache.
//   after the per-config controls, if refcfg= is given: z-sum of the two attention scores
//   z(attn best) + w z(attn ref) (+ SHUF control of the first), and with the models.
// Everything else (window, MRR@25 pessimistic ties, Tanimoto scoring, controls) is attn_eval's.
//
// attn_eval: memory-based "second engine" on the window yardstick (attn_STATUS.md).
//
// Idea (one Hopfield / attention retrieval step): for a query molecule, retrieve
// the library spectra most similar to its spectra, weight the retrieved library
// MOLECULES by w = softmax(sim / T) over the top k, and predict the query's
// fingerprint as the attention-weighted average of their fingerprints:
//     p_b = (1 - lam) * sum_j w_j x_jb + lam * f_b      (f = library bit frequency)
// Candidates of the PubChem +-10 ppm window (win_build cache) are scored by the
// continuous Tanimoto of p vs candidate bits, exactly as win_eval.cpp:
//     T(p, c) = sum_{b in c} p_b / (|c| + sum_b p_b - sum_{b in c} p_b).
// T -> 0 with k = 1 is "Tanimoto to the best library analog"; larger T blends.
//
// Similarities (all through an inverted peak index over the library, so every
// library spectrum is compared, no precursor-mass prefilter):
//   cos  : sqrt-intensity cosine (sum-normalised a: cos = sum sqrt(a_i b_j))
//   ent  : entropy similarity (Li 2021; entropy-weighted, sum-normalised):
//          sim = sum_matched [(a+b)ln(a+b) - a ln a - b ln b] / ln 4
//          (the unmatched-peak terms of 1-(2S_AB-S_A-S_B)/ln4 cancel exactly)
//   mcos / ment : shift-aware ("modified") versions: fragment matches plus
//          neutral-loss matches (precursor - m/z) when the precursors differ
//          by more than the tolerance.
// Pairs are summed (no 1:1 assignment) and clipped at 1.
// Per library molecule: max over its spectra and over the query's spectra.
// Polarity: library spectra of the other polarity are skipped.
//
//   attn_eval dir=C:\prog\casmi qry=hcqry win=w_hcqry lib=wlib vocabs=v4k,fp3c16k
//             [models=rev2_w.bin:v4k,rev3_fp3c16k_big.bin:fp3c16k] [pop=1]
//             [tol=0.01] [maxpk=40] [nbr=200] [threads=6] [shift=7]
//             [ks=10,50,200] [Ts=0.01,0.03,0.1,0.3] [lams=0,0.2] [sims=0,1,2,3]
//             [ensw=0.25,0.5,1,2] [dump=prefix]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "parallel.hpp"
#include "rev2_model.hpp"

using namespace rev2;

namespace {
std::vector<char> slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(1); }
  std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<char> buf((size_t(n)));
  f.read(buf.data(), n);
  return buf;
}
struct Store {
  std::vector<Head> head;
  std::vector<float> mz, inten;
  size_t nMol = 0;
};
Store loadSpectra(const std::string& dir, const std::string& stem) {
  Store s;
  std::vector<char> hb = slurp(dir + "/" + stem + "_head.bin");
  s.head.resize(hb.size() / sizeof(Head));
  std::memcpy(s.head.data(), hb.data(), s.head.size() * sizeof(Head));
  std::vector<char> pb = slurp(dir + "/" + stem + "_peaks.bin");
  size_t total = 0;
  for (const Head& h : s.head) total += size_t(h.n);
  s.mz.resize(total); s.inten.resize(total);
  const float* p = reinterpret_cast<const float*>(pb.data());
  size_t cur = 0;
  for (Head& h : s.head) {
    std::memcpy(s.mz.data() + cur, p, size_t(h.n) * 4); p += h.n;
    std::memcpy(s.inten.data() + cur, p, size_t(h.n) * 4); p += h.n;
    h.off = int32_t(cur);
    cur += size_t(h.n);
    s.nMol = std::max(s.nMol, size_t(h.mol + 1));
  }
  return s;
}
struct Fps { std::vector<uint64_t> ptr; std::vector<uint16_t> bits; uint32_t nBits = 0, n = 0; };
Fps loadFps(const std::string& path) {
  Fps f;
  std::vector<char> b = slurp(path);
  const char* p = b.data();
  uint32_t n; std::memcpy(&n, p, 4); std::memcpy(&f.nBits, p + 4, 4); p += 8;
  f.n = n;
  f.ptr.resize(n + 1); f.ptr[0] = 0;
  f.bits.reserve((b.size() - 8) / 2);
  for (uint32_t i = 0; i < n; ++i) {
    uint16_t c; std::memcpy(&c, p, 2); p += 2;
    size_t old = f.bits.size();
    f.bits.resize(old + c);
    std::memcpy(f.bits.data() + old, p, 2 * size_t(c)); p += 2 * size_t(c);
    f.ptr[i + 1] = f.bits.size();
  }
  return f;
}
struct Win { int mol, truth; std::vector<uint32_t> ids; };
std::vector<Win> loadWin(const std::string& path) {
  std::vector<Win> w;
  std::vector<char> b = slurp(path);
  const char* p = b.data();
  uint32_t nq; std::memcpy(&nq, p, 4); p += 4;
  for (uint32_t q = 0; q < nq; ++q) {
    Win x; uint32_t n;
    std::memcpy(&x.mol, p, 4); std::memcpy(&x.truth, p + 4, 4); std::memcpy(&n, p + 8, 4); p += 12;
    x.ids.resize(n);
    std::memcpy(x.ids.data(), p, 4 * size_t(n)); p += 4 * size_t(n);
    w.push_back(std::move(x));
  }
  return w;
}

std::vector<std::vector<float>> predictByMol(Model& m, const Store& q) {
  std::vector<std::vector<float>> sum(q.nMol);
  std::vector<int> cnt(q.nMol, 0);
  std::vector<int> spec;
  for (size_t s = 0; s < q.head.size(); ++s) if (q.head[s].n > 0 && q.head[s].mol >= 0) spec.push_back(int(s));
  const int B = 256;
  std::vector<std::vector<std::pair<int, float>>> rows;
  std::vector<float> P;
  for (size_t s0 = 0; s0 < spec.size(); s0 += B) {
    size_t s1 = std::min(spec.size(), s0 + B);
    rows.assign(s1 - s0, {});
    for (size_t i = 0; i < s1 - s0; ++i) {
      const Head& h = q.head[spec[s0 + i]];
      featurize(m.fc, h, q.mz.data() + h.off, q.inten.data() + h.off, rows[i]);
    }
    m.setBatch(rows, false);
    m.forward(int(s1 - s0), false, 0.f, 0);
    download(m.P, P, int64_t(s1 - s0) * m.NB);
    for (size_t s = s0; s < s1; ++s) {
      int mol = q.head[spec[s]].mol;
      auto& v = sum[mol];
      if (v.empty()) v.assign(m.NB, 0.f);
      for (int i = 0; i < m.NB; ++i) v[i] += P[(s - s0) * m.NB + i];
      ++cnt[mol];
    }
  }
  for (size_t i = 0; i < sum.size(); ++i)
    for (auto& x : sum[i]) x /= float(cnt[i]);
  return sum;
}

void zs(std::vector<float>& v) {
  double m = 0, s = 0;
  for (float x : v) m += x;
  m /= double(std::max<size_t>(1, v.size()));
  for (float x : v) s += (x - m) * (x - m);
  s = std::sqrt(s / double(std::max<size_t>(1, v.size()))) + 1e-9;
  for (auto& x : v) x = float((x - m) / s);
}
int rankOf(const std::vector<float>& f, int t) {
  int better = 0;
  for (size_t i = 0; i < f.size(); ++i) if (int(i) != t && f[i] >= f[t]) ++better;
  return better + 1;
}

std::map<std::string, std::string> A;
std::string get(const std::string& k, const std::string& d) { auto it = A.find(k); return it == A.end() ? d : it->second; }
std::vector<double> numList(const std::string& s) {
  std::vector<double> v; std::stringstream ss(s); std::string t;
  while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::atof(t.c_str()));
  return v;
}

// ---------- spectra for retrieval ----------
int adductOf(const Head& h) {
  if (h.tag < 0) return -1;
  return h.tag % 16 == 15 ? h.tag / 16 + 1 : h.tag / 16;
}
// +1 positive, -1 negative, 0 unknown
int polarity(const Head& h) {
  int a = adductOf(h);
  if (a < 0 || a >= N_ADDUCT) return 0;
  return ADDUCT_SHIFT[a] < 0.f && a != 7 ? -1 : 1;
}
struct Pk { float mz, s, e; };  // s = sqrt(a) (a sum-normalised), e = entropy-weighted, sum 1
void prepSpec(const Head& h, const float* mz, const float* it, int maxPk, std::vector<Pk>& out) {
  out.clear();
  std::vector<int> idx;
  for (int i = 0; i < h.n; ++i)
    if (it[i] > 0.f && mz[i] > 0.f && (h.precursor <= 0.f || mz[i] < h.precursor - 1.0f)) idx.push_back(i);
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
inline float gEnt(float a, float b) {
  double s = double(a) + b;
  return float((s * std::log(s) - double(a) * std::log(double(a)) - double(b) * std::log(double(b))) / std::log(4.0));
}

struct Post { uint32_t spec; float mz, s, e; };
struct Index {
  float bw = 0.01f; int nb = 0;
  std::vector<uint64_t> off;
  std::vector<Post> p;
};
// key(spec, peak) -> value (m/z or loss); built by counting sort over bins
template <class KeyFn>
Index buildIndex(const std::vector<std::vector<Pk>>& specs, float bw, float maxV, KeyFn key) {
  Index ix; ix.bw = bw; ix.nb = int(maxV / bw) + 2;
  ix.off.assign(size_t(ix.nb) + 1, 0);
  for (size_t s = 0; s < specs.size(); ++s)
    for (const Pk& k : specs[s]) { float v = key(s, k); if (v > 0.f && v < maxV) ++ix.off[size_t(v / bw) + 1]; }
  for (int b = 0; b < ix.nb; ++b) ix.off[b + 1] += ix.off[b];
  ix.p.resize(ix.off[ix.nb]);
  std::vector<uint64_t> cur(ix.off.begin(), ix.off.end() - 1);
  for (size_t s = 0; s < specs.size(); ++s)
    for (const Pk& k : specs[s]) {
      float v = key(s, k);
      if (v > 0.f && v < maxV) ix.p[cur[size_t(v / bw)]++] = {uint32_t(s), v, k.s, k.e};
    }
  return ix;
}

struct Nbr { float sim; int mol; };
const char* SIMN[] = {"embG", "embM", "embGM", "ment"};
}  // namespace

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    size_t e = s.find('=');
    if (e != std::string::npos) A[s.substr(0, e)] = s.substr(e + 1);
  }
  std::string dir = get("dir", "C:\\prog\\casmi"), qry = get("qry", "hcqry"), win = get("win", "w_" + qry),
              lib = get("lib", "wlib");
  int shift = std::atoi(get("shift", "7").c_str());
  float tol = float(std::atof(get("tol", "0.01").c_str()));
  int maxPk = std::atoi(get("maxpk", "40").c_str());
  int NBR = std::atoi(get("nbr", "200").c_str());
  float maxDm = float(std::atof(get("maxdm", "0").c_str()));  // 0 = no precursor-difference limit
  int mPen = std::atoi(get("mpen", "0").c_str());              // sim *= min(1, matched peaks / mpen)
  bool usePop = get("pop", "0") == "1";
  float popW = float(std::atof(get("popw", "0.15").c_str()));
  parallel_cap() = unsigned(std::atoi(get("threads", "6").c_str()));
  auto t00 = std::chrono::steady_clock::now();
  auto el = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t00).count(); };

  Store q = loadSpectra(dir, qry);
  auto wins = loadWin(dir + "/" + win + "_win.bin");
  const int nQ = int(wins.size());
  std::vector<float> pop;
  {
    auto b = slurp(dir + "/" + win + "_wpop.bin");
    pop.resize(b.size() / 4);
    std::memcpy(pop.data(), b.data(), b.size());
  }
  int nIn = 0; double wsz = 0;
  for (auto& w : wins) { nIn += w.truth >= 0; wsz += double(w.ids.size()); }
  std::printf("%s: %d queries, truth in window %d (%.3f), mean window %.0f candidates\n", qry.c_str(), nQ, nIn,
              double(nIn) / nQ, wsz / nQ);

  // ---------------- retrieval (or load cached neighbours) ----------------
  std::string nbrPath = dir + "/" + get("nbrfile", "attn_" + qry + "_" + lib + "_v2_t" + get("tol", "0.01") + "_p" + std::to_string(maxPk) +
                                                "_d" + get("maxdm", "0") + "_m" + get("mpen", "0") + "_nbr.bin");
  std::vector<std::vector<std::vector<Nbr>>> nbr(4, std::vector<std::vector<Nbr>>(q.nMol));
  bool haveCache = false;
  if (get("recompute", "0") != "1") {
    std::ifstream f(nbrPath, std::ios::binary);
    if (f) {
      uint32_t nm, K; f.read((char*)&nm, 4); f.read((char*)&K, 4);
      if (nm == q.nMol) {
        for (int t = 0; t < 4; ++t)
          for (uint32_t m = 0; m < nm; ++m) {
            uint32_t c; f.read((char*)&c, 4);
            nbr[t][m].resize(c);
            f.read((char*)nbr[t][m].data(), sizeof(Nbr) * c);
          }
        haveCache = bool(f);
        std::printf("loaded neighbours %s\n", nbrPath.c_str());
      }
    }
  }
  if (A.count("embnbr")) {
    if (!haveCache) { std::fprintf(stderr, "embnbr needs the attn neighbour cache %s\n", nbrPath.c_str()); return 1; }
    std::ifstream f(dir + "/" + A["embnbr"], std::ios::binary);
    uint32_t nm = 0, K = 0;
    if (f) { f.read((char*)&nm, 4); f.read((char*)&K, 4); }
    if (!f || nm != q.nMol) { std::fprintf(stderr, "cannot read embnbr %s\n", A["embnbr"].c_str()); return 1; }
    for (int t = 0; t < 4; ++t)
      for (uint32_t m = 0; m < nm; ++m) {
        uint32_t c; f.read((char*)&c, 4);
        std::vector<Nbr> v(c);
        if (c) f.read((char*)v.data(), sizeof(Nbr) * c);
        if (t < 3) nbr[t][m] = std::move(v);
      }
    std::printf("loaded embedding neighbours %s into slots 0-2\n", A["embnbr"].c_str());
  }
  if (!haveCache) {
    Store L = loadSpectra(dir, lib);
    std::printf("library %s: %zu spectra, %zu molecules (%.0fs)\n", lib.c_str(), L.head.size(), L.nMol, el());
    std::vector<std::vector<Pk>> lp(L.head.size());
    std::vector<int8_t> lpol(L.head.size());
    ::parallel_for(int64_t(L.head.size()), [&](int64_t s) {
      const Head& h = L.head[s];
      if (h.mol >= 0) prepSpec(h, L.mz.data() + h.off, L.inten.data() + h.off, maxPk, lp[s]);
      lpol[s] = int8_t(polarity(h));
    });
    std::vector<float> lprec(L.head.size());
    std::vector<int> lmol(L.head.size());
    for (size_t s = 0; s < L.head.size(); ++s) { lprec[s] = L.head[s].precursor; lmol[s] = L.head[s].mol; }
    { std::vector<float>().swap(L.mz); std::vector<float>().swap(L.inten); }
    Index fx = buildIndex(lp, tol, 2500.f, [](size_t, const Pk& k) { return k.mz; });
    Index lx = buildIndex(lp, tol, 2500.f, [&](size_t s, const Pk& k) { return lprec[s] > 0 ? lprec[s] - k.mz : -1.f; });
    std::printf("index: %zu fragment postings, %zu loss postings (%.0fs)\n", fx.p.size(), lx.p.size(), el());
    { std::vector<std::vector<Pk>>().swap(lp); }
    const size_t nLS = L.head.size(), nLM = L.nMol;
    std::vector<std::vector<int>> qspec(q.nMol);
    for (size_t s = 0; s < q.head.size(); ++s) if (q.head[s].mol >= 0 && q.head[s].n > 0) qspec[q.head[s].mol].push_back(int(s));
    std::vector<int> qmols;
    for (size_t m = 0; m < q.nMol; ++m) if (!qspec[m].empty()) qmols.push_back(int(m));
    int nT = int(std::max(1u, parallel_cap()));
    ::parallel_for(nT, [&](int64_t t) {
      // per library spectrum accumulators over the query's peaks:
      //   F, E    : fragment-only cosine / entropy terms
      //   MF, ME  : shift-aware: per query peak the better of the fragment and the neutral-loss match
      //   NM      : number of matched query peaks (shift-aware)
      // per query peak only the best posting of each library spectrum counts (no pair inflation)
      std::vector<float> F(nLS, 0.f), E(nLS, 0.f), MF(nLS, 0.f), ME(nLS, 0.f);
      std::vector<float> cF(nLS, 0.f), cE(nLS, 0.f), cMF(nLS, 0.f), cME(nLS, 0.f);
      std::vector<uint16_t> NM(nLS, 0);
      std::vector<char> seen(nLS, 0), cseen(nLS, 0);
      std::vector<uint32_t> touched, ctouched;
      std::vector<std::vector<float>> best(4, std::vector<float>(nLM, 0.f));
      std::vector<int> tm;
      std::vector<Pk> qp;
      for (size_t i = size_t(t); i < qmols.size(); i += size_t(nT)) {
        int m = qmols[i];
        tm.clear();
        for (int s : qspec[m]) {
          const Head& h = q.head[s];
          prepSpec(h, q.mz.data() + h.off, q.inten.data() + h.off, maxPk, qp);
          int pol = polarity(h);
          touched.clear();
          auto scan = [&](const Index& ix, float v, const Pk& k, bool loss) {
            if (v <= 0.f) return;
            int b0 = int(v / ix.bw);
            for (int b = std::max(0, b0 - 1); b <= std::min(ix.nb - 1, b0 + 1); ++b)
              for (uint64_t j = ix.off[b]; j < ix.off[b + 1]; ++j) {
                const Post& P = ix.p[j];
                if (std::fabs(P.mz - v) > tol) continue;
                const uint32_t sp = P.spec;
                if (pol != 0 && lpol[sp] != 0 && lpol[sp] != pol) continue;
                if (maxDm > 0.f && std::fabs(lprec[sp] - h.precursor) > maxDm) continue;
                if (loss && !(lprec[sp] > 0 && std::fabs(h.precursor - lprec[sp]) > tol)) continue;
                if (!cseen[sp]) { cseen[sp] = 1; ctouched.push_back(sp); }
                float vs = k.s * P.s, ve = gEnt(k.e, P.e);
                if (!loss) { cF[sp] = std::max(cF[sp], vs); cE[sp] = std::max(cE[sp], ve); }
                cMF[sp] = std::max(cMF[sp], vs); cME[sp] = std::max(cME[sp], ve);
              }
          };
          for (const Pk& k : qp) {
            ctouched.clear();
            scan(fx, k.mz, k, false);
            if (h.precursor > 0) scan(lx, h.precursor - k.mz, k, true);
            for (uint32_t sp : ctouched) {
              if (!seen[sp]) { seen[sp] = 1; touched.push_back(sp); }
              F[sp] += cF[sp]; E[sp] += cE[sp]; MF[sp] += cMF[sp]; ME[sp] += cME[sp];
              ++NM[sp];
              cF[sp] = cE[sp] = cMF[sp] = cME[sp] = 0.f; cseen[sp] = 0;
            }
          }
          for (uint32_t sp : touched) {
            int lm = lmol[sp];
            float pen = mPen > 0 ? std::min(1.f, float(NM[sp]) / float(mPen)) : 1.f;
            float v[4] = {std::min(1.f, F[sp]) * pen, std::min(1.f, E[sp]) * pen, std::min(1.f, MF[sp]) * pen,
                          std::min(1.f, ME[sp]) * pen};
            if (lm >= 0) {
              bool fresh = true;
              for (int ty = 0; ty < 4; ++ty) { if (best[ty][lm] > 0.f) fresh = false; }
              if (fresh) tm.push_back(lm);
              for (int ty = 0; ty < 4; ++ty) best[ty][lm] = std::max(best[ty][lm], std::max(v[ty], 1e-9f));
            }
            F[sp] = E[sp] = MF[sp] = ME[sp] = 0.f; NM[sp] = 0; seen[sp] = 0;
          }
        }
        for (int ty = 0; ty < 4; ++ty) {
          std::vector<Nbr> c;
          c.reserve(tm.size());
          for (int lm : tm) c.push_back({best[ty][lm], lm});
          int K = std::min<int>(NBR, int(c.size()));
          std::partial_sort(c.begin(), c.begin() + K, c.end(), [](const Nbr& a, const Nbr& b) { return a.sim > b.sim; });
          c.resize(K);
          nbr[ty][m] = std::move(c);
        }
        for (int lm : tm) for (int ty = 0; ty < 4; ++ty) best[ty][lm] = 0.f;
      }
    });
    std::printf("retrieval done (%.0fs)\n", el());
    std::ofstream f(nbrPath, std::ios::binary);
    uint32_t nm = uint32_t(q.nMol), K = uint32_t(NBR);
    f.write((char*)&nm, 4); f.write((char*)&K, 4);
    for (int t = 0; t < 4; ++t)
      for (uint32_t m = 0; m < nm; ++m) {
        uint32_t c = uint32_t(nbr[t][m].size());
        f.write((char*)&c, 4);
        f.write((char*)nbr[t][m].data(), sizeof(Nbr) * c);
      }
  }
  // neighbour statistics
  for (int t = 0; t < 4; ++t) {
    std::vector<float> top;
    for (auto& v : nbr[t]) if (!v.empty()) top.push_back(v[0].sim);
    std::sort(top.begin(), top.end());
    if (!top.empty())
      std::printf("  sim %-4s: %zu query mols with neighbours, top-1 sim median %.3f p10 %.3f p90 %.3f\n", SIMN[t],
                  top.size(), top[top.size() / 2], top[top.size() / 10], top[top.size() * 9 / 10]);
  }

  // pessimistic control: drop retrieved molecules that are fingerprint twins of the truth
  // (binary Tanimoto >= twinthr in vocabulary twinvocab) - stereo / tautomer / salt forms that
  // survived the InChIKey14 exclusion of the evaluation molecules from the library.
  if (get("notwin", "0") == "1") {
    std::string tv = get("twinvocab", "fp3c16k");
    double thr = std::atof(get("twinthr", "0.999").c_str());
    Fps LF = loadFps(dir + "/" + lib + "_" + tv + ".bin");
    Fps CF = loadFps(dir + "/" + win + "_" + tv + ".bin");
    std::vector<int> dropped(4, 0), qAff(4, 0);
    for (int qi = 0; qi < nQ; ++qi) {
      const Win& w = wins[qi];
      if (w.truth < 0) continue;
      uint32_t tid = w.ids[w.truth];
      std::vector<uint16_t> tb(CF.bits.begin() + CF.ptr[tid], CF.bits.begin() + CF.ptr[tid + 1]);
      std::sort(tb.begin(), tb.end());
      for (int ty = 0; ty < 4; ++ty) {
        auto& v = nbr[ty][w.mol];
        size_t before = v.size();
        v.erase(std::remove_if(v.begin(), v.end(), [&](const Nbr& x) {
                  int inter = 0, nl = 0;
                  for (uint64_t b = LF.ptr[x.mol]; b < LF.ptr[x.mol + 1]; ++b, ++nl)
                    inter += std::binary_search(tb.begin(), tb.end(), LF.bits[b]);
                  return double(inter) / std::max<int>(1, int(tb.size()) + nl - inter) >= thr;
                }),
                v.end());
        dropped[ty] += int(before - v.size());
        qAff[ty] += before != v.size();
      }
    }
    for (int ty = 0; ty < 4; ++ty)
      std::printf("  notwin (%s, Tanimoto >= %g): %s dropped %d neighbours in %d queries\n", tv.c_str(), thr, SIMN[ty],
                  dropped[ty], qAff[ty]);
  }

  // ---------------- evaluation ----------------
  struct R { double in = 0, all = 0, top1 = 0, top25 = 0, top100 = 0, medrank = 0; };
  auto evalScores = [&](const std::function<void(int, std::vector<float>&)>& sc) {
    R r;
    std::vector<int> ranks(nQ, 0);
    ::parallel_for(nQ, [&](int64_t qi) {
      if (wins[qi].truth < 0) return;
      std::vector<float> f;
      sc(int(qi), f);
      ranks[qi] = rankOf(f, wins[qi].truth);
    });
    std::vector<int> rk;
    for (int qi = 0; qi < nQ; ++qi) {
      if (wins[qi].truth < 0) continue;
      int k = ranks[qi];
      rk.push_back(k);
      if (k <= 25) r.in += 1.0 / k;
      r.top1 += k == 1; r.top25 += k <= 25; r.top100 += k <= 100;
    }
    int n = std::max<int>(1, int(rk.size()));
    r.all = r.in / nQ; r.in /= n; r.top1 /= n; r.top25 /= n; r.top100 /= n;
    std::sort(rk.begin(), rk.end());
    r.medrank = rk.empty() ? 0 : rk[rk.size() / 2];
    return r;
  };
  auto pr = [&](const std::string& lab, const R& r) {
    std::printf("  %-44s MRR_in %.4f  MRR_all %.4f  top1 %.3f top25 %.3f top100 %.3f  median rank %.0f\n", lab.c_str(),
                r.in, r.all, r.top1, r.top25, r.top100, r.medrank);
    std::fflush(stdout);
  };
  auto tanScore = [&](const Fps& F, const std::vector<float>& p, const Win& w, std::vector<float>& out) {
    out.resize(w.ids.size());
    if (p.empty()) { std::fill(out.begin(), out.end(), 0.f); return; }
    double sumP = 0;
    for (float x : p) sumP += x;
    for (size_t k = 0; k < w.ids.size(); ++k) {
      uint32_t id = w.ids[k];
      const uint16_t* b = F.bits.data() + F.ptr[id];
      int n = int(F.ptr[id + 1] - F.ptr[id]);
      double in = 0;
      for (int j = 0; j < n; ++j) in += p[b[j]];
      out[k] = float(in / (n + sumP - in + 1e-9));
    }
  };

  if (usePop) pr("popularity only", evalScores([&](int qi, std::vector<float>& f) {
                   f.resize(wins[qi].ids.size());
                   for (size_t k = 0; k < f.size(); ++k) f[k] = pop[wins[qi].ids[k]];
                 }));

  auto ks = numList(get("ks", "1,10,50,200")), Ts = numList(get("Ts", "0.01,0.03,0.1,0.3")),
       lams = numList(get("lams", "0,0.2")), sims = numList(get("sims", "0,1,2,3")),
       ensw = numList(get("ensw", "0.25,0.5,1,2"));
  // model scores (tanimoto), per vocab tag
  std::map<std::string, std::vector<std::vector<float>>> modelScore;  // tag -> [q][cand] (z-scored)
  std::map<std::string, std::string> modelOfTag;
  {
    std::stringstream ss(get("models", ""));
    std::string item;
    while (std::getline(ss, item, ',')) {
      size_t c = item.find(':');
      if (c == std::string::npos) continue;
      std::string path = item.substr(0, c), tag = item.substr(c + 1);
      Model m;
      if (!m.load(dir + "/" + path)) { std::fprintf(stderr, "cannot load %s\n", path.c_str()); return 1; }
      Fps F = loadFps(dir + "/" + win + "_" + tag + ".bin");
      auto P = predictByMol(m, q);
      auto& S = modelScore[tag];
      S.assign(nQ, {});
      ::parallel_for(nQ, [&](int64_t qi) { tanScore(F, P[wins[qi].mol], wins[qi], S[qi]); zs(S[qi]); });
      modelOfTag[tag] = path;
      std::printf(" model %s (%s) tanimoto:\n", path.c_str(), tag.c_str());
      pr("model", evalScores([&](int qi, std::vector<float>& f) { f = S[qi]; }));
      if (usePop) pr("model + pop", evalScores([&](int qi, std::vector<float>& f) {
                       f = S[qi];
                       for (size_t k = 0; k < f.size(); ++k) f[k] += popW * pop[wins[qi].ids[k]];
                     }));
    }
    if (modelScore.size() > 1) {
      std::printf(" models z-sum:\n");
      pr("models z-sum", evalScores([&](int qi, std::vector<float>& f) {
           f.assign(wins[qi].ids.size(), 0.f);
           for (auto& kv : modelScore) for (size_t k = 0; k < f.size(); ++k) f[k] += kv.second[qi][k];
         }));
    }
  }

  std::stringstream vs(get("vocabs", "v4k"));
  std::string tag;
  while (std::getline(vs, tag, ',')) {
    Fps LF = loadFps(dir + "/" + lib + "_" + tag + ".bin");
    Fps CF = loadFps(dir + "/" + win + "_" + tag + ".bin");
    const int NB = int(LF.nBits);
    std::vector<float> freq(NB, 0.f);
    for (uint32_t m = 0; m < LF.n; ++m)
      for (uint64_t j = LF.ptr[m]; j < LF.ptr[m + 1]; ++j) freq[LF.bits[j]] += 1.f;
    for (auto& x : freq) x /= float(std::max<uint32_t>(1, LF.n));
    std::printf("[%s] vocab %s: %d bits, library %u molecules (%.0fs)\n", qry.c_str(), tag.c_str(), NB, LF.n, el());
    auto predict = [&](int ty, int mol, int k, double T, double lam, std::vector<float>& p) {
      p.clear();
      const auto& v = nbr[ty][mol];
      if (v.empty()) { p = freq; return; }
      int K = std::min<int>(k, int(v.size()));
      std::vector<double> w(K);
      double s = 0;
      for (int j = 0; j < K; ++j) { w[j] = std::exp((double(v[j].sim) - v[0].sim) / T); s += w[j]; }
      p.assign(NB, 0.f);
      for (int j = 0; j < K; ++j) {
        float wj = float(w[j] / s * (1.0 - lam));
        int lm = v[j].mol;
        if (lm < 0 || uint32_t(lm) >= LF.n) continue;
        for (uint64_t b = LF.ptr[lm]; b < LF.ptr[lm + 1]; ++b) p[LF.bits[b]] += wj;
      }
      if (lam > 0) for (int b = 0; b < NB; ++b) p[b] += float(lam) * freq[b];
    };
    if (get("diag", "0") == "1") {
      // how close are the retrieved molecules to the truth? binary Tanimoto of neighbour j vs the truth's bits
      for (int ty = 0; ty < 4; ++ty) {
        int nq = 0, n1 = 0, nAny1 = 0, nSat = 0;
        double t1 = 0, tmax = 0, nsat1 = 0;
        for (int qi = 0; qi < nQ; ++qi) {
          const Win& w = wins[qi];
          if (w.truth < 0) continue;
          const auto& v = nbr[ty][w.mol];
          if (v.empty()) continue;
          uint32_t tid = w.ids[w.truth];
          std::vector<char> tb(NB, 0);
          int nt = 0;
          for (uint64_t j = CF.ptr[tid]; j < CF.ptr[tid + 1]; ++j) { tb[CF.bits[j]] = 1; ++nt; }
          double best = 0, first = 0;
          int K = std::min<int>(10, int(v.size()));
          for (int j = 0; j < K; ++j) {
            int lm = v[j].mol, inter = 0, nl = 0;
            for (uint64_t b = LF.ptr[lm]; b < LF.ptr[lm + 1]; ++b) { inter += tb[LF.bits[b]]; ++nl; }
            double t = double(inter) / std::max(1, nt + nl - inter);
            if (j == 0) first = t;
            best = std::max(best, t);
          }
          int sat = 0;
          for (auto& x : v) sat += x.sim >= 0.999f;
          ++nq; t1 += first; tmax += best; n1 += first >= 0.999; nAny1 += best >= 0.999; nSat += sat > 0; nsat1 += sat;
          if (qi < 8) {
            std::printf("   diag %s q%d:", SIMN[ty], qi);
            for (int j = 0; j < std::min<int>(5, int(v.size())); ++j) std::printf(" %.3f", v[j].sim);
            std::printf(" | top1 Tan %.3f best10 %.3f, sims>=0.999: %d\n", first, best, sat);
          }
        }
        std::printf("  diag %s %s: %d q, top1 Tanimoto-to-truth mean %.3f (==1: %d), best of top10 mean %.3f (==1: %d), "
                    "queries with a sim>=0.999 neighbour %d (mean count %.1f)\n",
                    tag.c_str(), SIMN[ty], nq, t1 / nq, n1, tmax / nq, nAny1, nSat, nsat1 / nq);
      }
      std::fflush(stdout);
      if (get("diagonly", "0") == "1") continue;
    }
    struct Best { double mrr = -1; int ty = 0, k = 0; double T = 0, lam = 0; };
    Best best;
    for (double tyd : sims) {
      int ty = int(tyd);
      for (double kd : ks)
        for (double T : Ts)
          for (double lam : lams) {
            int k = int(kd);
            if (k == 1 && (T != Ts[0] || lam != lams[0])) continue;
            char lab[160];
            std::snprintf(lab, sizeof lab, "attn %s %s k=%d T=%.3g lam=%.2g", tag.c_str(), SIMN[ty], k, T, lam);
            R r = evalScores([&](int qi, std::vector<float>& f) {
              std::vector<float> p;
              predict(ty, wins[qi].mol, k, T, lam, p);
              tanScore(CF, p, wins[qi], f);
            });
            pr(lab, r);
            if (r.in > best.mrr) best = {r.in, ty, k, T, lam};
          }
    }
    std::printf(" best attn %s: %s k=%d T=%.3g lam=%.2g MRR_in %.4f\n", tag.c_str(), SIMN[best.ty], best.k, best.T,
                best.lam, best.mrr);
    // configs for controls / ensembles: best + a fixed reference config
    std::vector<Best> cfgs = {best};
    if (A.count("refcfg")) {  // refcfg=ty:k:T:lam
      std::string s = A["refcfg"];
      for (auto& ch : s) if (ch == ':') ch = ',';
      auto v = numList(s);
      if (v.size() == 4) cfgs.push_back({0, int(v[0]), int(v[1]), v[2], v[3]});
    }
    std::vector<std::vector<std::vector<float>>> cAS, cASH;
    for (auto& c : cfgs) {
      char base[160];
      std::snprintf(base, sizeof base, "%s %s k=%d T=%.3g lam=%.2g", tag.c_str(), SIMN[c.ty], c.k, c.T, c.lam);
      std::printf(" config %s\n", base);
      std::vector<std::vector<float>> AS(nQ), ASH(nQ);
      ::parallel_for(nQ, [&](int64_t qi) {
        std::vector<float> p;
        predict(c.ty, wins[qi].mol, c.k, c.T, c.lam, p);
        tanScore(CF, p, wins[qi], AS[qi]); zs(AS[qi]);
        predict(c.ty, wins[(qi + shift) % nQ].mol, c.k, c.T, c.lam, p);
        tanScore(CF, p, wins[qi], ASH[qi]); zs(ASH[qi]);
      });
      cAS.push_back(AS); cASH.push_back(ASH);
      pr("attn", evalScores([&](int qi, std::vector<float>& f) { f = AS[qi]; }));
      pr("SHUFFLED attn", evalScores([&](int qi, std::vector<float>& f) { f = ASH[qi]; }));
      if (usePop) pr("attn + pop", evalScores([&](int qi, std::vector<float>& f) {
                       f = AS[qi];
                       for (size_t k = 0; k < f.size(); ++k) f[k] += popW * pop[wins[qi].ids[k]];
                     }));
      // ensembles: each model alone + w*attn, and all models + w*attn
      std::vector<std::pair<std::string, std::vector<std::string>>> combos;
      for (auto& kv : modelScore) combos.push_back({kv.second.empty() ? "" : modelOfTag[kv.first], {kv.first}});
      if (modelScore.size() > 1) {
        std::vector<std::string> allT;
        for (auto& kv : modelScore) allT.push_back(kv.first);
        combos.push_back({"all models", allT});
      }
      for (auto& cb : combos)
        for (double w : ensw)
          for (int sh = 0; sh < 2; ++sh) {
            auto& src = sh ? ASH : AS;
            char lab[200];
            std::snprintf(lab, sizeof lab, "%s%s + %.2g attn", sh ? "SHUF " : "", cb.first.c_str(), w);
            pr(lab, evalScores([&](int qi, std::vector<float>& f) {
                 f.assign(wins[qi].ids.size(), 0.f);
                 for (auto& t2 : cb.second)
                   for (size_t k = 0; k < f.size(); ++k) f[k] += modelScore[t2][qi][k];
                 for (size_t k = 0; k < f.size(); ++k) f[k] += float(w) * src[qi][k];
               }));
            if (usePop && !sh) {
              std::snprintf(lab, sizeof lab, "%s + %.2g attn + pop", cb.first.c_str(), w);
              pr(lab, evalScores([&](int qi, std::vector<float>& f) {
                   f.assign(wins[qi].ids.size(), 0.f);
                   for (auto& t2 : cb.second)
                     for (size_t k = 0; k < f.size(); ++k) f[k] += modelScore[t2][qi][k];
                   for (size_t k = 0; k < f.size(); ++k) f[k] += float(w) * src[qi][k] + popW * pop[wins[qi].ids[k]];
                 }));
            }
          }
    }
    if (cAS.size() >= 2) {
      std::printf(" two-attention z-sum: [%s k=%d T=%.3g] + w [%s k=%d T=%.3g]\n", SIMN[cfgs[0].ty], cfgs[0].k, cfgs[0].T,
                  SIMN[cfgs[1].ty], cfgs[1].k, cfgs[1].T);
      std::vector<std::string> allT;
      for (auto& kv : modelScore) allT.push_back(kv.first);
      for (int withModels = 0; withModels < 2; ++withModels) {
        if (withModels && allT.empty()) continue;
        for (double w : ensw)
          for (int sh = 0; sh < 2; ++sh) {
            char lab[200];
            std::snprintf(lab, sizeof lab, "%s%sattn1 + %.2g attn2", sh ? "SHUF(attn1) " : "", withModels ? "models + " : "", w);
            auto run = [&](bool addPop) {
              return evalScores([&](int qi, std::vector<float>& f) {
                f.assign(wins[qi].ids.size(), 0.f);
                const auto& a1 = sh ? cASH[0][qi] : cAS[0][qi];
                for (size_t k = 0; k < f.size(); ++k) f[k] = a1[k] + float(w) * cAS[1][qi][k];
                if (withModels) for (auto& t2 : allT) for (size_t k = 0; k < f.size(); ++k) f[k] += modelScore[t2][qi][k];
                if (addPop) for (size_t k = 0; k < f.size(); ++k) f[k] += popW * pop[wins[qi].ids[k]];
              });
            };
            pr(lab, run(false));
            if (usePop && !sh) { std::string l2 = std::string(lab) + " + pop"; pr(l2, run(true)); }
          }
      }
    }
  }
  std::printf("all done (%.0fs)\n", el());
  return 0;
}
