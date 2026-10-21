// win_dump: the current best window score (fp3 + attention, attn_STATUS.md) for every
// PubChem-window candidate, and a dump of each query's top N candidates by that score,
// for re-ranking by formula_eval (formula_STATUS.md).
//
// Scores are exactly attn_eval.cpp's: z_win(Tanimoto(fp3 model p, cand bits)) +
// z_win(Tanimoto(attention p, cand bits)), attention = ment k=10 T=0.1 lam=0 over
// fp3c16k bits, neighbours from attn_eval's cache (mpen=4). Also the shuffled
// attention (query q from q+shift's neighbours) and popularity.
// CPU build (no GPU needed; the fp3 forward pass for ~1k spectra takes seconds).
//
//   win_dump dir=C:\prog\casmi qry=hcqry [lib=wlib] [model=rev3_fp3c16k_big.bin] [vocab=fp3c16k]
//            [N=1000] [out=formula\<qry>_top.bin] [threads=4]
// Output: uint32 nQ, then per query: int32 mol, int32 truthWinIdx, int32 truthRankBase (pessimistic,
//   0 = not in window), uint32 winSize, uint32 n, then n x {uint32 widx, uint32 uid, float zfp3,
//   float zatt, float zattShuf, float pop}, sorted by zfp3 + zatt descending.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
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
struct Nbr { float sim; int mol; };
}  // namespace

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    size_t e = s.find('=');
    if (e != std::string::npos) A[s.substr(0, e)] = s.substr(e + 1);
  }
  std::string dir = get("dir", "C:\\prog\\casmi"), qry = get("qry", "hcqry"), win = get("win", "w_" + qry),
              lib = get("lib", "wlib"), vocab = get("vocab", "fp3c16k");
  int shift = std::atoi(get("shift", "7").c_str()), N = std::atoi(get("N", "1000").c_str());
  int ty = std::atoi(get("ty", "3").c_str()), K = std::atoi(get("k", "10").c_str());
  double T = std::atof(get("T", "0.1").c_str());
  parallel_cap() = unsigned(std::atoi(get("threads", "4").c_str()));
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
  // neighbours (attn_eval cache, mpen=4)
  std::string nbrPath = dir + "/" + get("nbrfile", "attn_" + qry + "_" + lib + "_v2_t0.01_p40_d0_m4_nbr.bin");
  std::vector<std::vector<std::vector<Nbr>>> nbr(4, std::vector<std::vector<Nbr>>(q.nMol));
  {
    std::ifstream f(nbrPath, std::ios::binary);
    if (!f) { std::fprintf(stderr, "no neighbour cache %s (run attn_eval first)\n", nbrPath.c_str()); return 1; }
    uint32_t nm, KK; f.read((char*)&nm, 4); f.read((char*)&KK, 4);
    if (nm != q.nMol) { std::fprintf(stderr, "neighbour cache mismatch\n"); return 1; }
    for (int t = 0; t < 4; ++t)
      for (uint32_t m = 0; m < nm; ++m) {
        uint32_t c; f.read((char*)&c, 4);
        nbr[t][m].resize(c);
        f.read((char*)nbr[t][m].data(), sizeof(Nbr) * c);
      }
  }
  Fps LF = loadFps(dir + "/" + lib + "_" + vocab + ".bin");
  Fps CF = loadFps(dir + "/" + win + "_" + vocab + ".bin");
  const int NB = int(LF.nBits);
  std::vector<float> freq(NB, 0.f);
  for (uint32_t m = 0; m < LF.n; ++m)
    for (uint64_t j = LF.ptr[m]; j < LF.ptr[m + 1]; ++j) freq[LF.bits[j]] += 1.f;
  for (auto& x : freq) x /= float(std::max<uint32_t>(1, LF.n));
  auto tanScore = [&](const std::vector<float>& p, const Win& w, std::vector<float>& out) {
    out.resize(w.ids.size());
    if (p.empty()) { std::fill(out.begin(), out.end(), 0.f); return; }
    double sumP = 0;
    for (float x : p) sumP += x;
    for (size_t k = 0; k < w.ids.size(); ++k) {
      uint32_t id = w.ids[k];
      const uint16_t* b = CF.bits.data() + CF.ptr[id];
      int n = int(CF.ptr[id + 1] - CF.ptr[id]);
      double in = 0;
      for (int j = 0; j < n; ++j) in += p[b[j]];
      out[k] = float(in / (n + sumP - in + 1e-9));
    }
  };
  auto attP = [&](int mol, std::vector<float>& p) {
    p.clear();
    const auto& v = nbr[ty][mol];
    if (v.empty()) { p = freq; return; }
    int k = std::min<int>(K, int(v.size()));
    std::vector<double> w(k);
    double s = 0;
    for (int j = 0; j < k; ++j) { w[j] = std::exp((double(v[j].sim) - v[0].sim) / T); s += w[j]; }
    p.assign(NB, 0.f);
    for (int j = 0; j < k; ++j) {
      int lm = v[j].mol;
      if (lm < 0 || uint32_t(lm) >= LF.n) continue;
      for (uint64_t b = LF.ptr[lm]; b < LF.ptr[lm + 1]; ++b) p[LF.bits[b]] += float(w[j] / s);
    }
  };
  Model m;
  if (!m.load(dir + "/" + get("model", "rev3_fp3c16k_big.bin"))) { std::fprintf(stderr, "cannot load model\n"); return 1; }
  auto P = predictByMol(m, q);
  std::printf("%s: %d queries, model predicted (%.0fs)\n", qry.c_str(), nQ, el());
  std::vector<std::vector<float>> S(nQ), AS(nQ), ASH(nQ);
  ::parallel_for(nQ, [&](int64_t qi) {
    tanScore(P[wins[qi].mol], wins[qi], S[qi]); zs(S[qi]);
    std::vector<float> p;
    attP(wins[qi].mol, p); tanScore(p, wins[qi], AS[qi]); zs(AS[qi]);
    attP(wins[(qi + shift) % nQ].mol, p); tanScore(p, wins[qi], ASH[qi]); zs(ASH[qi]);
  });
  // yardstick check (attn_STATUS: hcqry fp3 0.327, fp3 + 1 attn 0.454) and truth recall at N
  const int cuts[] = {25, 50, 100, 200, 300, 500, 1000, 2000};
  double mrr[4] = {0, 0, 0, 0};
  int nIn = 0, rec[8] = {0};
  std::vector<int> baseRank(nQ, 0);
  for (int qi = 0; qi < nQ; ++qi) {
    if (wins[qi].truth < 0) continue;
    ++nIn;
    std::vector<float> f[4];
    f[0] = S[qi]; f[1] = AS[qi]; f[2] = S[qi]; f[3] = S[qi];
    for (size_t k = 0; k < f[2].size(); ++k) { f[2][k] += AS[qi][k]; f[3][k] += ASH[qi][k]; }
    for (int v = 0; v < 4; ++v) {
      int r = rankOf(f[v], wins[qi].truth);
      if (v == 2) baseRank[qi] = r;
      if (r <= 25) mrr[v] += 1.0 / r;
    }
    for (int c = 0; c < 8; ++c) rec[c] += baseRank[qi] <= cuts[c];
  }
  std::printf("MRR_in (MRR_all): fp3 %.4f (%.4f)  attn %.4f (%.4f)  fp3+attn %.4f (%.4f)  fp3+SHUFattn %.4f (%.4f)  [nIn %d / %d]\n",
              mrr[0] / nIn, mrr[0] / nQ, mrr[1] / nIn, mrr[1] / nQ, mrr[2] / nIn, mrr[2] / nQ, mrr[3] / nIn, mrr[3] / nQ, nIn, nQ);
  std::printf("truth recall in the top N of fp3+attn (of truth-in-window):");
  for (int c = 0; c < 8; ++c) std::printf("  N=%d %.3f", cuts[c], double(rec[c]) / nIn);
  std::printf("\n");
  std::string out = get("out", dir + "/formula/" + qry + "_top.bin");
  std::ofstream f(out, std::ios::binary);
  uint32_t nq = uint32_t(nQ);
  f.write((char*)&nq, 4);
  for (int qi = 0; qi < nQ; ++qi) {
    const Win& w = wins[qi];
    std::vector<int> ord(w.ids.size());
    std::iota(ord.begin(), ord.end(), 0);
    int n = std::min<int>(N, int(ord.size()));
    std::partial_sort(ord.begin(), ord.begin() + n, ord.end(),
                      [&](int a, int b) { return S[qi][a] + AS[qi][a] > S[qi][b] + AS[qi][b]; });
    int32_t hd[3] = {w.mol, w.truth, w.truth >= 0 ? baseRank[qi] : 0};
    uint32_t ws = uint32_t(w.ids.size()), nn = uint32_t(n);
    f.write((char*)hd, 12); f.write((char*)&ws, 4); f.write((char*)&nn, 4);
    for (int j = 0; j < n; ++j) {
      int k = ord[j];
      uint32_t u[2] = {uint32_t(k), w.ids[k]};
      float v[4] = {S[qi][k], AS[qi][k], ASH[qi][k], pop[w.ids[k]]};
      f.write((char*)u, 8); f.write((char*)v, 16);
    }
  }
  std::printf("wrote %s (%.0fs)\n", out.c_str(), el());
  return 0;
}
