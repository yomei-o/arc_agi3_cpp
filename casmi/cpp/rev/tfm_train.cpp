// Train / evaluate / gradient-check the spectrum -> fingerprint Transformer (tfm_model.hpp).
//
//   tfm_train gradcheck [drop=0.2]               finite differences on a tiny config (exact in a -DTFM_DOUBLE CPU build)
//   tfm_train train key=value...
//     dir=C:\prog\casmi lib=xlib fp=v4k out=<stem>   (writes <stem>.bin per epoch end, <stem>_ck.bin checkpoints)
//     d=256 L=4 H=8 ff=4 demb=<d> HH=2048 maxPeaks=128 epochs=6 B=256 lr=3e-4 wd=0.05 drop=0.1 clip=1
//     warm=2000 ls=0 seed=31 hold=20 tf32=1 ckptEvery=2000 stopAfter=0(min) resume=0 valMax=20000
//     eval=cqry:f21_cqry.txt,tqry:f21_tqry.txt cand=fcand
//   tfm_train eval models=a.bin[,b.bin] fp=v4k [mlp=rev2_vh.bin: z-sum combination grid] [lib=xlib hold=20 for held-out BCE/F1]
//
// hold=K leaves out every molecule whose InChIKey14 hashes to 0 mod K (same split as rev2_train).
// The fused evaluation is rev2_train's: only the reverse-model column of the phys_ana feature dumps
// (f21_*.txt) is recomputed and fused as z(phys) + 0.25 z(mlp) + wr z(rev) + 8 lib^32 + 24 anaP.
// Molecule probability = mean over its spectra of sigmoid(logit) (as rev2).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "parallel.hpp"
#include "tfm_model.hpp"
#include "rev2_model.hpp"

using tfm::real;

namespace {

struct Head { float precursor; int32_t mol, off, n, tag; };
constexpr float OUR_MODE[10] = {1, -1, 1, 1, 1, -1, -1, 1, 1, 1};

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
  std::vector<std::string> name;
  const float* mzs(const Head& h) const { return mz.data() + h.off; }
  const float* its(const Head& h) const { return inten.data() + h.off; }
};

Store loadSpectra(const std::string& dir, const std::string& stem) {
  Store s;
  std::vector<char> hb = slurp(dir + "/" + stem + "_head.bin");
  s.head.resize(hb.size() / sizeof(Head));
  std::memcpy(s.head.data(), hb.data(), s.head.size() * sizeof(Head));
  hb.clear(); hb.shrink_to_fit();
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
  }
  std::ifstream f(dir + "/" + stem + "_mols.txt");
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    s.name.push_back(line.substr(0, line.find('\t')));
  }
  std::printf("%s: %zu spectra, %zu molecules\n", stem.c_str(), s.head.size(), s.name.size());
  return s;
}

struct Fps { std::vector<uint32_t> ptr; std::vector<uint16_t> bits; uint32_t nBits = 0; };
Fps loadFps(const std::string& path) {
  Fps f;
  std::vector<char> b = slurp(path);
  const char* p = b.data();
  uint32_t n; std::memcpy(&n, p, 4); std::memcpy(&f.nBits, p + 4, 4); p += 8;
  f.ptr.resize(n + 1); f.ptr[0] = 0;
  for (uint32_t i = 0; i < n; ++i) {
    uint16_t c; std::memcpy(&c, p, 2); p += 2;
    size_t old = f.bits.size();
    f.bits.resize(old + c);
    std::memcpy(f.bits.data() + old, p, 2 * size_t(c)); p += 2 * size_t(c);
    f.ptr[i + 1] = uint32_t(f.bits.size());
  }
  std::printf("%s: %u molecules, nBits %u\n", path.c_str(), n, f.nBits);
  return f;
}

bool loadVocab(const std::string& path, tfm::Model& m) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  uint32_t magic, n; int32_t rE, rF;
  f.read(reinterpret_cast<char*>(&magic), 4); f.read(reinterpret_cast<char*>(&rE), 4);
  f.read(reinterpret_cast<char*>(&rF), 4); f.read(reinterpret_cast<char*>(&n), 4);
  if (magic != 0x32565046) return false;
  m.rE = rE; m.rF = rF;
  m.vocab.resize(n);
  f.read(reinterpret_cast<char*>(m.vocab.data()), 8 * size_t(n));
  return true;
}

uint64_t nameHash(const std::string& s) {
  uint64_t h = 1469598103934665603ull;
  for (char c : s) h = (h ^ uint8_t(c)) * 1099511628211ull;
  return h ^ (h >> 29);
}

// ---- peaks -> tokens (FPNet prep_peaks: mz<=prec+2, I/Imax>=1e-3, top-k by intensity, mz order, sqrt)
void prepPeaks(const Head& h, const float* mz, const float* it, int maxP, std::vector<float>& om, std::vector<float>& oi) {
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
  if (om.empty()) { om.push_back(0.f); oi.push_back(0.f); }  // FPNet collate: empty -> one zero peak
}
int countPeaks(const Head& h, const float* mz, const float* it, int maxP) {
  float mx = 0.f;
  for (int i = 0; i < h.n; ++i) if (mz[i] > 0.f && mz[i] <= h.precursor + 2.f) mx = std::max(mx, it[i]);
  if (mx <= 0.f) return 1;
  int c = 0;
  for (int i = 0; i < h.n; ++i) if (mz[i] > 0.f && mz[i] <= h.precursor + 2.f && it[i] / mx >= 1e-3f) ++c;
  return std::max(1, std::min(c, maxP));
}
void decodeTag(int tag, int& ad, int& in, float& mode) {
  int a = -1, s = -1;
  if (tag >= 0) {
    bool unkInstr = tag % 16 == 15;
    a = unkInstr ? tag / 16 + 1 : tag / 16;
    s = unkInstr ? -1 : tag % 16;
  }
  ad = (a >= 0 && a < 10) ? a : 10;
  in = (s >= 0 && s < 5) ? s : 5;
  mode = (a >= 0 && a < 10) ? OUR_MODE[a] : 0.f;
}

void buildBatch(const Store& st, const std::vector<int>& specs, int maxP, tfm::HostBatch& hb) {
  const int B = int(specs.size());
  std::vector<std::vector<float>> M(B), I(B);
  ::parallel_for(B, [&](int64_t b) {
    const Head& h = st.head[specs[b]];
    prepPeaks(h, st.mzs(h), st.its(h), maxP, M[b], I[b]);
  });
  int mx = 1;
  for (auto& v : M) mx = std::max(mx, int(v.size()));
  hb.B = B; hb.N = 1 + mx;
  hb.mz.assign(size_t(B) * mx, 0.f); hb.it.assign(size_t(B) * mx, 0.f);
  hb.prec.resize(B); hb.mode.resize(B); hb.np.resize(B); hb.ad.resize(B); hb.in.resize(B);
  for (int b = 0; b < B; ++b) {
    const Head& h = st.head[specs[b]];
    std::copy(M[b].begin(), M[b].end(), hb.mz.begin() + size_t(b) * mx);
    std::copy(I[b].begin(), I[b].end(), hb.it.begin() + size_t(b) * mx);
    hb.np[b] = int(M[b].size());
    hb.prec[b] = h.precursor;
    decodeTag(h.tag, hb.ad[b], hb.in[b], hb.mode[b]);
  }
}

// logits for a list of spectra (row-major n x NB)
void predictLogits(tfm::Model& m, const Store& st, const std::vector<int>& specs, std::vector<float>& out, int B = 128) {
  out.assign(specs.size() * size_t(m.c.NB), 0.f);
  tfm::HostBatch hb;
  std::vector<real> z;
  // sort by peak count to limit padding
  std::vector<int> ord(specs.size());
  std::iota(ord.begin(), ord.end(), 0);
  std::vector<int> cnt(specs.size());
  for (size_t i = 0; i < specs.size(); ++i) { const Head& h = st.head[specs[i]]; cnt[i] = countPeaks(h, st.mzs(h), st.its(h), m.c.maxPeaks); }
  std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return cnt[a] < cnt[b]; });
  for (size_t s0 = 0; s0 < ord.size(); s0 += B) {
    size_t s1 = std::min(ord.size(), s0 + B);
    std::vector<int> sp;
    for (size_t i = s0; i < s1; ++i) sp.push_back(specs[ord[i]]);
    buildBatch(st, sp, m.c.maxPeaks, hb);
    m.setBatch(hb);
    m.forward(false, 0.f, 0);
    z.resize(size_t(hb.B) * m.c.NB);
    tfm::download(m.Z, z.data(), int64_t(z.size()));
    for (size_t i = s0; i < s1; ++i)
      for (int k = 0; k < m.c.NB; ++k) out[size_t(ord[i]) * m.c.NB + k] = float(z[(i - s0) * m.c.NB + k]);
  }
}

// ---- evaluation on phys_ana feature dumps (as rev2_train) ----------------------
struct Cand { float phys, mlp, rev, lib, ana; int key; };
struct DQ { std::string name; int mol = -1, truth = -1; std::vector<Cand> c; };
struct EvalSet { std::string name; Store q; std::vector<DQ> qs; };

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
double rr(const std::vector<float>& f, int t) {
  if (t < 0) return 0.0;
  int better = 0;
  for (size_t i = 0; i < f.size(); ++i) if (int(i) != t && f[i] >= f[t]) ++better;
  return better < 25 ? 1.0 / (better + 1) : 0.0;
}
struct CandFps { std::unordered_map<std::string, int> keyIdx; Fps fps; };

std::vector<EvalSet> loadEval(const std::string& dir, const std::string& spec, CandFps& cf) {
  std::vector<EvalSet> out;
  std::stringstream ss(spec);
  std::string item;
  while (std::getline(ss, item, ',')) {
    size_t c = item.find(':');
    EvalSet e;
    e.name = item.substr(0, c);
    e.q = loadSpectra(dir, e.name);
    std::unordered_map<std::string, int> molOf;
    for (size_t i = 0; i < e.q.name.size(); ++i) molOf[e.q.name[i]] = int(i);
    std::ifstream f(dir + "/" + item.substr(c + 1));
    if (!f) { std::fprintf(stderr, "cannot read %s\n", item.c_str()); std::exit(1); }
    std::string line;
    int nMiss = 0;
    while (std::getline(f, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line.empty()) continue;
      std::istringstream ls(line);
      if (line[0] == 'Q') {
        std::string tag; DQ q; int n;
        ls >> tag >> q.name >> q.truth >> n;
        auto it = molOf.find(q.name);
        q.mol = it == molOf.end() ? -1 : it->second;
        e.qs.push_back(q);
      } else {
        Cand cd; int inLib; std::string key;
        ls >> cd.phys >> cd.mlp >> cd.rev >> cd.lib >> cd.ana >> inLib >> key;
        auto it = cf.keyIdx.find(key);
        cd.key = it == cf.keyIdx.end() ? -1 : it->second;
        if (cd.key < 0) ++nMiss;
        e.qs.back().c.push_back(cd);
      }
    }
    std::printf("eval %s: %zu queries, %d candidates without fingerprint\n", e.name.c_str(), e.qs.size(), nMiss);
    out.push_back(std::move(e));
  }
  return out;
}

// molecule -> mean probability (and mean logit) over its spectra, per model; averaged over models
struct MolPred { std::vector<float> p, z; };
std::unordered_map<int, MolPred> predictByMol(std::vector<tfm::Model*>& ms, const Store& q, const std::vector<int>& spec) {
  std::unordered_map<int, MolPred> res;
  std::unordered_map<int, int> cnt;
  const int NB = ms[0]->c.NB;
  for (tfm::Model* m : ms) {
    std::vector<float> Z;
    predictLogits(*m, q, spec, Z);
    for (size_t s = 0; s < spec.size(); ++s) {
      int mol = q.head[spec[s]].mol;
      auto& v = res[mol];
      if (v.p.empty()) { v.p.assign(NB, 0.f); v.z.assign(NB, 0.f); }
      for (int i = 0; i < NB; ++i) {
        float z = Z[s * NB + i];
        v.p[i] += 1.f / (1.f + std::exp(-z));
        v.z[i] += z;
      }
      ++cnt[mol];
    }
  }
  for (auto& kv : res) {
    float c = float(cnt[kv.first]);
    for (auto& x : kv.second.p) x /= c;
    for (auto& x : kv.second.z) x /= c;
  }
  return res;
}

const char* VNAME[] = {"logit", "logit_c2", "logp", "tanimoto", "meanlogit"};
constexpr int NV = 5;
struct Scorer {
  std::vector<std::vector<float>> u;
  std::vector<float> p;
  double sumP = 0;
  void prep(const MolPred& mp) {
    p = mp.p;
    const int NB = int(p.size());
    u.assign(NV, std::vector<float>(NB, 0.f));
    sumP = 0;
    for (int b = 0; b < NB; ++b) {
      float pi = std::min(1.f - 1e-6f, std::max(1e-6f, p[b]));
      u[0][b] = std::log(pi / (1.f - pi));
      float pc = std::min(0.99f, std::max(0.01f, p[b]));
      u[1][b] = std::log(pc / (1.f - pc));
      u[2][b] = std::log(pi);
      u[4][b] = mp.z[b];
      sumP += p[b];
    }
  }
  float score(int v, const uint16_t* bits, int n) const {
    if (v == 3) {
      double inter = 0;
      for (int k = 0; k < n; ++k) inter += p[bits[k]];
      return float(inter / (double(n) + sumP - inter + 1e-9));
    }
    float s = 0.f;
    for (int k = 0; k < n; ++k) s += u[v][bits[k]];
    return s;
  }
};

// returns sum over sets of fused MRR (logit, wr=1)
double evaluateSets(std::vector<tfm::Model*>& ms, const Fps& F, std::vector<EvalSet>& sets, bool verbose) {
  double ret = 0;
  for (auto& e : sets) {
    std::vector<int> spec;
    std::vector<char> want(e.q.name.size(), 0);
    for (auto& q : e.qs) if (q.mol >= 0) want[q.mol] = 1;
    for (size_t s = 0; s < e.q.head.size(); ++s)
      if (e.q.head[s].mol >= 0 && want[e.q.head[s].mol] && e.q.head[s].n > 0) spec.push_back(int(s));
    auto probs = predictByMol(ms, e.q, spec);
    std::vector<std::vector<std::vector<float>>> rs(NV, std::vector<std::vector<float>>(e.qs.size()));
    for (size_t qi = 0; qi < e.qs.size(); ++qi) {
      const DQ& q = e.qs[qi];
      for (int v = 0; v < NV; ++v) rs[v][qi].assign(q.c.size(), 0.f);
      auto it = probs.find(q.mol);
      if (it == probs.end()) continue;
      Scorer sc; sc.prep(it->second);
      for (int v = 0; v < NV; ++v)
        for (size_t ci = 0; ci < q.c.size(); ++ci) {
          int k = q.c[ci].key;
          if (k < 0) continue;
          rs[v][qi][ci] = sc.score(v, F.bits.data() + F.ptr[k], int(F.ptr[k + 1] - F.ptr[k]));
        }
    }
    auto fused = [&](int v, float wr) {
      double tot = 0;
      for (size_t qi = 0; qi < e.qs.size(); ++qi) {
        const DQ& q = e.qs[qi];
        size_t n = q.c.size();
        std::vector<float> ph(n), ml(n), rv(n);
        for (size_t i = 0; i < n; ++i) { ph[i] = q.c[i].phys; ml[i] = q.c[i].mlp; rv[i] = v < 0 ? q.c[i].rev : rs[v][qi][i]; }
        auto zp = zscore(ph), zm = zscore(ml), zr = zscore(rv);
        std::vector<float> f(n);
        for (size_t i = 0; i < n; ++i)
          f[i] = (wr < 0 ? zr[i] : zp[i] + 0.25f * zm[i] + wr * zr[i] + 8.f * std::pow(q.c[i].lib, 32.f) + 24.f * q.c[i].ana);
        tot += rr(f, q.truth);
      }
      return tot / std::max<size_t>(1, e.qs.size());
    };
    const float WR[] = {0.5f, 1.f, 1.5f, 2.f, 3.f};
    if (verbose)
      std::printf("  [%s] dump rev column: alone %.4f fused(wr=1) %.4f | wr=0 %.4f\n", e.name.c_str(), fused(-1, -1.f),
                  fused(-1, 1.f), fused(-1, 0.f));
    for (int v = 0; v < NV; ++v) {
      if (!verbose && v > 0) break;
      double alone = 0;
      for (size_t qi = 0; qi < e.qs.size(); ++qi) alone += rr(rs[v][qi], e.qs[qi].truth);
      std::printf("  [%s] %-9s alone %.4f  fused wr:", e.name.c_str(), VNAME[v], alone / std::max<size_t>(1, e.qs.size()));
      for (float wr : WR) {
        double x = fused(v, wr);
        std::printf(" %g=%.4f", wr, x);
        if (v == 0 && wr == 1.f) ret += x;
      }
      std::printf("\n");
    }
  }
  std::fflush(stdout);
  return ret;
}

// rev2 MLP: molecule -> mean probability over its spectra (as rev2_train predictByMol)
std::unordered_map<int, std::vector<float>> mlpByMol(rev2::Model& m, const Store& q, const std::vector<int>& spec) {
  std::unordered_map<int, std::vector<float>> sum;
  std::unordered_map<int, int> cnt;
  const int B = 256;
  std::vector<std::vector<std::pair<int, float>>> rows;
  std::vector<float> P;
  for (size_t s0 = 0; s0 < spec.size(); s0 += B) {
    size_t s1 = std::min(spec.size(), s0 + B);
    rows.assign(s1 - s0, {});
    for (size_t i = s0; i < s1; ++i) {
      const Head& h = q.head[spec[i]];
      rev2::featurize(m.fc, reinterpret_cast<const rev2::Head&>(h), q.mzs(h), q.its(h), rows[i - s0]);
    }
    m.setBatch(rows, false);
    m.forward(int(s1 - s0), false, 0.f, 0);
    rev2::download(m.P, P, int64_t(s1 - s0) * m.NB);
    for (size_t s = s0; s < s1; ++s) {
      int mol = q.head[spec[s]].mol;
      auto& v = sum[mol];
      if (v.empty()) v.assign(m.NB, 0.f);
      for (int i = 0; i < m.NB; ++i) v[i] += P[(s - s0) * m.NB + i];
      ++cnt[mol];
    }
  }
  for (auto& kv : sum) for (auto& x : kv.second) x /= float(cnt[kv.first]);
  return sum;
}

// rev column = z(TFM score variant v) + a * z(MLP logit score); fused MRR for a grid of (a, wr)
void evalCombo(std::vector<tfm::Model*>& ms, rev2::Model& mlp, const Fps& F, std::vector<EvalSet>& sets) {
  const float AS[] = {0.f, 0.25f, 0.5f, 0.75f, 1.f, 1.5f, 100.f};
  const float WR[] = {1.f, 1.5f, 2.f, 2.5f, 3.f};
  for (auto& e : sets) {
    std::vector<int> spec;
    std::vector<char> want(e.q.name.size(), 0);
    for (auto& q : e.qs) if (q.mol >= 0) want[q.mol] = 1;
    for (size_t s = 0; s < e.q.head.size(); ++s)
      if (e.q.head[s].mol >= 0 && want[e.q.head[s].mol] && e.q.head[s].n > 0) spec.push_back(int(s));
    auto pt = predictByMol(ms, e.q, spec);
    auto pm = mlpByMol(mlp, e.q, spec);
    for (int v : {0, 3}) {
      std::vector<std::vector<float>> zt(e.qs.size()), zm(e.qs.size());
      for (size_t qi = 0; qi < e.qs.size(); ++qi) {
        const DQ& q = e.qs[qi];
        std::vector<float> a(q.c.size(), 0.f), b(q.c.size(), 0.f);
        auto it = pt.find(q.mol); auto im = pm.find(q.mol);
        if (it != pt.end() && im != pm.end()) {
          Scorer sc; sc.prep(it->second);
          MolPred mp; mp.p = im->second; mp.z.assign(mp.p.size(), 0.f);
          Scorer sm; sm.prep(mp);
          for (size_t ci = 0; ci < q.c.size(); ++ci) {
            int k = q.c[ci].key;
            if (k < 0) continue;
            const uint16_t* bits = F.bits.data() + F.ptr[k]; int n = int(F.ptr[k + 1] - F.ptr[k]);
            a[ci] = sc.score(v, bits, n); b[ci] = sm.score(0, bits, n);
          }
        }
        zt[qi] = zscore(a); zm[qi] = zscore(b);
      }
      for (float al : AS) {
        std::printf("  [%s] combo %s + %g*z(mlp):", e.name.c_str(), VNAME[v], al);
        for (float wr : WR) {
          double tot = 0;
          for (size_t qi = 0; qi < e.qs.size(); ++qi) {
            const DQ& q = e.qs[qi];
            size_t n = q.c.size();
            std::vector<float> ph(n), ml(n), rv(n);
            for (size_t i = 0; i < n; ++i) {
              ph[i] = q.c[i].phys; ml[i] = q.c[i].mlp;
              rv[i] = al >= 100.f ? zm[qi][i] : zt[qi][i] + al * zm[qi][i];
            }
            auto zp = zscore(ph), zmm = zscore(ml), zr = zscore(rv);
            std::vector<float> f(n);
            for (size_t i = 0; i < n; ++i)
              f[i] = zp[i] + 0.25f * zmm[i] + wr * zr[i] + 8.f * std::pow(q.c[i].lib, 32.f) + 24.f * q.c[i].ana;
            tot += rr(f, q.truth);
          }
          std::printf(" wr%g=%.4f", wr, tot / std::max<size_t>(1, e.qs.size()));
        }
        std::printf("\n");
      }
    }
  }
  std::fflush(stdout);
}

// held-out spectra: mean BCE per spectrum (sum over bits), micro F1 at p>=0.5, and per-bit-macro AUC-free stats
void validate(tfm::Model& m, const Store& st, const Fps& fps, const std::vector<int>& val) {
  if (val.empty()) return;
  std::vector<float> Z;
  predictLogits(m, st, val, Z);
  const int NB = m.c.NB;
  double bce = 0; long long tp = 0, fp = 0, fn = 0;
  std::vector<uint8_t> t(NB);
  for (size_t s = 0; s < val.size(); ++s) {
    std::fill(t.begin(), t.end(), 0);
    int mol = st.head[val[s]].mol;
    for (uint32_t k = fps.ptr[mol]; k < fps.ptr[mol + 1]; ++k) t[fps.bits[k]] = 1;
    for (int b = 0; b < NB; ++b) {
      float z = Z[s * NB + b];
      // stable BCE with logits
      bce += std::max(z, 0.f) - z * t[b] + std::log1p(std::exp(-std::fabs(z)));
      bool on = z >= 0.f;
      tp += on && t[b]; fp += on && !t[b]; fn += !on && t[b];
    }
  }
  std::printf("  [val] %zu held-out spectra: BCE/spectrum %.3f  micro-F1 %.4f (P %.4f R %.4f)\n", val.size(),
              bce / val.size(), 2.0 * tp / std::max<double>(1, 2.0 * tp + fp + fn), double(tp) / std::max<long long>(1, tp + fp),
              double(tp) / std::max<long long>(1, tp + fn));
  std::fflush(stdout);
}

std::map<std::string, std::string> parseArgs(int argc, char** argv) {
  std::map<std::string, std::string> a;
  for (int i = 2; i < argc; ++i) {
    std::string s = argv[i];
    size_t e = s.find('=');
    if (e != std::string::npos) a[s.substr(0, e)] = s.substr(e + 1);
  }
  return a;
}
std::string get(std::map<std::string, std::string>& a, const std::string& k, const std::string& d) {
  auto it = a.find(k);
  return it == a.end() ? d : it->second;
}
int geti(std::map<std::string, std::string>& a, const std::string& k, int d) { return std::atoi(get(a, k, std::to_string(d)).c_str()); }
float getf(std::map<std::string, std::string>& a, const std::string& k, float d) {
  auto it = a.find(k);
  return it == a.end() ? d : float(std::atof(it->second.c_str()));
}

// ---- gradient check ---------------------------------------------------------
double lossHost(tfm::Model& m, const std::vector<real>& T) {
  std::vector<real> z(size_t(m.B) * m.c.NB);
  tfm::download(m.Z, z.data(), int64_t(z.size()));
  double l = 0;
  for (size_t i = 0; i < z.size(); ++i) {
    double zz = double(z[i]);
    l += std::max(zz, 0.0) - zz * double(T[i]) + std::log1p(std::exp(-std::fabs(zz)));
  }
  return l / m.B;
}

int gradcheck(std::map<std::string, std::string>& A) {
  tfm::Model m;
  m.c.d = 8; m.c.L = 2; m.c.H = 2; m.c.ff = 2; m.c.demb = 6; m.c.HH = 10; m.c.NB = 7; m.c.maxPeaks = 6;
  m.c.nAd = 3; m.c.nIn = 2;
  const float drop = getf(A, "drop", 0.2f);
  const double eps = std::is_same<real, double>::value ? 1e-6 : 3e-3;
  tfm::tf32Flag() = false;
  m.init(5, {});
  // perturb every parameter (so LN gains/biases etc. are not at symmetric points)
  {
    std::vector<real> h(m.nP);
    tfm::download(m.W, h.data(), m.nP);
    uint64_t s = 99;
    std::vector<real> r(m.nP);
    tfm::Model::gauss(r, 0, m.nP, 0.3, s);
    for (int64_t i = 0; i < m.nP; ++i) h[i] += r[i];
    tfm::upload(m.W, h.data(), m.nP);
  }
  tfm::HostBatch hb;
  hb.B = 3; int np[3] = {1, 4, 6}; hb.N = 7;
  hb.mz.assign(3 * 6, 0.f); hb.it.assign(3 * 6, 0.f);
  uint64_t s = 1234;
  auto rnd = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return float((s >> 11) * (1.0 / 9007199254740992.0)); };
  for (int b = 0; b < 3; ++b) {
    hb.prec.push_back(200.f + 300.f * rnd());
    for (int j = 0; j < np[b]; ++j) { hb.mz[b * 6 + j] = 30.f + 170.f * rnd(); hb.it[b * 6 + j] = 0.1f + 0.9f * rnd(); }
    hb.np.push_back(np[b]); hb.ad.push_back(b % 3); hb.in.push_back(b % 2); hb.mode.push_back(b == 1 ? -1.f : 1.f);
  }
  std::vector<real> T(size_t(3) * m.c.NB);
  for (auto& t : T) t = rnd() < 0.4f ? 1 : 0;
  m.setBatch(hb);
  tfm::upload(m.T, T.data(), int64_t(T.size()));
  const uint64_t seed = 77;
  m.forward(true, drop, seed);
  double L0 = lossHost(m, T);
  m.lossGrad(0.f);
  m.backward(drop, seed);
  std::vector<real> G(m.nP), W(m.nP);
  tfm::download(m.G, G.data(), m.nP);
  tfm::download(m.W, W.data(), m.nP);
  std::printf("gradcheck: real=%s, nP=%lld, loss %.6f, drop %.2f, eps %g\n", sizeof(real) == 8 ? "double" : "float",
              (long long)m.nP, L0, drop, eps);
  double worst = 0;
  for (tfm::PRef* p : m.plist) {
    double maxRel = 0, maxAbs = 0, gmax = 0;
    int64_t nCheck = std::min<int64_t>(p->n, 40);
    for (int64_t k = 0; k < nCheck; ++k) {
      int64_t i = p->off + (p->n <= 40 ? k : (k * 7919) % p->n);
      real old = W[i];
      real v = real(old + eps);
      tfm::upload(m.W, &v, 1, i);
      m.forward(true, drop, seed);
      double lp = lossHost(m, T);
      v = real(old - eps);
      tfm::upload(m.W, &v, 1, i);
      m.forward(true, drop, seed);
      double lm = lossHost(m, T);
      tfm::upload(m.W, &old, 1, i);
      double num = (lp - lm) / (2 * eps), an = double(G[i]);
      double ad = std::fabs(num - an);
      maxAbs = std::max(maxAbs, ad);
      gmax = std::max(gmax, std::fabs(an));
      maxRel = std::max(maxRel, ad / std::max(1e-6, std::fabs(num) + std::fabs(an)));
    }
    // relative to the tensor's gradient scale (tiny individual entries are noise-dominated in float)
    double relScale = maxAbs / std::max(1e-12, gmax);
    worst = std::max(worst, relScale);
    std::printf("  %-6s %3dx%-4d checked %2lld  max|g| %.3e  max|num-an| %.3e  maxRel(entry) %.2e  rel(tensor) %.2e\n", p->name,
                p->rows, p->cols, (long long)nCheck, gmax, maxAbs, maxRel, relScale);
  }
  std::printf("gradcheck worst rel(tensor) = %.3e  -> %s\n", worst,
              worst < (sizeof(real) == 8 ? 1e-6 : 3e-2) ? "PASS" : "CHECK");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: tfm_train gradcheck|train|eval key=value...\n"); return 1; }
  std::string mode = argv[1];
  auto A = parseArgs(argc, argv);
  std::printf("backend: %s, real=%s\n", bk::backend_name(), sizeof(real) == 8 ? "double" : "float");
  if (mode == "gradcheck") return gradcheck(A);

  std::string dir = get(A, "dir", "C:\\prog\\casmi");
  std::string candStem = get(A, "cand", "fcand");
  std::string evalSpec = get(A, "eval", "cqry:f21_cqry.txt,tqry:f21_tqry.txt");
  std::string fp = get(A, "fp", "v4k");
  tfm::tf32Flag() = geti(A, "tf32", 1) != 0;

  CandFps cf;
  {
    std::ifstream f(dir + "/" + candStem + "_mols.txt");
    std::string line;
    int i = 0;
    while (std::getline(f, line)) cf.keyIdx[line.substr(0, line.find('\t'))] = i++;
  }
  cf.fps = loadFps(dir + "/" + candStem + "_" + fp + ".bin");

  if (mode == "eval") {
    std::vector<std::unique_ptr<tfm::Model>> owned;
    std::vector<tfm::Model*> ms;
    std::stringstream ss(get(A, "models", ""));
    std::string item;
    while (std::getline(ss, item, ',')) {
      owned.emplace_back(new tfm::Model());
      if (!owned.back()->load(dir + "/" + item, false)) { std::fprintf(stderr, "cannot load %s\n", item.c_str()); return 1; }
      tfm::Model& m = *owned.back();
      std::printf("model %s: d %d L %d H %d ff %d demb %d HH %d NB %d maxPeaks %d, %lld params\n", item.c_str(), m.c.d, m.c.L,
                  m.c.H, m.c.ff, m.c.demb, m.c.HH, m.c.NB, m.c.maxPeaks, (long long)m.nP);
      ms.push_back(&m);
    }
    auto sets = loadEval(dir, evalSpec, cf);
    evaluateSets(ms, cf.fps, sets, true);
    std::string mlpPath = get(A, "mlp", "");
    if (!mlpPath.empty()) {
      rev2::Model mm;
      if (!mm.load(dir + "/" + mlpPath)) { std::fprintf(stderr, "cannot load %s\n", mlpPath.c_str()); return 1; }
      std::printf("combo with MLP %s (%d bits; 'a=100' = MLP alone)\n", mlpPath.c_str(), mm.NB);
      evalCombo(ms, mm, cf.fps, sets);
    }
    int hold = geti(A, "hold", 0);
    if (hold > 0 && ms.size() == 1) {
      Store ls = loadSpectra(dir, get(A, "lib", "xlib"));
      Fps fps = loadFps(dir + "/" + get(A, "lib", "xlib") + "_" + fp + ".bin");
      std::vector<int> val;
      for (size_t s = 0; s < ls.head.size(); ++s) {
        int mol = ls.head[s].mol;
        if (ls.head[s].n > 0 && fps.ptr[mol + 1] > fps.ptr[mol] && nameHash(ls.name[mol]) % uint64_t(hold) == 0) val.push_back(int(s));
      }
      int vmax = geti(A, "valMax", 20000);
      if (int(val.size()) > vmax) {
        std::vector<int> v2;
        for (int i = 0; i < vmax; ++i) v2.push_back(val[size_t(i) * val.size() / vmax]);
        val.swap(v2);
      }
      validate(*ms[0], ls, fps, val);
    }
    return 0;
  }
  if (mode != "train") { std::fprintf(stderr, "unknown mode\n"); return 1; }

  // ------------------------------------------------------------------ train
  std::string lib = get(A, "lib", "xlib"), out = get(A, "out", "tfm");
  const int epochs = geti(A, "epochs", 6), B = geti(A, "B", 256);
  const float lr = getf(A, "lr", 3e-4f), wd = getf(A, "wd", 0.05f), drop = getf(A, "drop", 0.1f), clip = getf(A, "clip", 1.f);
  const float ls = getf(A, "ls", 0.f), b2 = getf(A, "b2", 0.999f);
  const int warm = geti(A, "warm", 2000), hold = geti(A, "hold", 20), ckptEvery = geti(A, "ckptEvery", 2000);
  const uint64_t seed = uint64_t(geti(A, "seed", 31));
  const double stopAfter = getf(A, "stopAfter", 0.f);  // minutes, 0 = run to the end
  const bool resume = geti(A, "resume", 0) != 0;
  const int valMax = geti(A, "valMax", 20000), logEvery = geti(A, "logEvery", 500);

  Store st = loadSpectra(dir, lib);
  Fps fps = loadFps(dir + "/" + lib + "_" + fp + ".bin");
  auto sets = loadEval(dir, evalSpec, cf);

  tfm::Model m;
  const std::string ckPath = dir + "/" + out + "_ck.bin";
  int64_t step0 = 0;
  if (resume) {
    if (!m.load(ckPath, true)) { std::fprintf(stderr, "cannot resume from %s\n", ckPath.c_str()); return 1; }
    step0 = m.adamT;
    std::printf("resumed %s at step %lld\n", ckPath.c_str(), (long long)step0);
  } else {
    m.c.d = geti(A, "d", 256); m.c.L = geti(A, "L", 4); m.c.H = geti(A, "H", 8); m.c.ff = geti(A, "ff", 4);
    m.c.demb = geti(A, "demb", m.c.d); m.c.HH = geti(A, "HH", 2048); m.c.maxPeaks = geti(A, "maxPeaks", 128);
    m.c.NB = int(fps.nBits);
    if (fp != "ecfp" && !loadVocab(dir + "/" + fp + ".bin", m)) std::printf("note: no vocabulary %s.bin\n", fp.c_str());
  }
  if (m.c.NB != int(fps.nBits)) { std::fprintf(stderr, "NB mismatch\n"); return 1; }

  std::vector<int> idx, val;
  for (size_t s = 0; s < st.head.size(); ++s) {
    int mol = st.head[s].mol;
    if (st.head[s].n <= 0 || fps.ptr[mol + 1] == fps.ptr[mol]) continue;
    if (hold > 0 && nameHash(st.name[mol]) % uint64_t(hold) == 0) val.push_back(int(s));
    else idx.push_back(int(s));
  }
  if (int(val.size()) > valMax) {
    std::vector<int> v2;
    for (int i = 0; i < valMax; ++i) v2.push_back(val[size_t(i) * val.size() / valMax]);
    val.swap(v2);
  }
  std::vector<int> npEff(st.head.size(), 1);
  ::parallel_for(int64_t(st.head.size()), [&](int64_t s) {
    const Head& h = st.head[s];
    npEff[s] = countPeaks(h, st.mzs(h), st.its(h), m.c.maxPeaks);
  });
  {
    double mean = 0; int full = 0;
    for (int s : idx) { mean += npEff[s]; full += npEff[s] == m.c.maxPeaks; }
    std::printf("training spectra %zu, val spectra %zu; mean tokens %.1f, %.1f%% capped at %d peaks\n", idx.size(), val.size(),
                mean / idx.size(), 100.0 * full / idx.size(), m.c.maxPeaks);
  }
  if (!resume) {
    // output bias = logit of the bit base rate over training molecules
    std::vector<double> cnt(m.c.NB, 0.0);
    double nm = 0;
    for (int s : idx) {
      int mol = st.head[s].mol;
      for (uint32_t k = fps.ptr[mol]; k < fps.ptr[mol + 1]; ++k) cnt[fps.bits[k]] += 1;
      nm += 1;
    }
    std::vector<float> bias(m.c.NB);
    for (int b = 0; b < m.c.NB; ++b) {
      double p = std::min(0.999, std::max(1e-5, cnt[b] / nm));
      bias[b] = float(std::log(p / (1 - p)));
    }
    m.init(seed, bias);
  }
  std::printf("model: d %d L %d H %d ff %d demb %d HH %d NB %d maxPeaks %d; %lld params (%.1f M); B %d lr %g wd %g drop %g clip %g tf32 %d\n",
              m.c.d, m.c.L, m.c.H, m.c.ff, m.c.demb, m.c.HH, m.c.NB, m.c.maxPeaks, (long long)m.nP, m.nP / 1e6, B, lr, wd, drop,
              clip, int(tfm::tf32Flag()));
  m.reserve(B, m.c.maxPeaks + 1);

  // deterministic per-epoch batches: shuffle, sort chunks of 64 batches by token count, shuffle batches
  auto makeBatches = [&](int ep) {
    std::vector<int> o = idx;
    uint64_t r = 0x1234567ull + seed * 7919 + uint64_t(ep) * 104729;
    auto nx = [&]() { r ^= r << 13; r ^= r >> 7; r ^= r << 17; return r; };
    for (size_t i = o.size(); i > 1; --i) std::swap(o[i - 1], o[nx() % i]);
    const size_t chunk = size_t(B) * 64;
    for (size_t c0 = 0; c0 < o.size(); c0 += chunk) {
      size_t c1 = std::min(o.size(), c0 + chunk);
      std::stable_sort(o.begin() + c0, o.begin() + c1, [&](int a, int b) { return npEff[a] < npEff[b]; });
    }
    std::vector<std::vector<int>> bs;
    for (size_t b0 = 0; b0 + B <= o.size(); b0 += B) bs.emplace_back(o.begin() + b0, o.begin() + b0 + B);
    for (size_t i = bs.size(); i > 1; --i) std::swap(bs[i - 1], bs[nx() % i]);
    return bs;
  };
  const int64_t stepsPerEpoch = int64_t(idx.size() / size_t(B));
  const int64_t totalSteps = stepsPerEpoch * epochs;
  std::printf("steps/epoch %lld, total %lld\n", (long long)stepsPerEpoch, (long long)totalSteps);
  std::fflush(stdout);

  struct Prep { tfm::HostBatch hb; std::vector<real> T; };
  auto prep = [&](const std::vector<int>& specs, Prep& p) {
    buildBatch(st, specs, m.c.maxPeaks, p.hb);
    p.T.assign(size_t(B) * m.c.NB, real(0));
    for (int b = 0; b < B; ++b) {
      int mol = st.head[specs[b]].mol;
      for (uint32_t k = fps.ptr[mol]; k < fps.ptr[mol + 1]; ++k) p.T[size_t(b) * m.c.NB + fps.bits[k]] = 1;
    }
  };
  std::vector<tfm::Model*> ms{&m};
  auto t0 = std::chrono::steady_clock::now();
  auto secs = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
  int64_t step = step0;
  double lossAcc = 0; int lossN = 0; double tokAcc = 0, tLast = 0;
  std::vector<real> zh;
  bool stopped = false;
  for (int ep = int(step0 / stepsPerEpoch); ep < epochs && !stopped; ++ep) {
    auto bs = makeBatches(ep);
    int64_t st0 = step - int64_t(ep) * stepsPerEpoch;
    Prep cur, nxt;
    prep(bs[st0], cur);
    for (int64_t si = st0; si < stepsPerEpoch; ++si, ++step) {
      std::thread th;
      if (si + 1 < stepsPerEpoch) th = std::thread([&, si]() { prep(bs[si + 1], nxt); });
      m.setBatch(cur.hb);
      tfm::upload(m.T, cur.T.data(), int64_t(cur.T.size()));
      m.forward(true, drop, uint64_t(step) + 1);
      m.lossGrad(ls);
      m.backward(drop, uint64_t(step) + 1);
      double gn = std::sqrt(m.gradNorm2());
      float gs = (clip > 0 && gn > clip) ? float(clip / gn) : 1.f;
      float t = float(step) / float(std::max<int64_t>(1, totalSteps));
      float lrNow = step < warm ? lr * float(step + 1) / float(warm) : lr * 0.5f * (1.f + std::cos(3.14159265f * t));
      if (std::isfinite(gn)) m.adamw(lrNow, wd, gs, 0.9f, b2);
      else std::printf("non-finite grad norm at step %lld, skipped\n", (long long)step);
      tokAcc += double(cur.hb.B) * cur.hb.N;
      if (si % 50 == 0) {
        zh.resize(size_t(B) * m.c.NB);
        tfm::download(m.Z, zh.data(), int64_t(zh.size()));
        double l = 0;
        for (size_t i = 0; i < zh.size(); ++i) {
          double z = double(zh[i]);
          l += std::max(z, 0.0) - z * double(cur.T[i]) + std::log1p(std::exp(-std::fabs(z)));
        }
        lossAcc += l / B; ++lossN;
      }
      if (si % logEvery == 0) {
        double now = secs();
        std::printf("ep %d step %lld/%lld (global %lld)  train BCE/spectrum %.3f  |g| %.3f  lr %.2e  %.0f tok/s  %.0fs\n", ep,
                    (long long)si, (long long)stepsPerEpoch, (long long)step, lossAcc / std::max(1, lossN), gn, lrNow,
                    tokAcc / std::max(1e-9, now - tLast), now);
        std::fflush(stdout);
        lossAcc = 0; lossN = 0; tokAcc = 0; tLast = now;
      }
      if (th.joinable()) th.join();
      std::swap(cur, nxt);
      bool stopNow = stopAfter > 0 && secs() > stopAfter * 60.0;
      if ((ckptEvery > 0 && (step + 1) % ckptEvery == 0) || stopNow) {
        m.save(ckPath, true);
        std::printf("checkpoint %s at step %lld (%.0fs)\n", ckPath.c_str(), (long long)(step + 1), secs());
        std::fflush(stdout);
      }
      if (stopNow) { stopped = true; ++step; break; }
    }
    if (stopped) break;
    std::printf("== epoch %d done (%.0fs)\n", ep + 1, secs());
    m.save(ckPath, true);
    m.save(dir + "/" + out + "_ep" + std::to_string(ep + 1) + ".bin", false);
    validate(m, st, fps, val);
    evaluateSets(ms, cf.fps, sets, ep + 1 == epochs);
  }
  if (!stopped) {
    m.save(dir + "/" + out + ".bin", false);
    std::printf("wrote %s.bin (%.0fs)\n", out.c_str(), secs());
  } else {
    std::printf("stopped after %.1f min at step %lld; resume with resume=1\n", secs() / 60, (long long)step);
  }
  return 0;
}
