// Masked-peak self-supervised pretraining (pretrain_STATUS.md) on the own backend.
//
//   pt_train gradcheck [drop=0.1]        finite differences on a tiny config (exact in the -DTFM_DOUBLE CPU build)
//   pt_train train key=value...
//     dir=C:\prog\casmi lib=wlib out=<stem> frac=1 (fraction of training spectra, pilot 0.1)
//     d=256 L=4 H=8 ff=4 demb=<d> HH=512 NC=1536 maxPeaks=64 epochs=1 B=128 lr=3e-4 wd=0.05 drop=0.1 clip=1
//     warm=2000 seed=31 hold=20 tf32=1 ckptEvery=2000 stopAfter=0(min) resume=0 valMax=20000 logEvery=500
//     mask=0.3 mwi=1 (masked peaks sampled proportionally to sqrt-intensity) wOff=1 wInt=5
//   pt_train nbr model=<stem>.bin lib=wlib qrys=hcqry,hxqry,haqry tag=<tag> [nbr=200 B=256 libMax=0]
//     embeds every query and library spectrum (unmasked forward), cosine similarity on the GPU,
//     per query molecule the max over its spectra x each library molecule's spectra (polarity matched as
//     attn_eval), keeps the top nbr library molecules per variant:
//       slot 0 = global (precursor) token  Y_g,  1 = mean of peak tokens Y_m,  2 = (cos_g + cos_m)/2
//     -> <dir>/ptnbr_<qry>_<tag>.bin in attn_eval's neighbour-cache format (4 slots, slot 3 empty).
//   pt_train embed model=<stem>.bin set=<stem of spectra> out=<file> [var=0] : fp16 L2-normalised embeddings
//     (one row per spectrum) for shipping precomputed library embeddings (Kaggle CPU inference = query side only).
//   pt_train export model=<stem>.bin lib=wlib fp=fp3c16k HH=2048 out=<file>  : TFM1 file with the pretrained
//     encoder (identical layout up to lnfb) + fresh fingerprint head -> tfm_train train resume=1 (out=<stem>,
//     reads <stem>_ck.bin) fine-tunes it.
//   pt_train mollogits model=<TFM1 file> qry=hcqry out=<file>  : per-molecule mean logits (fpnet_ggml _mol.bin
//     format) for fpown\build\win_eval.exe (window yardstick of a fine-tuned / from-scratch tfm model).
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
#include <thread>
#include <unordered_map>
#include <vector>

#include "parallel.hpp"
#include "mpm_model.hpp"
#include "tfm_model.hpp"

using mpm::real;

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
  std::fflush(stdout);
  return s;
}

uint64_t nameHash(const std::string& s) {
  uint64_t h = 1469598103934665603ull;
  for (char c : s) h = (h ^ uint8_t(c)) * 1099511628211ull;
  return h ^ (h >> 29);
}

// peaks -> tokens (as tfm_train: mz<=prec+2, I/Imax>=1e-3, top-k by intensity, mz order, sqrt)
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
  if (om.empty()) { om.push_back(0.f); oi.push_back(0.f); }
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

struct MaskCfg { float frac = 0.3f; bool byInt = true; int NC = 1536; };

// batch with masking. key seeds the per-batch mask RNG (deterministic). frac <= 0: no masking (embedding).
void buildBatch(const Store& st, const std::vector<int>& specs, int maxP, const MaskCfg& mc, uint64_t key,
                mpm::HostBatch& hb) {
  const int B = int(specs.size());
  std::vector<std::vector<float>> Mz(B), I(B);
  ::parallel_for(B, [&](int64_t b) {
    const Head& h = st.head[specs[b]];
    prepPeaks(h, st.mzs(h), st.its(h), maxP, Mz[b], I[b]);
  });
  int mx = 1;
  for (auto& v : Mz) mx = std::max(mx, int(v.size()));
  hb.B = B; hb.N = 1 + mx;
  hb.mz.assign(size_t(B) * mx, 0.f); hb.it.assign(size_t(B) * mx, 0.f);
  hb.prec.resize(B); hb.mode.resize(B); hb.np.resize(B); hb.ad.resize(B); hb.in.resize(B);
  hb.mt.clear(); hb.mcls.clear(); hb.moff.clear(); hb.mit.clear();
  uint64_t r = 0x9E3779B97F4A7C15ull ^ (key * 0xD1B54A32D192ED03ull + 12345);
  auto nx = [&]() { r ^= r << 13; r ^= r >> 7; r ^= r << 17; return double(r >> 11) * (1.0 / 9007199254740992.0); };
  std::vector<std::pair<double, int>> keys;
  for (int b = 0; b < B; ++b) {
    const Head& h = st.head[specs[b]];
    std::copy(Mz[b].begin(), Mz[b].end(), hb.mz.begin() + size_t(b) * mx);
    std::copy(I[b].begin(), I[b].end(), hb.it.begin() + size_t(b) * mx);
    int np = int(Mz[b].size());
    hb.np[b] = np;
    hb.prec[b] = h.precursor;
    decodeTag(h.tag, hb.ad[b], hb.in[b], hb.mode[b]);
    if (mc.frac <= 0.f || np < 2) continue;
    int k = std::max(1, std::min(np - 1, int(std::lround(mc.frac * np))));
    // weighted sampling without replacement (Efraimidis-Spirakis keys u^(1/w))
    keys.clear();
    for (int j = 0; j < np; ++j) {
      double w = mc.byInt ? std::max(1e-3, double(I[b][j])) : 1.0;
      double u = std::max(1e-300, nx());
      keys.emplace_back(std::log(u) / w, j);
    }
    std::partial_sort(keys.begin(), keys.begin() + k, keys.end(), [](auto& a, auto& c) { return a.first > c.first; });
    std::vector<int> sel;
    for (int i = 0; i < k; ++i) sel.push_back(keys[i].second);
    std::sort(sel.begin(), sel.end());
    for (int j : sel) {
      int cls; float off;
      mpm::binOf(Mz[b][j], mc.NC, cls, off);
      hb.mt.push_back(int32_t(size_t(b) * mx + j));
      hb.mcls.push_back(cls); hb.moff.push_back(off); hb.mit.push_back(I[b][j]);
    }
  }
}

std::map<std::string, std::string> A;
std::string get(const std::string& k, const std::string& d) { auto it = A.find(k); return it == A.end() ? d : it->second; }
int geti(const std::string& k, int d) { auto it = A.find(k); return it == A.end() ? d : std::atoi(it->second.c_str()); }
float getf(const std::string& k, float d) { auto it = A.find(k); return it == A.end() ? d : float(std::atof(it->second.c_str())); }

// ------------------------------------------------------------------ gradient check
int gradcheck() {
  mpm::Model m;
  m.c.d = 8; m.c.L = 2; m.c.H = 2; m.c.ff = 2; m.c.demb = 6; m.c.HH = 10; m.c.NC = 9; m.c.maxPeaks = 6;
  m.c.nAd = 3; m.c.nIn = 2;
  const float drop = getf("drop", 0.1f), wOff = getf("wOff", 0.7f), wInt = getf("wInt", 1.3f);
  const double eps = std::is_same<real, double>::value ? 1e-6 : 3e-3;
  mpm::tf32Flag() = false;
  m.init(5, {});
  {
    std::vector<real> h(m.nP);
    mpm::download(m.W, h.data(), m.nP);
    uint64_t s = 99;
    std::vector<real> r(m.nP);
    mpm::Model::gauss(r, 0, m.nP, 0.3, s);
    for (int64_t i = 0; i < m.nP; ++i) h[i] += r[i];
    mpm::upload(m.W, h.data(), m.nP);
  }
  mpm::HostBatch hb;
  hb.B = 3; int np[3] = {1, 4, 6}; hb.N = 7;
  hb.mz.assign(3 * 6, 0.f); hb.it.assign(3 * 6, 0.f);
  uint64_t s = 1234;
  auto rnd = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return float((s >> 11) * (1.0 / 9007199254740992.0)); };
  for (int b = 0; b < 3; ++b) {
    hb.prec.push_back(5.f + 4.f * rnd());
    for (int j = 0; j < np[b]; ++j) { hb.mz[b * 6 + j] = 0.5f + 8.f * rnd(); hb.it[b * 6 + j] = 0.1f + 0.9f * rnd(); }
    hb.np.push_back(np[b]); hb.ad.push_back(b % 3); hb.in.push_back(b % 2); hb.mode.push_back(b == 1 ? -1.f : 1.f);
  }
  // masked: spectrum 1 peaks 0,2; spectrum 2 peaks 1,4,5 (bins in [0, NC-1], also one clamped)
  for (int t : {6 + 0, 6 + 2, 12 + 1, 12 + 4, 12 + 5}) {
    int cls; float off;
    mpm::binOf(hb.mz[t], m.c.NC, cls, off);
    hb.mt.push_back(t); hb.mcls.push_back(cls); hb.moff.push_back(off); hb.mit.push_back(hb.it[t]);
  }
  hb.mcls[4] = m.c.NC - 1;
  m.setBatch(hb);
  const uint64_t seed = 77;
  auto lossHost = [&]() {
    m.forward(true, drop, seed);
    m.lossGrad(wOff, wInt);
    double p[4];
    m.lossParts(p);
    return p[0] + wOff * p[1] + wInt * p[2];
  };
  double L0 = lossHost();
  m.backward(drop, seed);
  std::vector<real> G(m.nP), W(m.nP);
  mpm::download(m.G, G.data(), m.nP);
  mpm::download(m.W, W.data(), m.nP);
  std::printf("gradcheck: real=%s, nP=%lld, masked %d, loss %.6f, drop %.2f, eps %g\n", sizeof(real) == 8 ? "double" : "float",
              (long long)m.nP, m.M, L0, drop, eps);
  double worst = 0;
  for (mpm::PRef* p : m.plist) {
    double maxRel = 0, maxAbs = 0, gmax = 0;
    int64_t nCheck = std::min<int64_t>(p->n, 40);
    for (int64_t k = 0; k < nCheck; ++k) {
      int64_t i = p->off + (p->n <= 40 ? k : (k * 7919) % p->n);
      real old = W[i];
      real v = real(old + eps);
      mpm::upload(m.W, &v, 1, i);
      double lp = lossHost();
      v = real(old - eps);
      mpm::upload(m.W, &v, 1, i);
      double lm = lossHost();
      mpm::upload(m.W, &old, 1, i);
      double num = (lp - lm) / (2 * eps), an = double(G[i]);
      double ad = std::fabs(num - an);
      maxAbs = std::max(maxAbs, ad);
      gmax = std::max(gmax, std::fabs(an));
      maxRel = std::max(maxRel, ad / std::max(1e-6, std::fabs(num) + std::fabs(an)));
    }
    double relScale = maxAbs / std::max(1e-12, gmax);
    worst = std::max(worst, relScale);
    std::printf("  %-6s %3dx%-4d checked %2lld  max|g| %.3e  max|num-an| %.3e  maxRel(entry) %.2e  rel(tensor) %.2e\n", p->name,
                p->rows, p->cols, (long long)nCheck, gmax, maxAbs, maxRel, relScale);
  }
  std::printf("gradcheck worst rel(tensor) = %.3e  -> %s\n", worst,
              worst < (sizeof(real) == 8 ? 1e-6 : 3e-2) ? "PASS" : "CHECK");
  return 0;
}

// masked-peak loss on a fixed set of spectra (deterministic masks)
void validate(mpm::Model& m, const Store& st, const std::vector<int>& val, const MaskCfg& mc, float wOff, float wInt,
              const char* lab) {
  if (val.empty()) return;
  const int B = 128;
  double acc[4] = {0, 0, 0, 0}, nTok = 0;
  mpm::HostBatch hb;
  for (size_t s0 = 0; s0 < val.size(); s0 += B) {
    std::vector<int> sp(val.begin() + s0, val.begin() + std::min(val.size(), s0 + B));
    buildBatch(st, sp, m.c.maxPeaks, mc, 0xABCDEFull + s0, hb);
    if (hb.mt.empty()) continue;
    m.setBatch(hb);
    m.forward(false, 0.f, 0);
    m.lossGrad(wOff, wInt);
    double p[4];
    m.lossParts(p);
    for (int k = 0; k < 4; ++k) acc[k] += p[k] * m.M;
    nTok += m.M;
  }
  for (auto& x : acc) x /= std::max(1.0, nTok);
  std::printf("  %s: %zu spectra, %.0f masked peaks: CE(bin) %.4f  acc(bin) %.4f  MSE(10*offset) %.4f  MSE(sqrt int) %.4f  total %.4f\n",
              lab, val.size(), nTok, acc[0], acc[3], acc[1], acc[2], acc[0] + wOff * acc[1] + wInt * acc[2]);
  std::fflush(stdout);
}

// ------------------------------------------------------------------ train
int train() {
  std::string dir = get("dir", "C:\\prog\\casmi"), lib = get("lib", "wlib"), out = get("out", "mpm");
  const int epochs = geti("epochs", 1), B = geti("B", 128);
  const float lr = getf("lr", 3e-4f), wd = getf("wd", 0.05f), drop = getf("drop", 0.1f), clip = getf("clip", 1.f);
  const float wOff = getf("wOff", 1.f), wInt = getf("wInt", 5.f), frac = getf("frac", 1.f);
  const int warm = geti("warm", 2000), hold = geti("hold", 20), ckptEvery = geti("ckptEvery", 2000);
  const uint64_t seed = uint64_t(geti("seed", 31));
  const double stopAfter = getf("stopAfter", 0.f);
  const bool resume = geti("resume", 0) != 0;
  const int valMax = geti("valMax", 20000), logEvery = geti("logEvery", 500);
  mpm::tf32Flag() = geti("tf32", 1) != 0;
  MaskCfg mc;
  mc.frac = getf("mask", 0.3f); mc.byInt = geti("mwi", 1) != 0;

  Store st = loadSpectra(dir, lib);
  mpm::Model m;
  const std::string ckPath = dir + "/" + out + "_ck.bin";
  int64_t step0 = 0;
  if (resume) {
    if (!m.load(ckPath, true)) { std::fprintf(stderr, "cannot resume from %s\n", ckPath.c_str()); return 1; }
    step0 = m.adamT;
    std::printf("resumed %s at step %lld\n", ckPath.c_str(), (long long)step0);
  } else {
    m.c.d = geti("d", 256); m.c.L = geti("L", 4); m.c.H = geti("H", 8); m.c.ff = geti("ff", 4);
    m.c.demb = geti("demb", m.c.d); m.c.HH = geti("HH", 512); m.c.NC = geti("NC", 1536); m.c.maxPeaks = geti("maxPeaks", 64);
  }
  mc.NC = m.c.NC;
  std::vector<int> idx, val;
  {
    uint64_t r = 0x5EEDull + seed;
    auto nx = [&]() { r ^= r << 13; r ^= r >> 7; r ^= r << 17; return double(r >> 11) * (1.0 / 9007199254740992.0); };
    for (size_t s = 0; s < st.head.size(); ++s) {
      int mol = st.head[s].mol;
      if (st.head[s].n <= 0 || mol < 0) continue;
      if (hold > 0 && nameHash(st.name[mol]) % uint64_t(hold) == 0) val.push_back(int(s));
      else if (frac >= 1.f || nx() < frac) idx.push_back(int(s));
    }
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
    std::printf("training spectra %zu (frac %.3f), val spectra %zu (held-out molecules); mean tokens %.1f, %.1f%% capped at %d peaks\n",
                idx.size(), frac, val.size(), mean / idx.size(), 100.0 * full / idx.size(), m.c.maxPeaks);
  }
  if (!resume) {
    // class bias = log frequency of the bins over (masked-weighted) peaks of a sample of training spectra
    std::vector<double> cnt(m.c.NC, 1.0);
    double so = 0, si = 0, n = 0;
    for (size_t i = 0; i < idx.size(); i += 50) {
      const Head& h = st.head[idx[i]];
      std::vector<float> om, oi;
      prepPeaks(h, st.mzs(h), st.its(h), m.c.maxPeaks, om, oi);
      for (size_t j = 0; j < om.size(); ++j) {
        int cls; float off;
        mpm::binOf(om[j], m.c.NC, cls, off);
        double w = mc.byInt ? oi[j] : 1.0;
        cnt[cls] += w; so += w * off; si += w * oi[j]; n += w;
      }
    }
    double tot = 0;
    for (double c : cnt) tot += c;
    std::vector<float> bias(m.c.NC + 2);
    for (int k = 0; k < m.c.NC; ++k) bias[k] = float(std::log(cnt[k] / tot));
    bias[m.c.NC] = float(so / std::max(1.0, n)); bias[m.c.NC + 1] = float(si / std::max(1.0, n));
    m.init(seed, bias);
    // baseline: loss of the marginal (bias-only) predictor ~ entropy of the bin distribution
    double H = 0;
    for (double c : cnt) H -= c / tot * std::log(c / tot);
    std::printf("bin marginal entropy %.4f nats (CE of the unconditional predictor); mean 10*offset %.3f, mean it %.3f\n", H,
                so / std::max(1.0, n), si / std::max(1.0, n));
  }
  std::printf("model: d %d L %d H %d ff %d demb %d HH %d NC %d maxPeaks %d; %lld params (%.1f M); B %d lr %g wd %g drop %g clip %g "
              "mask %.2f mwi %d wOff %g wInt %g tf32 %d\n",
              m.c.d, m.c.L, m.c.H, m.c.ff, m.c.demb, m.c.HH, m.c.NC, m.c.maxPeaks, (long long)m.nP, m.nP / 1e6, B, lr, wd, drop,
              clip, mc.frac, int(mc.byInt), wOff, wInt, int(mpm::tf32Flag()));
  m.reserve(B, m.c.maxPeaks + 1);
  if (!resume) validate(m, st, val, mc, wOff, wInt, "val (init)");

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

  auto t0 = std::chrono::steady_clock::now();
  auto secs = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
  int64_t step = step0;
  double acc[4] = {0, 0, 0, 0}; double accN = 0, tokAcc = 0, tLast = 0;
  bool stopped = false;
  for (int ep = int(step0 / stepsPerEpoch); ep < epochs && !stopped; ++ep) {
    auto bs = makeBatches(ep);
    int64_t st0 = step - int64_t(ep) * stepsPerEpoch;
    mpm::HostBatch cur, nxt;
    buildBatch(st, bs[st0], m.c.maxPeaks, mc, uint64_t(step) * 2 + 1, cur);
    for (int64_t si = st0; si < stepsPerEpoch; ++si, ++step) {
      std::thread th;
      if (si + 1 < stepsPerEpoch)
        th = std::thread([&, si]() { buildBatch(st, bs[si + 1], m.c.maxPeaks, mc, uint64_t(step + 1) * 2 + 1, nxt); });
      if (!cur.mt.empty()) {
        m.setBatch(cur);
        m.forward(true, drop, uint64_t(step) + 1);
        m.lossGrad(wOff, wInt);
        m.backward(drop, uint64_t(step) + 1);
        double gn = std::sqrt(m.gradNorm2());
        float gs = (clip > 0 && gn > clip) ? float(clip / gn) : 1.f;
        float t = float(step) / float(std::max<int64_t>(1, totalSteps));
        float lrNow = step < warm ? lr * float(step + 1) / float(warm) : lr * 0.5f * (1.f + std::cos(3.14159265f * t));
        if (std::isfinite(gn)) m.adamw(lrNow, wd, gs, 0.9f, 0.999f);
        else std::printf("non-finite grad norm at step %lld, skipped\n", (long long)step);
        tokAcc += double(cur.B) * cur.N;
        if (si % 20 == 0) {
          double p[4];
          m.lossParts(p);
          for (int k = 0; k < 4; ++k) acc[k] += p[k];
          accN += 1;
        }
        if (si % logEvery == 0) {
          double now = secs();
          for (auto& x : acc) x /= std::max(1.0, accN);
          std::printf("ep %d step %lld/%lld (global %lld)  CE(bin) %.4f acc %.4f  MSE off %.4f int %.4f  total %.4f  |g| %.3f  lr %.2e  %.0f tok/s  %.0fs\n",
                      ep, (long long)si, (long long)stepsPerEpoch, (long long)step, acc[0], acc[3], acc[1], acc[2],
                      acc[0] + wOff * acc[1] + wInt * acc[2], gn, lrNow, tokAcc / std::max(1e-9, now - tLast), now);
          std::fflush(stdout);
          for (auto& x : acc) x = 0;
          accN = 0; tokAcc = 0; tLast = now;
        }
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
    validate(m, st, val, mc, wOff, wInt, "val");
  }
  m.save(dir + "/" + out + ".bin", false);
  std::printf("%s %s.bin (%.0fs)\n", stopped ? "stopped; wrote (partial)" : "wrote", out.c_str(), secs());
  return 0;
}

// ------------------------------------------------------------------ embeddings
// unmasked forward of a list of spectra -> L2-normalised embeddings per variant (0 global, 1 mean)
void embedSpectra(mpm::Model& m, const Store& st, const std::vector<int>& specs, std::vector<float>& eg, std::vector<float>& em,
                  int B, const std::vector<int>& npEff) {
  const int d = m.c.d;
  eg.assign(specs.size() * size_t(d), 0.f); em.assign(specs.size() * size_t(d), 0.f);
  std::vector<int> ord(specs.size());
  std::iota(ord.begin(), ord.end(), 0);
  std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return npEff[specs[a]] < npEff[specs[b]]; });
  MaskCfg none; none.frac = 0.f; none.NC = m.c.NC;
  mpm::HostBatch hb;
  std::vector<real> E;
  for (size_t s0 = 0; s0 < ord.size(); s0 += B) {
    size_t s1 = std::min(ord.size(), s0 + B);
    std::vector<int> sp;
    for (size_t i = s0; i < s1; ++i) sp.push_back(specs[ord[i]]);
    buildBatch(st, sp, m.c.maxPeaks, none, 0, hb);
    m.setBatch(hb);
    m.forward(false, 0.f, 0);
    E.resize(size_t(hb.B) * 2 * d);
    mpm::download(m.Emb, E.data(), int64_t(E.size()));
    for (size_t i = s0; i < s1; ++i) {
      const real* e = E.data() + (i - s0) * 2 * d;
      for (int v = 0; v < 2; ++v) {
        double n2 = 0;
        for (int j = 0; j < d; ++j) n2 += double(e[v * d + j]) * double(e[v * d + j]);
        float inv = float(1.0 / std::sqrt(n2 + 1e-12));
        float* o = (v == 0 ? eg.data() : em.data()) + size_t(ord[i]) * d;
        for (int j = 0; j < d; ++j) o[j] = float(e[v * d + j]) * inv;
      }
    }
  }
}
int polarityOf(const Head& h) {
  int ad, in; float mode;
  decodeTag(h.tag, ad, in, mode);
  return mode > 0 ? 1 : (mode < 0 ? -1 : 0);
}
std::vector<int> peakCounts(const Store& st, int maxP) {
  std::vector<int> c(st.head.size(), 1);
  ::parallel_for(int64_t(st.head.size()), [&](int64_t s) { const Head& h = st.head[s]; c[s] = countPeaks(h, st.mzs(h), st.its(h), maxP); });
  return c;
}

struct Nbr { float sim; int mol; };

int nbrMode() {
  std::string dir = get("dir", "C:\\prog\\casmi"), lib = get("lib", "wlib"), tag = get("tag", "pt");
  const int NBR = geti("nbr", 200), B = geti("B", 256), libMax = geti("libMax", 0);
  mpm::tf32Flag() = geti("tf32", 0) != 0;
  mpm::Model m;
  if (!m.load(dir + "/" + get("model", "mpm.bin"), false)) { std::fprintf(stderr, "cannot load model\n"); return 1; }
  const int d = m.c.d;
  m.reserve(B, m.c.maxPeaks + 1);
  auto t0 = std::chrono::steady_clock::now();
  auto secs = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
  // queries of all sets
  std::vector<std::string> qn;
  { std::stringstream ss(get("qrys", "hcqry,hxqry,haqry")); std::string t; while (std::getline(ss, t, ',')) if (!t.empty()) qn.push_back(t); }
  struct QS { Store st; std::vector<int> spec; std::vector<float> eg, em; std::vector<int8_t> pol; };
  std::vector<QS> Q(qn.size());
  std::vector<float> QG, QM;  // all query spectra stacked
  std::vector<int> qSet, qMol;
  std::vector<int8_t> qPol;
  for (size_t i = 0; i < qn.size(); ++i) {
    Q[i].st = loadSpectra(dir, qn[i]);
    for (size_t s = 0; s < Q[i].st.head.size(); ++s)
      if (Q[i].st.head[s].n > 0 && Q[i].st.head[s].mol >= 0) Q[i].spec.push_back(int(s));
    auto np = peakCounts(Q[i].st, m.c.maxPeaks);
    embedSpectra(m, Q[i].st, Q[i].spec, Q[i].eg, Q[i].em, B, np);
    QG.insert(QG.end(), Q[i].eg.begin(), Q[i].eg.end());
    QM.insert(QM.end(), Q[i].em.begin(), Q[i].em.end());
    for (int s : Q[i].spec) { qSet.push_back(int(i)); qMol.push_back(Q[i].st.head[s].mol); qPol.push_back(int8_t(polarityOf(Q[i].st.head[s]))); }
  }
  const int nq = int(qSet.size());
  std::printf("query spectra %d (%.0fs)\n", nq, secs());
  Store L = loadSpectra(dir, lib);
  const int nLM = int(L.name.size());
  // best[v][set][qmol][libmol] dense max (float), v = 0 g, 1 m, 2 gm
  std::vector<std::vector<std::vector<float>>> best(3);
  std::vector<int> qOff(qn.size() + 1, 0);
  for (size_t i = 0; i < qn.size(); ++i) qOff[i + 1] = qOff[i] + int(Q[i].st.name.size());
  const int nQM = qOff.back();
  for (int v = 0; v < 3; ++v) best[v].assign(nQM, std::vector<float>(nLM, -2.f));
  std::vector<int> qRow(nq);
  for (int i = 0; i < nq; ++i) qRow[i] = qOff[qSet[i]] + qMol[i];
  mpm::Buf dQG, dQM, dLG, dLM, dSG, dSM;
  {
    std::vector<real> h(QG.begin(), QG.end());
    dQG.alloc(int64_t(h.size())); mpm::upload(dQG, h.data(), int64_t(h.size()));
    std::vector<real> h2(QM.begin(), QM.end());
    dQM.alloc(int64_t(h2.size())); mpm::upload(dQM, h2.data(), int64_t(h2.size()));
  }
  const int LB = 4096;
  dLG.alloc(int64_t(LB) * d); dLM.alloc(int64_t(LB) * d); dSG.alloc(int64_t(nq) * LB); dSM.alloc(int64_t(nq) * LB);
  std::vector<int> lspec;
  for (size_t s = 0; s < L.head.size(); ++s) if (L.head[s].n > 0 && L.head[s].mol >= 0) lspec.push_back(int(s));
  if (libMax > 0 && int(lspec.size()) > libMax) lspec.resize(libMax);
  auto lnp = peakCounts(L, m.c.maxPeaks);
  std::vector<float> eg, em;
  std::vector<real> SG(size_t(nq) * LB), SM(size_t(nq) * LB), hl;
  for (size_t c0 = 0; c0 < lspec.size(); c0 += LB) {
    size_t c1 = std::min(lspec.size(), c0 + LB);
    std::vector<int> chunk(lspec.begin() + c0, lspec.begin() + c1);
    const int nb = int(chunk.size());
    embedSpectra(m, L, chunk, eg, em, B, lnp);
    hl.assign(eg.begin(), eg.end()); mpm::upload(dLG, hl.data(), int64_t(hl.size()));
    hl.assign(em.begin(), em.end()); mpm::upload(dLM, hl.data(), int64_t(hl.size()));
    // S[q, l] = Q[q,:] . L[l,:]
    mpm::gemm(false, true, nq, nb, d, 1, dQG.data(), d, 0, dLG.data(), d, 0, 0, dSG.data(), nb, 0);
    mpm::gemm(false, true, nq, nb, d, 1, dQM.data(), d, 0, dLM.data(), d, 0, 0, dSM.data(), nb, 0);
    mpm::download(dSG, SG.data(), int64_t(nq) * nb);
    mpm::download(dSM, SM.data(), int64_t(nq) * nb);
    std::vector<int8_t> lp(nb);
    std::vector<int> lm(nb);
    for (int j = 0; j < nb; ++j) { lp[j] = int8_t(polarityOf(L.head[chunk[j]])); lm[j] = L.head[chunk[j]].mol; }
    // rows of the same query molecule may be processed by different threads -> partition by query molecule row
    ::parallel_for(int64_t(nQM), [&](int64_t row) {
      for (int i = 0; i < nq; ++i) {
        if (qRow[i] != row) continue;
        const real* sg = SG.data() + size_t(i) * nb; const real* sm = SM.data() + size_t(i) * nb;
        float* bg = best[0][row].data(); float* bm = best[1][row].data(); float* bgm = best[2][row].data();
        for (int j = 0; j < nb; ++j) {
          if (qPol[i] != 0 && lp[j] != 0 && qPol[i] != lp[j]) continue;
          int mo = lm[j];
          float g = float(sg[j]), mm = float(sm[j]), gm = 0.5f * (g + mm);
          if (g > bg[mo]) bg[mo] = g;
          if (mm > bm[mo]) bm[mo] = mm;
          if (gm > bgm[mo]) bgm[mo] = gm;
        }
      }
    });
    if ((c0 / LB) % 50 == 0) { std::printf("  library %zu / %zu spectra (%.0fs)\n", c1, lspec.size(), secs()); std::fflush(stdout); }
  }
  // write one neighbour file per query set (attn_eval format: nMol, K, then 4 slots x nMol x {count, Nbr[count]})
  for (size_t si = 0; si < qn.size(); ++si) {
    std::string path = dir + "/ptnbr_" + qn[si] + "_" + tag + ".bin";
    std::ofstream f(path, std::ios::binary);
    uint32_t nm = uint32_t(Q[si].st.name.size()), K = uint32_t(NBR);
    f.write((char*)&nm, 4); f.write((char*)&K, 4);
    double topMed[3] = {0, 0, 0};
    for (int v = 0; v < 4; ++v)
      for (uint32_t mo = 0; mo < nm; ++mo) {
        std::vector<Nbr> nb;
        if (v < 3) {
          const auto& row = best[v][qOff[si] + mo];
          std::vector<int> ids;
          for (int j = 0; j < nLM; ++j) if (row[j] > -1.5f) ids.push_back(j);
          int k = std::min<int>(NBR, int(ids.size()));
          std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), [&](int a, int b) { return row[a] > row[b]; });
          for (int j = 0; j < k; ++j) nb.push_back({row[ids[j]], ids[j]});
          if (!nb.empty()) topMed[v] += nb[0].sim;
        }
        uint32_t c = uint32_t(nb.size());
        f.write((char*)&c, 4);
        if (c) f.write((char*)nb.data(), sizeof(Nbr) * c);
      }
    std::printf("wrote %s (%u query molecules; mean top-1 cosine g %.3f m %.3f gm %.3f)\n", path.c_str(), nm, topMed[0] / nm,
                topMed[1] / nm, topMed[2] / nm);
  }
  std::printf("nbr done (%.0fs)\n", secs());
  return 0;
}

// fp16 embeddings of every spectrum of a store (row = spectrum index; empty spectra get zeros)
uint16_t toHalf(float f) {
  uint32_t x; std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u; int32_t e = int32_t((x >> 23) & 0xFF) - 127 + 15; uint32_t mant = x & 0x7FFFFFu;
  if (e <= 0) return uint16_t(sign);
  if (e >= 31) return uint16_t(sign | 0x7C00u);
  uint32_t h = sign | (uint32_t(e) << 10) | (mant >> 13);
  if (mant & 0x1000u) ++h;  // round half up
  return uint16_t(h);
}
int embedMode() {
  std::string dir = get("dir", "C:\\prog\\casmi"), set = get("set", "wlib"), out = get("out", "emb.bin");
  const int B = geti("B", 256), var = geti("var", 0);
  mpm::Model m;
  if (!m.load(dir + "/" + get("model", "mpm.bin"), false)) { std::fprintf(stderr, "cannot load model\n"); return 1; }
  m.reserve(B, m.c.maxPeaks + 1);
  Store st = loadSpectra(dir, set);
  std::vector<int> all;
  for (size_t s = 0; s < st.head.size(); ++s) all.push_back(int(s));
  auto np = peakCounts(st, m.c.maxPeaks);
  std::vector<float> eg, em;
  embedSpectra(m, st, all, eg, em, B, np);
  const auto& e = var == 0 ? eg : em;
  std::ofstream f(out, std::ios::binary);
  uint32_t n = uint32_t(all.size()), d = uint32_t(m.c.d);
  f.write((char*)&n, 4); f.write((char*)&d, 4);
  std::vector<uint16_t> h(e.size());
  for (size_t i = 0; i < e.size(); ++i) h[i] = st.head[i / d].n > 0 ? toHalf(e[i]) : 0;
  f.write((char*)h.data(), 2 * h.size());
  std::printf("wrote %s: %u x %u fp16\n", out.c_str(), n, d);
  return 0;
}

// pretrained encoder -> TFM1 (fingerprint head freshly initialised, output bias = logit of the bit base rate)
int exportMode() {
  std::string dir = get("dir", "C:\\prog\\casmi"), lib = get("lib", "wlib"), fp = get("fp", "fp3c16k"), out = get("out", "");
  mpm::Model pm;
  if (!pm.load(dir + "/" + get("model", "mpm.bin"), false)) { std::fprintf(stderr, "cannot load model\n"); return 1; }
  // bit base rates from the fingerprint file (molecule level)
  std::vector<char> b = slurp(dir + "/" + lib + "_" + fp + ".bin");
  const char* p = b.data();
  uint32_t n, nBits; std::memcpy(&n, p, 4); std::memcpy(&nBits, p + 4, 4); p += 8;
  std::vector<double> cnt(nBits, 0.0);
  for (uint32_t i = 0; i < n; ++i) {
    uint16_t c; std::memcpy(&c, p, 2); p += 2;
    for (int k = 0; k < c; ++k) { uint16_t bb; std::memcpy(&bb, p + 2 * k, 2); cnt[bb] += 1; }
    p += 2 * size_t(c);
  }
  tfm::Model tm;
  tm.c.d = pm.c.d; tm.c.L = pm.c.L; tm.c.H = pm.c.H; tm.c.ff = pm.c.ff; tm.c.demb = pm.c.demb; tm.c.maxPeaks = pm.c.maxPeaks;
  tm.c.nAd = pm.c.nAd; tm.c.nIn = pm.c.nIn; tm.c.lo = pm.c.lo; tm.c.hi = pm.c.hi;
  tm.c.HH = geti("HH", 2048); tm.c.NB = int(nBits);
  std::vector<float> bias(nBits);
  for (uint32_t k = 0; k < nBits; ++k) { double q = std::min(0.999, std::max(1e-5, cnt[k] / std::max(1u, n))); bias[k] = float(std::log(q / (1 - q))); }
  tm.init(uint64_t(geti("seed", 31)), bias);
  // vocabulary (as tfm_train loadVocab)
  {
    std::ifstream f(dir + "/" + fp + ".bin", std::ios::binary);
    uint32_t magic = 0, nv = 0; int32_t rE = -1, rF = -1;
    if (f) { f.read((char*)&magic, 4); f.read((char*)&rE, 4); f.read((char*)&rF, 4); f.read((char*)&nv, 4); }
    if (magic == 0x32565046) { tm.rE = rE; tm.rF = rF; tm.vocab.resize(nv); f.read((char*)tm.vocab.data(), 8 * size_t(nv)); }
    else std::printf("note: no vocabulary %s.bin\n", fp.c_str());
  }
  const int64_t enc = pm.lnfb.off + pm.lnfb.n;
  if (tm.lnfb.off + tm.lnfb.n != enc) { std::fprintf(stderr, "encoder layout mismatch\n"); return 1; }
  std::vector<real> hw(pm.nP);
  mpm::download(pm.W, hw.data(), pm.nP);
  std::vector<tfm::real> tw(hw.begin(), hw.begin() + enc);
  tfm::upload(tm.W, tw.data(), enc);
  tm.save(dir + "/" + out, false);
  std::printf("exported encoder (%lld params) + fresh head (HH %d, NB %u) -> %s\n", (long long)enc, tm.c.HH, nBits, out.c_str());
  return 0;
}

// per-molecule mean logits of a TFM1 model (tfm_model.hpp) for a query set, in fpnet_ggml's _mol.bin format
// (i32 nmol, i32 nbits, per molecule i32 mol + f32[nbits]) -> scored on the window yardstick by
// fpown\build\win_eval.exe models=<file>:<vocab>. Featurisation = tfm_train's buildBatch (gFpnet off).
int molLogitsMode() {
  std::string dir = get("dir", "C:\\prog\\casmi"), qry = get("qry", "hcqry"), out = get("out", "tfm_mol.bin");
  const int B = geti("B", 128);
  tfm::Model m;
  if (!m.load(dir + "/" + get("model", "tfm.bin"), false)) { std::fprintf(stderr, "cannot load TFM1 model\n"); return 1; }
  Store st = loadSpectra(dir, qry);
  const int NB = m.c.NB;
  std::vector<std::vector<double>> sum(st.name.size());
  std::vector<int> cnt(st.name.size(), 0);
  std::vector<int> spec;
  for (size_t s = 0; s < st.head.size(); ++s) if (st.head[s].n > 0 && st.head[s].mol >= 0) spec.push_back(int(s));
  m.reserve(B, m.c.maxPeaks + 1);
  std::vector<tfm::real> z;
  for (size_t s0 = 0; s0 < spec.size(); s0 += B) {
    size_t s1 = std::min(spec.size(), s0 + B);
    tfm::HostBatch hb;
    const int nb = int(s1 - s0);
    std::vector<std::vector<float>> Mz(nb), I(nb);
    int mx = 1;
    for (int b = 0; b < nb; ++b) {
      const Head& h = st.head[spec[s0 + b]];
      prepPeaks(h, st.mzs(h), st.its(h), m.c.maxPeaks, Mz[b], I[b]);
      mx = std::max(mx, int(Mz[b].size()));
    }
    hb.B = nb; hb.N = 1 + mx;
    hb.mz.assign(size_t(nb) * mx, 0.f); hb.it.assign(size_t(nb) * mx, 0.f);
    hb.prec.resize(nb); hb.mode.resize(nb); hb.np.resize(nb); hb.ad.resize(nb); hb.in.resize(nb);
    for (int b = 0; b < nb; ++b) {
      const Head& h = st.head[spec[s0 + b]];
      std::copy(Mz[b].begin(), Mz[b].end(), hb.mz.begin() + size_t(b) * mx);
      std::copy(I[b].begin(), I[b].end(), hb.it.begin() + size_t(b) * mx);
      hb.np[b] = int(Mz[b].size()); hb.prec[b] = h.precursor;
      decodeTag(h.tag, hb.ad[b], hb.in[b], hb.mode[b]);
    }
    m.setBatch(hb);
    m.forward(false, 0.f, 0);
    z.resize(size_t(nb) * NB);
    tfm::download(m.Z, z.data(), int64_t(z.size()));
    for (int b = 0; b < nb; ++b) {
      int mol = st.head[spec[s0 + b]].mol;
      auto& v = sum[mol];
      if (v.empty()) v.assign(NB, 0.0);
      for (int k = 0; k < NB; ++k) v[k] += double(z[size_t(b) * NB + k]);
      ++cnt[mol];
    }
  }
  std::ofstream f(dir + "/" + out, std::ios::binary);
  int32_t nm = 0;
  for (int c : cnt) nm += c > 0;
  int32_t nb32 = NB;
  f.write((char*)&nm, 4); f.write((char*)&nb32, 4);
  std::vector<float> row(NB);
  for (size_t mo = 0; mo < sum.size(); ++mo) {
    if (!cnt[mo]) continue;
    int32_t id = int32_t(mo);
    for (int k = 0; k < NB; ++k) row[k] = float(sum[mo][k] / cnt[mo]);
    f.write((char*)&id, 4); f.write((char*)row.data(), 4 * size_t(NB));
  }
  std::printf("wrote %s: %d molecules x %d bits\n", out.c_str(), nm, NB);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: pt_train gradcheck|train|nbr|embed|export key=value...\n"); return 1; }
  std::string mode = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string s = argv[i];
    size_t e = s.find('=');
    if (e != std::string::npos) A[s.substr(0, e)] = s.substr(e + 1);
  }
  parallel_cap() = unsigned(geti("threads", 4));
  std::printf("pt_train %s on %s\n", mode.c_str(), bk::backend_name());
  if (mode == "gradcheck") return gradcheck();
  if (mode == "train") return train();
  if (mode == "nbr") return nbrMode();
  if (mode == "embed") return embedMode();
  if (mode == "export") return exportMode();
  if (mode == "mollogits") return molLogitsMode();
  std::fprintf(stderr, "unknown mode\n");
  return 1;
}
