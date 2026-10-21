// cnn_train: train the CNN reverse model (cnn_model.hpp, casmi/cnn_STATUS.md) on a
// spectral library with fp2/fp3 vocabulary targets, same data, targets, held-out
// split and Platt calibration as rev3_train (REV2), so the two are comparable.
//
//   cnn_train train dir=C:\prog\casmi lib=wlib fp=v4k out=cnn_a.bin epochs=12 B=512 lr=6e-4
//             drop=0.4 hold=20 seed=31 warm=500 threads=6 maxSteps=0
//             F=100 Lf=1024 Ln=512 C0=32 k=5 chans=64,64,128,128 flat=1 H=3072 pairK=60
//   cnn_train calib dir=... model=a.bin lib=wlib fp=v4k hold=20 out=b.bin
//   cnn_train cputime dir=... model=a.bin qry=hcqry threads=4 [max=2000] [B=64]   (CPU build)
//   cnn_train speed  (synthetic config sweep timing on the current backend)
// Output: the model after every epoch (uncalibrated) and at the end with calibration.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "parallel.hpp"
#include "cnn_model.hpp"

using namespace cnn;

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
  std::vector<std::string> name;
  const float* mzs(const Head& h) const { return mz.data() + h.off; }
  const float* its(const Head& h) const { return inten.data() + h.off; }
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

bool loadVocab(const std::string& path, Model& m) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  uint32_t magic, n;
  int32_t rE, rF;
  f.read(reinterpret_cast<char*>(&magic), 4);
  f.read(reinterpret_cast<char*>(&rE), 4);
  f.read(reinterpret_cast<char*>(&rF), 4);
  f.read(reinterpret_cast<char*>(&n), 4);
  if (magic != 0x32565046) return false;
  m.rE = rE; m.rF = rF;
  m.vocab.resize(n);
  f.read(reinterpret_cast<char*>(m.vocab.data()), 8 * size_t(n));
  return true;
}

uint64_t nameHash(const std::string& s) {  // identical to rev3_train (same held-out molecules)
  uint64_t h = 1469598103934665603ull;
  for (char c : s) h = (h ^ uint8_t(c)) * 1099511628211ull;
  return h ^ (h >> 29);
}

// Mean predicted probabilities per molecule (averaged over its spectra).
std::unordered_map<int, std::vector<float>> predictByMol(Model& m, const Store& q, const std::vector<int>& specIdx,
                                                         int B = 256) {
  std::unordered_map<int, std::vector<float>> sum;
  std::unordered_map<int, int> cnt;
  std::vector<Sample> ss;
  std::vector<real> P;
  for (size_t s0 = 0; s0 < specIdx.size(); s0 += B) {
    size_t s1 = std::min(specIdx.size(), s0 + B);
    ss.assign(s1 - s0, Sample());
    ::parallel_for(int64_t(s1 - s0), [&](int64_t i) {
      const Head& h = q.head[specIdx[s0 + i]];
      featurize(m.cfg, h, q.mzs(h), q.its(h), ss[i]);
    });
    m.setBatch(ss, false);
    m.forward(int(s1 - s0), false, 0.f, 0);
    download(m.P, P, int64_t(s1 - s0) * m.NB);
    for (size_t s = s0; s < s1; ++s) {
      int mol = q.head[specIdx[s]].mol;
      auto& v = sum[mol];
      if (v.empty()) v.assign(m.NB, 0.f);
      for (int i = 0; i < m.NB; ++i) v[i] += float(P[(s - s0) * m.NB + i]);
      ++cnt[mol];
    }
  }
  for (auto& kv : sum)
    for (auto& x : kv.second) x /= float(cnt[kv.first]);
  return sum;
}

// ---- Platt calibration on held-out molecules (same procedure as rev3_train) ----
void calibrate(Model& m, const Store& lib, const Fps& fps, int hold, int maxMols) {
  std::vector<int> spec;
  std::unordered_map<int, int> nMol;
  for (size_t s = 0; s < lib.head.size(); ++s) {
    int mol = lib.head[s].mol;
    if (lib.head[s].n <= 0 || fps.ptr[mol + 1] == fps.ptr[mol]) continue;
    if (nameHash(lib.name[mol]) % uint64_t(hold) != 0) continue;
    if (!nMol.count(mol) && int(nMol.size()) >= maxMols) continue;
    ++nMol[mol];
    spec.push_back(int(s));
  }
  std::printf("calibration: %zu held-out molecules, %zu spectra\n", nMol.size(), spec.size());
  auto probs = predictByMol(m, lib, spec);
  std::vector<int> mols;
  for (auto& kv : probs) mols.push_back(kv.first);
  std::sort(mols.begin(), mols.end());
  const int NB = m.NB;
  const size_t N = mols.size();
  std::vector<float> X(N * NB);
  std::vector<uint8_t> Y(N * NB, 0);
  for (size_t i = 0; i < N; ++i) {
    const auto& p = probs[mols[i]];
    for (int b = 0; b < NB; ++b) {
      float pi = std::min(1.f - 1e-6f, std::max(1e-6f, p[b]));
      X[i * NB + b] = std::log(pi / (1.f - pi));
    }
    int mol = mols[i];
    for (uint32_t k = fps.ptr[mol]; k < fps.ptr[mol + 1]; ++k) Y[i * NB + fps.bits[k]] = 1;
  }
  m.calA.assign(NB, 1.f); m.calC.assign(NB, 0.f); m.quality.assign(NB, 0.f);
  std::vector<double> llr(NB), llc(NB);
  ::parallel_for(NB, [&](int64_t b) {
    double a = 1, c = 0;
    const double lam = 1.0;
    for (int it = 0; it < 30; ++it) {
      double ga = lam * (a - 1), gc = lam * c, haa = lam, hac = 0, hcc = lam;
      for (size_t i = 0; i < N; ++i) {
        double x = X[i * NB + b], z = a * x + c;
        double p = 1.0 / (1.0 + std::exp(-z));
        double r = p - Y[i * NB + b], w = p * (1 - p);
        ga += r * x; gc += r;
        haa += w * x * x; hac += w * x; hcc += w;
      }
      double det = haa * hcc - hac * hac;
      if (det <= 1e-12) break;
      double da = (hcc * ga - hac * gc) / det, dc = (haa * gc - hac * ga) / det;
      a -= da; c -= dc;
      if (std::fabs(da) + std::fabs(dc) < 1e-6) break;
    }
    int tp = 0, fp = 0, fn = 0;
    double lr = 0, lc = 0;
    for (size_t i = 0; i < N; ++i) {
      double x = X[i * NB + b];
      double pc = 1.0 / (1.0 + std::exp(-(a * x + c)));
      double pr = 1.0 / (1.0 + std::exp(-x));
      int y = Y[i * NB + b];
      bool on = pc >= 0.5;
      tp += on && y; fp += on && !y; fn += !on && y;
      lr -= y ? std::log(std::max(1e-9, pr)) : std::log(std::max(1e-9, 1 - pr));
      lc -= y ? std::log(std::max(1e-9, pc)) : std::log(std::max(1e-9, 1 - pc));
    }
    m.calA[b] = float(a); m.calC[b] = float(c);
    m.quality[b] = tp ? float(2.0 * tp / (2.0 * tp + fp + fn)) : 0.f;
    llr[b] = lr; llc[b] = lc;
  });
  double llRaw = 0, llCal = 0, meanA = 0;
  int nGood = 0;
  for (int b = 0; b < NB; ++b) { llRaw += llr[b]; llCal += llc[b]; nGood += m.quality[b] >= 0.5f; meanA += m.calA[b]; }
  std::printf("calibration: held-out BCE/mol raw %.3f -> Platt %.3f; mean a %.3f; bits with F1>=0.5: %d/%d\n",
              llRaw / N, llCal / N, meanA / NB, nGood, NB);
}

std::map<std::string, std::string> A;
std::string get(const std::string& k, const std::string& d) { auto it = A.find(k); return it == A.end() ? d : it->second; }
std::vector<int> ints(const std::string& s) {
  std::vector<int> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::atoi(t.c_str()));
  return v;
}
void cfgFromArgs(Cfg& c) {
  c.F = std::atoi(get("F", std::to_string(c.F)).c_str());
  c.Lf = std::atoi(get("Lf", std::to_string(c.Lf)).c_str());
  c.Ln = c.Lp = std::atoi(get("Ln", std::to_string(c.Ln)).c_str());
  c.C0 = std::atoi(get("C0", std::to_string(c.C0)).c_str());
  c.k = std::atoi(get("k", std::to_string(c.k)).c_str());
  if (A.count("chans")) c.chans = ints(A["chans"]);
  c.flat = std::atoi(get("flat", std::to_string(c.flat)).c_str());
  if (A.count("H")) c.H = ints(A["H"]);
  c.pairK = std::atoi(get("pairK", std::to_string(c.pairK)).c_str());
  c.maxPeaks = std::atoi(get("maxPeaks", std::to_string(c.maxPeaks)).c_str());
}
void printCfg(const Model& m) {
  const Cfg& c = m.cfg;
  std::printf("cnn cfg: F %d (%.3f Da) off %.2f Lf %d Ln=Lp %d C0 %d k %d chans", c.F, 1.0 / c.F, c.off, c.Lf, c.Ln, c.C0, c.k);
  for (int x : c.chans) std::printf(" %d", x);
  std::printf(" flat %d head", c.flat);
  for (int h : c.H) std::printf(" %d", h);
  std::printf(" NB %d pairK %d maxPeaks %d | head input %lld\n", m.NB, c.pairK, c.maxPeaks, (long long)m.D());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: cnn_train train|calib|cputime|speed key=value...\n"); return 1; }
  std::string mode = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string s = argv[i];
    size_t e = s.find('=');
    if (e != std::string::npos) A[s.substr(0, e)] = s.substr(e + 1);
  }
  std::printf("backend: %s\n", bk::backend_name());
  parallel_cap() = unsigned(std::atoi(get("threads", "6").c_str()));
  std::string dir = get("dir", "C:\\prog\\casmi");

  if (mode == "cputime") {
    Model m;
    if (!m.load(dir + "/" + get("model", ""))) { std::fprintf(stderr, "cannot load model\n"); return 1; }
    printCfg(m);
    Store q = loadSpectra(dir, get("qry", "hcqry"));
    int mx = std::atoi(get("max", "2000").c_str()), B = std::atoi(get("B", "64").c_str());
    std::vector<int> spec;
    for (size_t s = 0; s < q.head.size() && int(spec.size()) < mx; ++s) if (q.head[s].n > 0) spec.push_back(int(s));
    auto t0 = std::chrono::steady_clock::now();
    auto P = predictByMol(m, q, spec, B);
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("cputime: %zu spectra in %.2fs = %.1f ms/spectrum (threads %u, B %d)\n", spec.size(), sec,
                1000.0 * sec / spec.size(), parallel_cap(), B);
    return 0;
  }

  if (mode == "speed") {  // forward+backward step time for the current config on random items
    Model m;
    cfgFromArgs(m.cfg);
    m.NB = std::atoi(get("NB", "4096").c_str());
    m.build(1, true);
    printCfg(m);
    std::printf("params %.1fM\n", m.nParams() / 1e6);
    int B = std::atoi(get("B", "512").c_str());
    std::vector<Sample> ss(B);
    uint64_t r = 99;
    auto rnd = [&]() { r ^= r << 13; r ^= r >> 7; r ^= r << 17; return r; };
    for (auto& s : ss) {
      int n = 60;
      std::vector<float> mz(n), it(n);
      for (int i = 0; i < n; ++i) { mz[i] = 50.f + float(rnd() % 40000) * 0.01f; it[i] = float(rnd() % 1000) + 1.f; }
      Head h{460.f, 0, 0, n, 0};
      featurize(m.cfg, h, mz.data(), it.data(), s);
    }
    std::vector<real> T(size_t(B) * m.NB, 0.f);
    for (size_t i = 0; i < T.size(); i += 37) T[i] = 1.f;
    upload(m.T, T);
    auto now = [] { bk::sync(); return std::chrono::steady_clock::now(); };
    auto ms = [](auto a, auto b) { return 1000.0 * std::chrono::duration<double>(b - a).count(); };
    Model::HostBatch hb;
    for (int it = 0; it < 8; ++it) {
      auto t0 = now();
      m.prepBatch(ss, true, hb);
      auto t1 = now();
      m.uploadBatch(hb);
      auto t2 = now();
      m.forward(B, true, 0.4f, it);
      auto t3 = now();
      m.backward(B);
      auto t4 = now();
      for (Param* p : m.params()) p->adam(1e-4f, it + 1, 0.f);
      auto t5 = now();
      if (it >= 2)
        std::printf("step %d: prep(host) %.1f  upload %.1f  fwd %.1f  bwd %.1f  adam %.1f  = GPU-thread %.1f ms\n", it,
                    ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4), ms(t4, t5), ms(t1, t5));
    }
    return 0;
  }

  std::string lib = get("lib", "wlib"), fp = get("fp", "v4k");
  if (mode == "calib") {
    Model m;
    if (!m.load(dir + "/" + get("model", ""))) { std::fprintf(stderr, "cannot load model\n"); return 1; }
    Store ls = loadSpectra(dir, lib);
    Fps fps = loadFps(dir + "/" + lib + "_" + fp + ".bin");
    calibrate(m, ls, fps, std::atoi(get("hold", "20").c_str()), std::atoi(get("maxMols", "20000").c_str()));
    m.save(dir + "/" + get("out", "calib.bin"));
    return 0;
  }
  if (mode != "train") { std::fprintf(stderr, "unknown mode\n"); return 1; }

  std::string outPath = get("out", "cnn.bin");
  int epochs = std::atoi(get("epochs", "12").c_str()), B = std::atoi(get("B", "512").c_str());
  float lr = float(std::atof(get("lr", "6e-4").c_str()));
  float dropout = float(std::atof(get("drop", "0.4").c_str()));
  float wd = float(std::atof(get("wd", "0").c_str()));
  uint64_t seed = uint64_t(std::atoll(get("seed", "31").c_str()));
  int warm = std::atoi(get("warm", "500").c_str());
  int hold = std::atoi(get("hold", "20").c_str());
  int64_t maxSteps = std::atoll(get("maxSteps", "0").c_str());

  Model m;
  cfgFromArgs(m.cfg);
  Store lib_s = loadSpectra(dir, lib);
  Fps fps = loadFps(dir + "/" + lib + "_" + fp + ".bin");
  m.NB = int(fps.nBits);
  if (!loadVocab(dir + "/" + fp + ".bin", m)) { std::fprintf(stderr, "cannot read vocab %s.bin\n", fp.c_str()); return 1; }

  std::vector<int> idx, vidx;
  for (size_t s = 0; s < lib_s.head.size(); ++s) {
    int mol = lib_s.head[s].mol;
    if (lib_s.head[s].n <= 0 || fps.ptr[mol + 1] == fps.ptr[mol]) continue;
    if (hold > 0 && nameHash(lib_s.name[mol]) % uint64_t(hold) == 0) { if (vidx.size() < 4096 && s % 7 == 0) vidx.push_back(int(s)); continue; }
    idx.push_back(int(s));
  }
  if (A.count("init")) {
    Cfg keep = m.cfg; (void)keep;
    if (!m.load(dir + "/" + A["init"], true) || m.NB != int(fps.nBits)) { std::fprintf(stderr, "bad init model\n"); return 1; }
    loadVocab(dir + "/" + fp + ".bin", m);
    std::printf("init from %s\n", A["init"].c_str());
  } else {
    m.build(seed, true);
  }
  printCfg(m);
  std::printf("training spectra: %zu, validation spectra (held-out molecules) %zu, params %.1fM\n", idx.size(),
              vidx.size(), m.nParams() / 1e6);
  m.reserve(B);

  struct Batch { std::vector<Sample> ss; std::vector<real> T; Model::HostBatch hb; };
  auto prep = [&](Batch& bt, const std::vector<int>& from, size_t start, int n) {
    bt.ss.resize(n);
    bt.T.assign(size_t(n) * m.NB, 0.f);
    ::parallel_for(n, [&](int64_t b) {
      const Head& h = lib_s.head[from[start + b]];
      featurize(m.cfg, h, lib_s.mzs(h), lib_s.its(h), bt.ss[b]);
      for (uint32_t k = fps.ptr[h.mol]; k < fps.ptr[h.mol + 1]; ++k) bt.T[size_t(b) * m.NB + fps.bits[k]] = 1.f;
    });
    m.prepBatch(bt.ss, true, bt.hb);
  };
  auto bceOf = [&](const std::vector<real>& T, int n) {
    std::vector<real> Z;
    download(m.Zo, Z, int64_t(n) * m.NB);
    double l = 0;
    for (size_t i = 0; i < Z.size(); ++i) {
      double z = Z[i];
      l += (z > 0 ? z + std::log1p(std::exp(-z)) : std::log1p(std::exp(z))) - T[i] * z;
    }
    return l / n;
  };
  auto validate = [&]() {
    double tot = 0; int n = 0;
    Batch vb;
    for (size_t s0 = 0; s0 + B <= vidx.size(); s0 += B) {
      prep(vb, vidx, s0, B);
      m.setBatch(vb.ss, false);
      m.forward(B, false, 0.f, 0);
      tot += bceOf(vb.T, B) * B; n += B;
    }
    return n ? tot / n : 0.0;
  };

  int stepsPerEpoch = int(idx.size() / size_t(B));
  int64_t totalSteps = int64_t(stepsPerEpoch) * epochs, step = 0;
  if (maxSteps > 0) totalSteps = std::min<int64_t>(totalSteps, maxSteps);
  uint64_t rng = 12345 + seed * 7919;
  Batch cur, nxt;
  auto t0 = std::chrono::steady_clock::now();
  double runL = 0; int runN = 0;
  for (int ep = 0; ep < epochs && step < totalSteps; ++ep) {
    for (size_t i = idx.size(); i > 1; --i) {
      rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
      std::swap(idx[i - 1], idx[rng % i]);
    }
    prep(cur, idx, 0, B);
    for (int st = 0; st < stepsPerEpoch && step < totalSteps; ++st, ++step) {
      std::thread th;
      if (st + 1 < stepsPerEpoch) th = std::thread([&, st]() { prep(nxt, idx, size_t(st + 1) * B, B); });
      m.uploadBatch(cur.hb);
      upload(m.T, cur.T);
      m.forward(B, true, dropout, uint64_t(step));
      m.backward(B);
      float t = float(step) / float(std::max<int64_t>(1, totalSteps));
      float lrNow = step < warm ? lr * float(step + 1) / float(warm) : lr * 0.5f * (1.f + std::cos(3.14159265f * t));
      for (Param* p : m.params()) p->adam(lrNow, int(step + 1), wd);
      if (st % 100 == 0) { runL += bceOf(cur.T, B); ++runN; }
      if (st % 1000 == 0) {
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("ep %d step %d/%d  train loss/spectrum %.3f (mean of %d)  lr %.6f  %.0fs\n", ep, st, stepsPerEpoch,
                    runL / std::max(1, runN), runN, lrNow, sec);
        runL = 0; runN = 0;
        std::fflush(stdout);
      }
      if (th.joinable()) th.join();
      std::swap(cur, nxt);
    }
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("== epoch %d done (%.0fs)  held-out loss/spectrum %.3f\n", ep + 1, sec, validate());
    m.save(dir + "/" + outPath);
    std::fflush(stdout);
  }
  if (hold > 0) calibrate(m, lib_s, fps, hold, 20000);
  m.save(dir + "/" + outPath);
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("wrote %s (%.0fs total)\n", outPath.c_str(), sec);
  return 0;
}
