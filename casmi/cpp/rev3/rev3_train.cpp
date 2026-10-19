// rev3_train: copy of rev/rev2_train.cpp (unchanged model/format) with eval=- (no
// phys_ana dumps needed, any fp2/fp3 vocabulary) and threads=N (CPU cap).
// Train / evaluate the reverse model v2 (rev2_model.hpp).
//
//   rev2_train train key=value...
//     dir=C:\prog\casmi lib=xlib fp=ecfp|<vocabtag> out=<model.bin>
//     epochs=15 B=512 lr=6e-4 drop=0.4 H=3072,3072 bin=0.1 frag=1000 nl=300
//     ls=0 wd=0 pw=1 seed=31 warm=500 hold=0 evalEvery=5
//     eval=cqry:f21_cqry.txt,tqry:f21_tqry.txt cand=fcand
//   rev2_train eval dir=... models=a.bin:ecfp,b.bin:<vocabtag> eval=... cand=fcand
//   rev2_train calib dir=... model=a.bin lib=xlib fp=... hold=20 out=b.bin
//
// fp=ecfp: targets <lib>_ecfp.bin (ecfp.h r2 2048, fp_prep); otherwise
// <lib>_<tag>.bin and vocabulary <tag>.bin from fp2_prep, embedded in the
// model. hold=K leaves out every molecule whose name hashes to 0 mod K
// (structure-disjoint) and, after training, fits per-bit Platt calibration
// and a held-out F1 per bit on them (stored in the model).
//
// Evaluation reuses phys_ana feature dumps ("Q name truth n" + n lines
// "phys mlp rev libSim anaP inLib key"): the candidates and every other
// score are fixed, only the reverse-model column is recomputed, so the
// fused MRR is exactly phys_ana's  z(phys) + 0.25 z(mlp) + wr z(rev)
// + 8 lib^32 + 24 anaP  (variant 4, p3) for any new model in seconds.
// Candidate fingerprints: <cand>_mols.txt / <cand>_<fp>.bin.
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
#include "rev3_model.hpp"

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

uint64_t nameHash(const std::string& s) {
  uint64_t h = 1469598103934665603ull;
  for (char c : s) h = (h ^ uint8_t(c)) * 1099511628211ull;
  return h ^ (h >> 29);
}

// ---- evaluation on phys_ana feature dumps ------------------------------------
struct Cand { float phys, mlp, rev, lib, ana; int key; };
struct DQ { std::string name; int mol = -1, truth = -1; std::vector<Cand> c; };
struct EvalSet {
  std::string name;
  Store q;
  std::vector<DQ> qs;
};

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

struct CandFps {
  std::unordered_map<std::string, int> keyIdx;
  std::map<std::string, Fps> byFp;
};

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

// Mean predicted probabilities per molecule (averaged over its spectra).
std::unordered_map<int, std::vector<float>> predictByMol(Model& m, const Store& q, const std::vector<int>& specIdx) {
  std::unordered_map<int, std::vector<float>> sum;
  std::unordered_map<int, int> cnt;
  const int B = 256;
  std::vector<std::vector<std::pair<int, float>>> rows;
  std::vector<float> P;
  for (size_t s0 = 0; s0 < specIdx.size(); s0 += B) {
    size_t s1 = std::min(specIdx.size(), s0 + B);
    rows.assign(s1 - s0, {});
    ::parallel_for(int64_t(s1 - s0), [&](int64_t i) {
      const Head& h = q.head[specIdx[s0 + i]];
      featurize(m.fc, h, q.mzs(h), q.its(h), rows[i]);
    });
    m.setBatch(rows, false);
    m.forward(int(s1 - s0), false, 0.f, 0);
    download(m.P, P, int64_t(s1 - s0) * m.NB);
    for (size_t s = s0; s < s1; ++s) {
      int mol = q.head[specIdx[s]].mol;
      auto& v = sum[mol];
      if (v.empty()) v.assign(m.NB, 0.f);
      for (int i = 0; i < m.NB; ++i) v[i] += P[(s - s0) * m.NB + i];
      ++cnt[mol];
    }
  }
  for (auto& kv : sum)
    for (auto& x : kv.second) x /= float(cnt[kv.first]);
  return sum;
}

const char* VNAME[] = {"logit", "logit_c2", "logp", "tanimoto", "cal", "cal_q50", "cal_wq", "cal_wq2"};
constexpr int NV = 8;

// Candidate score of variant v from probabilities p (mean over spectra).
struct Scorer {
  std::vector<std::vector<float>> u;  // per linear variant, per bit
  std::vector<float> p;
  double sumP = 0;
  void prep(const Model& m, const std::vector<float>& pm) {
    p = pm;
    u.assign(NV, std::vector<float>(m.NB, 0.f));
    sumP = 0;
    bool cal = int(m.calA.size()) == m.NB;
    for (int b = 0; b < m.NB; ++b) {
      float pi = std::min(1.f - 1e-6f, std::max(1e-6f, p[b]));
      float lg = std::log(pi / (1.f - pi));
      u[0][b] = lg;
      float pc = std::min(0.99f, std::max(0.01f, p[b]));
      u[1][b] = std::log(pc / (1.f - pc));
      u[2][b] = std::log(pi);
      sumP += p[b];
      if (cal) {
        float z = m.calA[b] * lg + m.calC[b];
        z = std::min(13.8f, std::max(-13.8f, z));
        u[4][b] = z;
        u[5][b] = m.quality[b] >= 0.5f ? z : 0.f;
        u[6][b] = m.quality[b] * z;
        u[7][b] = m.quality[b] * m.quality[b] * z;
      }
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

struct EvalModel { Model* m; const Fps* fps; };

// Prints rev-alone and fused MRR for each scoring variant. With several
// models, each model's per-query z-scored scores are summed (ensemble).
double evaluateSets(std::vector<EvalModel>& models, std::vector<EvalSet>& sets, bool verbose) {
  double ret = 0;
  bool cal = true;
  for (auto& em : models) cal = cal && int(em.m->calA.size()) == em.m->NB;
  int nv = cal ? NV : 4;
  for (auto& e : sets) {
    std::vector<int> spec;
    std::vector<char> want(e.q.name.size(), 0);
    for (auto& q : e.qs) if (q.mol >= 0) want[q.mol] = 1;
    for (size_t s = 0; s < e.q.head.size(); ++s)
      if (e.q.head[s].mol >= 0 && want[e.q.head[s].mol] && e.q.head[s].n > 0) spec.push_back(int(s));
    std::vector<std::unordered_map<int, std::vector<float>>> probs;
    for (auto& em : models) probs.push_back(predictByMol(*em.m, e.q, spec));
    // revScore[v][qi][ci]
    std::vector<std::vector<std::vector<float>>> rs(nv, std::vector<std::vector<float>>(e.qs.size()));
    for (size_t qi = 0; qi < e.qs.size(); ++qi) {
      const DQ& q = e.qs[qi];
      for (int v = 0; v < nv; ++v) rs[v][qi].assign(q.c.size(), 0.f);
      for (size_t mi = 0; mi < models.size(); ++mi) {
        auto it = probs[mi].find(q.mol);
        if (it == probs[mi].end()) continue;
        Scorer sc;
        sc.prep(*models[mi].m, it->second);
        const Fps& F = *models[mi].fps;
        for (int v = 0; v < nv; ++v) {
          std::vector<float> s(q.c.size(), 0.f);
          for (size_t ci = 0; ci < q.c.size(); ++ci) {
            int k = q.c[ci].key;
            if (k < 0) continue;
            s[ci] = sc.score(v, F.bits.data() + F.ptr[k], int(F.ptr[k + 1] - F.ptr[k]));
          }
          if (models.size() > 1) s = zscore(s);
          for (size_t ci = 0; ci < s.size(); ++ci) rs[v][qi][ci] += s[ci];
        }
      }
    }
    auto fused = [&](int v, float wr) {  // v = -1: the dump's own rev column
      double tot = 0;
      for (size_t qi = 0; qi < e.qs.size(); ++qi) {
        const DQ& q = e.qs[qi];
        size_t n = q.c.size();
        std::vector<float> ph(n), ml(n), rv(n);
        for (size_t i = 0; i < n; ++i) { ph[i] = q.c[i].phys; ml[i] = q.c[i].mlp; rv[i] = v < 0 ? q.c[i].rev : rs[v][qi][i]; }
        auto zp = zscore(ph), zm = zscore(ml), zr = zscore(rv);
        std::vector<float> f(n);
        for (size_t i = 0; i < n; ++i)
          f[i] = (wr < 0 ? zr[i] : zp[i] + 0.25f * zm[i] + wr * zr[i] +
                  8.f * std::pow(q.c[i].lib, 32.f) + 24.f * q.c[i].ana);
        tot += rr(f, q.truth);
      }
      return tot / std::max<size_t>(1, e.qs.size());
    };
    const float WR[] = {0.5f, 1.f, 1.5f, 2.f, 3.f};
    if (verbose) {
      std::printf("  [%s] dump rev column: alone %.4f fused(wr=1) %.4f | wr=0 %.4f\n", e.name.c_str(), fused(-1, -1.f),
                  fused(-1, 1.f), fused(-1, 0.f));
    }
    for (int v = 0; v < nv; ++v) {
      if (!verbose && v > 0) break;
      std::printf("  [%s] %-9s alone %.4f  fused wr:", e.name.c_str(), VNAME[v], [&] {
        double tot = 0;
        for (size_t qi = 0; qi < e.qs.size(); ++qi) tot += rr(rs[v][qi], e.qs[qi].truth);
        return tot / std::max<size_t>(1, e.qs.size());
      }());
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

// ---- Platt calibration on held-out molecules -------------------------------
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
  double llRaw = 0, llCal = 0;
  std::vector<double> llr(NB), llc(NB);
  ::parallel_for(NB, [&](int64_t b) {
    double a = 1, c = 0;
    const double lam = 1.0;  // ridge toward (1, 0)
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
  int nGood = 0;
  double meanA = 0;
  for (int b = 0; b < NB; ++b) { llRaw += llr[b]; llCal += llc[b]; nGood += m.quality[b] >= 0.5f; meanA += m.calA[b]; }
  std::printf("calibration: held-out BCE/mol raw %.3f -> Platt %.3f; mean a %.3f; bits with F1>=0.5: %d/%d\n",
              llRaw / N, llCal / N, meanA / NB, nGood, NB);
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

std::string fpFile(const std::string& dir, const std::string& stem, const std::string& fp) {
  return dir + "/" + stem + "_" + fp + ".bin";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: rev2_train train|eval|calib key=value...\n"); return 1; }
  std::string mode = argv[1];
  auto A = parseArgs(argc, argv);
  std::printf("backend: %s\n", bk::backend_name());
  parallel_cap() = unsigned(std::atoi(get(A, "threads", "8").c_str()));
  std::string dir = get(A, "dir", "C:\\prog\\casmi");
  std::string candStem = get(A, "cand", "fcand");
  std::string evalSpec = get(A, "eval", "cqry:f21_cqry.txt,tqry:f21_tqry.txt");

  CandFps cf;
  {
    std::ifstream f(dir + "/" + candStem + "_mols.txt");
    std::string line;
    int i = 0;
    while (std::getline(f, line)) cf.keyIdx[line.substr(0, line.find('\t'))] = i++;
  }

  if (mode == "eval") {
    std::vector<std::unique_ptr<Model>> owned;
    std::vector<EvalModel> ems;
    std::stringstream ss(get(A, "models", ""));
    std::string item;
    while (std::getline(ss, item, ',')) {
      size_t c = item.find(':');
      std::string path = item.substr(0, c), fp = c == std::string::npos ? "ecfp" : item.substr(c + 1);
      owned.emplace_back(new Model());
      if (!owned.back()->load(dir + "/" + path)) { std::fprintf(stderr, "cannot load %s\n", path.c_str()); return 1; }
      if (!cf.byFp.count(fp)) cf.byFp.emplace(fp, loadFps(fpFile(dir, candStem, fp)));
      ems.push_back({owned.back().get(), &cf.byFp[fp]});
      Model& m = *owned.back();
      std::printf("model %s: bin %.3f frag %d nl %d, layers", path.c_str(), m.fc.bin, m.fc.nbFrag, m.fc.nbNL);
      for (int h : m.H) std::printf(" %d", h);
      std::printf(", %d bits, vocab %zu, calibrated %d\n", m.NB, m.vocab.size(), int(m.calA.size()) == m.NB);
    }
    auto sets = loadEval(dir, evalSpec, cf);
    evaluateSets(ems, sets, true);
    return 0;
  }

  if (mode == "calib") {
    std::string lib = get(A, "lib", "xlib"), fp = get(A, "fp", "ecfp");
    Model m;
    if (!m.load(dir + "/" + get(A, "model", ""))) { std::fprintf(stderr, "cannot load model\n"); return 1; }
    Store ls = loadSpectra(dir, lib);
    Fps fps = loadFps(fpFile(dir, lib, fp));
    calibrate(m, ls, fps, std::atoi(get(A, "hold", "20").c_str()), std::atoi(get(A, "maxMols", "20000").c_str()));
    m.save(dir + "/" + get(A, "out", "calib.bin"));
    return 0;
  }

  if (mode != "train") { std::fprintf(stderr, "unknown mode\n"); return 1; }
  std::string lib = get(A, "lib", "xlib"), fp = get(A, "fp", "ecfp"), outPath = get(A, "out", "rev2.bin");
  int epochs = std::atoi(get(A, "epochs", "15").c_str()), B = std::atoi(get(A, "B", "512").c_str());
  float lr = float(std::atof(get(A, "lr", "6e-4").c_str()));
  float dropout = float(std::atof(get(A, "drop", "0.4").c_str()));
  float ls = float(std::atof(get(A, "ls", "0").c_str()));
  float wd = float(std::atof(get(A, "wd", "0").c_str()));
  float pw = float(std::atof(get(A, "pw", "1").c_str()));
  uint64_t seed = uint64_t(std::atoll(get(A, "seed", "31").c_str()));
  int warm = std::atoi(get(A, "warm", "500").c_str());
  int hold = std::atoi(get(A, "hold", "0").c_str());
  int evalEvery = std::atoi(get(A, "evalEvery", "5").c_str());

  Model m;
  m.fc.bin = float(std::atof(get(A, "bin", "0.1").c_str()));
  m.fc.nbFrag = int(std::lround(std::atof(get(A, "frag", "1000").c_str()) / m.fc.bin));
  m.fc.nbNL = int(std::lround(std::atof(get(A, "nl", "300").c_str()) / m.fc.bin));
  m.fc.maxPeaks = std::atoi(get(A, "maxPeaks", "300").c_str());
  m.H.clear();
  {
    std::stringstream ss(get(A, "H", "3072,3072"));
    std::string t;
    while (std::getline(ss, t, ',')) m.H.push_back(std::atoi(t.c_str()));
  }
  Store lib_s = loadSpectra(dir, lib);
  Fps fps = loadFps(fpFile(dir, lib, fp));
  m.NB = int(fps.nBits);
  if (fp != "ecfp" && !loadVocab(dir + "/" + fp + ".bin", m)) { std::fprintf(stderr, "cannot read vocab %s.bin\n", fp.c_str()); return 1; }
  if (evalSpec == "-") evalSpec.clear();
  if (!evalSpec.empty()) cf.byFp.emplace(fp, loadFps(fpFile(dir, candStem, fp)));
  auto sets = loadEval(dir, evalSpec, cf);

  std::vector<int> idx;
  for (size_t s = 0; s < lib_s.head.size(); ++s) {
    int mol = lib_s.head[s].mol;
    if (lib_s.head[s].n <= 0 || fps.ptr[mol + 1] == fps.ptr[mol]) continue;
    if (hold > 0 && nameHash(lib_s.name[mol]) % uint64_t(hold) == 0) continue;
    idx.push_back(int(s));
  }
  std::printf("training spectra: %zu; input dim %d, layers", idx.size(), m.fc.dIn());
  for (int h : m.H) std::printf(" %d", h);
  std::printf(", %d bits\n", m.NB);

  // ---- rev3: within-window contrastive term (ctr=<decoy win stem>, see fp3_STATUS.md)
  //   ctr=d_wlib : <ctr>_win.bin (win_build sampleK on a query set of library molecules,
  //                ctrq=<that query stem> for the molecule names), decoy bits <ctr>_<fp>.bin
  //   lam=0.5 ctrS=0.2 : loss += lam * CE(softmax(ctrS * logit-sum score) over truth + decoys)
  //   init=<model.bin> : start from a trained model (fine-tune) instead of random init
  std::string ctrStem = get(A, "ctr", "");
  float lam = float(std::atof(get(A, "lam", "0.5").c_str())), ctrS = float(std::atof(get(A, "ctrS", "0.2").c_str()));
  int maxDec = std::atoi(get(A, "maxDec", "64").c_str());
  std::vector<std::vector<uint32_t>> decOf(lib_s.name.size());
  Fps dfp;
  if (!ctrStem.empty()) {
    std::unordered_map<std::string, int> libIdx;
    for (size_t i = 0; i < lib_s.name.size(); ++i) libIdx[lib_s.name[i]] = int(i);
    std::vector<std::string> qn;
    {
      std::ifstream f(dir + "/" + get(A, "ctrq", "dq") + "_mols.txt");
      std::string line;
      while (std::getline(f, line)) qn.push_back(line.substr(0, line.find('\t')));
    }
    std::vector<char> wb = slurp(dir + "/" + ctrStem + "_win.bin");
    const char* p = wb.data();
    uint32_t nq; std::memcpy(&nq, p, 4); p += 4;
    size_t nWith = 0, nDec = 0;
    for (uint32_t q = 0; q < nq; ++q) {
      int32_t mol, tr; uint32_t n;
      std::memcpy(&mol, p, 4); std::memcpy(&tr, p + 4, 4); std::memcpy(&n, p + 8, 4); p += 12;
      const uint32_t* ids = reinterpret_cast<const uint32_t*>(p); p += 4 * size_t(n);
      if (mol < 0 || mol >= int(qn.size())) continue;
      auto it = libIdx.find(qn[mol]);
      if (it == libIdx.end()) continue;
      auto& d = decOf[it->second];
      for (uint32_t k = 0; k < n && int(d.size()) < maxDec; ++k) if (int(k) != tr) d.push_back(ids[k]);
      if (!d.empty()) { ++nWith; nDec += d.size(); }
    }
    dfp = loadFps(dir + "/" + ctrStem + "_" + fp + ".bin");
    std::printf("contrastive: %zu library molecules with decoys (mean %.1f), lam %g, scale %g\n", nWith,
                double(nDec) / std::max<size_t>(1, nWith), lam, ctrS);
  }
  if (A.count("init")) {
    if (!m.load(dir + "/" + A["init"], true) || m.NB != int(fps.nBits)) { std::fprintf(stderr, "bad init model\n"); return 1; }
    std::printf("init from %s\n", A["init"].c_str());
    if (fp != "ecfp") loadVocab(dir + "/" + fp + ".bin", m);
  } else {
    m.init(seed);
  }
  m.reserve(B);
  bk::Buffer<float> gExtra;
  std::vector<float> Zh, Gh;
  double ctrLoss = 0, ctrAcc = 0; size_t ctrN = 0;
  int stepsPerEpoch = int(idx.size() / size_t(B));
  int totalSteps = stepsPerEpoch * epochs, step = 0;
  uint64_t rng = 12345 + seed * 7919;
  struct Batch { std::vector<std::vector<std::pair<int, float>>> rows; std::vector<float> T; std::vector<int> mol; };
  Batch cur, nxt;
  auto prep = [&](Batch& bt, int st) {
    bt.rows.resize(B);
    bt.mol.resize(B);
    bt.T.assign(size_t(B) * m.NB, 0.f);
    ::parallel_for(B, [&](int64_t b) {
      const Head& h = lib_s.head[idx[size_t(st) * B + b]];
      featurize(m.fc, h, lib_s.mzs(h), lib_s.its(h), bt.rows[b]);
      bt.mol[b] = h.mol;
      for (uint32_t k = fps.ptr[h.mol]; k < fps.ptr[h.mol + 1]; ++k) bt.T[size_t(b) * m.NB + fps.bits[k]] = 1.f;
    });
  };
  std::vector<float> P;
  auto t0 = std::chrono::steady_clock::now();
  EvalModel em{&m, &cf.byFp[fp]};
  std::vector<EvalModel> ems{em};
  for (int ep = 0; ep < epochs; ++ep) {
    for (size_t i = idx.size(); i > 1; --i) {
      rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
      std::swap(idx[i - 1], idx[rng % i]);
    }
    prep(cur, 0);
    for (int st = 0; st < stepsPerEpoch; ++st, ++step) {
      std::thread th;
      if (st + 1 < stepsPerEpoch) th = std::thread([&, st]() { prep(nxt, st + 1); });
      m.setBatch(cur.rows, true);
      upload(m.T, cur.T);
      m.forward(B, true, dropout, uint64_t(step));
      if (!ctrStem.empty() && lam > 0.f) {
        download(m.Zo, Zh, int64_t(B) * m.NB);
        Gh.assign(size_t(B) * m.NB, 0.f);
        std::vector<double> lb(B, 0.0), ab(B, 0.0);
        std::vector<char> has(B, 0);
        ::parallel_for(B, [&](int64_t b) {
          const auto& d = decOf[cur.mol[b]];
          if (d.empty()) return;
          has[b] = 1;
          const float* z = Zh.data() + size_t(b) * m.NB;
          float* g = Gh.data() + size_t(b) * m.NB;
          const int nc = int(d.size()) + 1;
          std::vector<double> sc(nc);
          auto bitsOf = [&](int c, const uint16_t*& bp, int& n) {
            if (c == 0) { bp = fps.bits.data() + fps.ptr[cur.mol[b]]; n = int(fps.ptr[cur.mol[b] + 1] - fps.ptr[cur.mol[b]]); }
            else { uint32_t u = d[c - 1]; bp = dfp.bits.data() + dfp.ptr[u]; n = int(dfp.ptr[u + 1] - dfp.ptr[u]); }
          };
          double mx = -1e30;
          for (int c = 0; c < nc; ++c) {
            const uint16_t* bp; int n; bitsOf(c, bp, n);
            double s = 0; for (int k = 0; k < n; ++k) s += z[bp[k]];
            sc[c] = ctrS * s; mx = std::max(mx, sc[c]);
          }
          double Z = 0; for (double& x : sc) { x = std::exp(x - mx); Z += x; }
          int better = 0;
          for (int c = 1; c < nc; ++c) better += sc[c] >= sc[0];
          lb[b] = -std::log(std::max(1e-30, sc[0] / Z)); ab[b] = better == 0;
          for (int c = 0; c < nc; ++c) {
            double coef = lam * ctrS * (sc[c] / Z - (c == 0 ? 1.0 : 0.0)) / B;
            const uint16_t* bp; int n; bitsOf(c, bp, n);
            for (int k = 0; k < n; ++k) g[bp[k]] += float(coef);
          }
        });
        for (int b = 0; b < B; ++b) if (has[b]) { ctrLoss += lb[b]; ctrAcc += ab[b]; ++ctrN; }
        upload(gExtra, Gh);
        m.extraGrad = gExtra.data();
      }
      m.backward(B, pw, ls);
      float t = float(step) / float(std::max(1, totalSteps));
      float lrNow = step < warm ? lr * float(step + 1) / float(warm) : lr * 0.5f * (1.f + std::cos(3.14159265f * t));
      for (Param* p : m.params()) p->adam(lrNow, step + 1, wd);
      if (st % 1000 == 0) {
        download(m.P, P, int64_t(B) * m.NB);
        double l = 0.0;
        for (size_t i = 0; i < P.size(); ++i) {
          float pp = std::min(1.f - 1e-7f, std::max(1e-7f, P[i]));
          l -= cur.T[i] > 0.f ? std::log(pp) : std::log(1.f - pp);
        }
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("ep %d step %d/%d  loss/spectrum %.3f  lr %.6f  %.0fs", ep, st, stepsPerEpoch, l / B, lrNow, sec);
        if (ctrN) std::printf("  ctr CE %.3f top1 %.3f (n %zu)", ctrLoss / ctrN, ctrAcc / ctrN, ctrN);
        std::printf("\n");
        ctrLoss = ctrAcc = 0; ctrN = 0;
        std::fflush(stdout);
      }
      if (th.joinable()) th.join();
      std::swap(cur, nxt);
    }
    if ((ep + 1) % evalEvery == 0 || ep + 1 == epochs) {
      double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::printf("== epoch %d done (%.0fs)\n", ep + 1, sec);
      evaluateSets(ems, sets, ep + 1 == epochs);
    }
  }
  if (hold > 0) calibrate(m, lib_s, fps, hold, 20000);
  m.save(dir + "/" + outPath);
  if (hold > 0) evaluateSets(ems, sets, true);
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("wrote %s (%.0fs total)\n", outPath.c_str(), sec);
  return 0;
}
