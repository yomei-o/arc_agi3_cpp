// cnn_win_eval: copy of rev3/win_eval.cpp that also loads CNN reverse models
// (cnn_model.hpp, file magic CNN1; detected per model file), so REV2 and CNN
// models can be scored and z-sum ensembled in one run. Everything else is
// unchanged from win_eval:
// win_eval: "window MRR" yardstick (fp3_STATUS.md). Ranks EVERY PubChem
// candidate of each query's +-10 ppm window (win_build cache) by a reverse
// model's fingerprint score, MRR@25 of the truth (stereo-blind key match).
//
//   win_eval dir=C:\prog\casmi qry=hcqry win=w_hcqry models=rev2_y.bin:v4k[,b.bin:tag] [ens=0] [shift=7]
//            [out=ranks.txt] [threads=8]
// Candidate bits: <dir>/<win>_<tag>.bin (fp3_prep apply on <win>_wmols.txt).
// Prints, per model (and for the z-sum ensemble when ens=1), each scoring
// variant alone, + 0.15 popularity, popularity alone, and a shuffled-label
// control (query q scored with the predictions of query q+shift).
// MRR_in = over queries whose truth is in the window; MRR_all = over all.
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
#include <unordered_map>
#include <vector>

#include "parallel.hpp"
#include "rev2_model.hpp"
#include "cnn_model.hpp"

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
struct Fps { std::vector<uint64_t> ptr; std::vector<uint16_t> bits; uint32_t nBits = 0; };
Fps loadFps(const std::string& path) {
  Fps f;
  std::vector<char> b = slurp(path);
  const char* p = b.data();
  uint32_t n; std::memcpy(&n, p, 4); std::memcpy(&f.nBits, p + 4, 4); p += 8;
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

struct ModelInfo { int NB = 0; std::vector<float> calA, calC, quality; std::vector<int> H; std::string kind; };

std::vector<std::vector<float>> predictByMolCnn(cnn::Model& m, const Store& q) {
  std::vector<std::vector<float>> sum(q.nMol);
  std::vector<int> cnt(q.nMol, 0);
  std::vector<int> spec;
  for (size_t s = 0; s < q.head.size(); ++s) if (q.head[s].n > 0 && q.head[s].mol >= 0) spec.push_back(int(s));
  const int B = 256;
  std::vector<cnn::Sample> ss;
  std::vector<cnn::real> P;
  for (size_t s0 = 0; s0 < spec.size(); s0 += B) {
    size_t s1 = std::min(spec.size(), s0 + B);
    ss.assign(s1 - s0, cnn::Sample());
    ::parallel_for(int64_t(s1 - s0), [&](int64_t i) {
      const Head& h = q.head[spec[s0 + i]];
      cnn::featurize(m.cfg, h, q.mz.data() + h.off, q.inten.data() + h.off, ss[i]);
    });
    m.setBatch(ss, false);
    m.forward(int(s1 - s0), false, 0.f, 0);
    download(m.P, P, int64_t(s1 - s0) * m.NB);
    for (size_t s = s0; s < s1; ++s) {
      int mol = q.head[spec[s]].mol;
      auto& v = sum[mol];
      if (v.empty()) v.assign(m.NB, 0.f);
      for (int i = 0; i < m.NB; ++i) v[i] += float(P[(s - s0) * m.NB + i]);
      ++cnt[mol];
    }
  }
  for (size_t i = 0; i < sum.size(); ++i)
    for (auto& x : sum[i]) x /= float(cnt[i]);
  return sum;
}

const char* VN[] = {"logit", "logit_c01", "cal", "cal_wq", "tanimoto"};
constexpr int NV = 5;

// per-bit weights of variant v (linear variants), from mean probabilities p
std::vector<float> weights(const ModelInfo& m, const std::vector<float>& p, int v) {
  std::vector<float> u(m.NB, 0.f);
  bool cal = int(m.calA.size()) == m.NB;
  for (int b = 0; b < m.NB; ++b) {
    float pi = std::min(1.f - 1e-6f, std::max(1e-6f, p[b]));
    float lg = std::log(pi / (1.f - pi));
    if (v == 0) u[b] = lg;
    else if (v == 1) { float pc = std::min(0.99f, std::max(0.01f, p[b])); u[b] = std::log(pc / (1.f - pc)); }
    else if (v == 2 || v == 3) {
      float z = cal ? m.calA[b] * lg + m.calC[b] : lg;
      z = std::min(13.8f, std::max(-13.8f, z));
      u[b] = v == 2 ? z : z * (cal ? m.quality[b] : 1.f);
    }
  }
  return u;
}

void zs(std::vector<float>& v) {
  double m = 0, s = 0;
  for (float x : v) m += x;
  m /= double(std::max<size_t>(1, v.size()));
  for (float x : v) s += (x - m) * (x - m);
  s = std::sqrt(s / double(std::max<size_t>(1, v.size()))) + 1e-9;
  for (auto& x : v) x = float((x - m) / s);
}
// pessimistic rank (1-based) of t
int rankOf(const std::vector<float>& f, int t) {
  int better = 0;
  for (size_t i = 0; i < f.size(); ++i) if (int(i) != t && f[i] >= f[t]) ++better;
  return better + 1;
}

std::map<std::string, std::string> A;
std::string get(const std::string& k, const std::string& d) { auto it = A.find(k); return it == A.end() ? d : it->second; }
}  // namespace

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    size_t e = s.find('=');
    if (e != std::string::npos) A[s.substr(0, e)] = s.substr(e + 1);
  }
  std::string dir = get("dir", "C:\\prog\\casmi"), qry = get("qry", "hcqry"), win = get("win", "w_" + qry);
  int shift = std::atoi(get("shift", "7").c_str());
  bool ens = get("ens", "0") == "1";
  float popW = float(std::atof(get("popw", "0.15").c_str()));
  parallel_cap() = unsigned(std::atoi(get("threads", "8").c_str()));
  Store q = loadSpectra(dir, qry);
  auto wins = loadWin(dir + "/" + win + "_win.bin");
  std::vector<float> pop;
  {
    auto b = slurp(dir + "/" + win + "_wpop.bin");
    pop.resize(b.size() / 4);
    std::memcpy(pop.data(), b.data(), b.size());
  }
  const int nQ = int(wins.size());
  int nIn = 0;
  double wsz = 0;
  for (auto& w : wins) { nIn += w.truth >= 0; wsz += double(w.ids.size()); }
  std::printf("%s: %d queries, truth in window %d (%.3f), mean window %.0f candidates\n", qry.c_str(), nQ, nIn,
              double(nIn) / nQ, wsz / nQ);

  // score[model][variant][q][cand]
  struct MS { std::string name; std::vector<std::vector<std::vector<float>>> s, sh; };
  std::vector<MS> all;
  std::stringstream ss(get("models", ""));
  std::string item;
  while (std::getline(ss, item, ',')) {
    size_t c = item.find(':');
    std::string path = item.substr(0, c), tag = item.substr(c + 1);
    ModelInfo m;
    std::vector<std::vector<float>> P;
    auto t0 = std::chrono::steady_clock::now();
    const bool isMol = path.size() > 8 && path.compare(path.size() - 8, 8, "_mol.bin") == 0;
    if (isMol) {
      // "<x>_mol.bin" = per-molecule logits (fpnet_ggml / pt_train mollogits format: i32 nmol, i32 nbits,
      // per molecule i32 mol + f32[nbits]); p = sigmoid(logit), no calibration (as rev3/win_eval)
      auto b = slurp(dir + "/" + path);
      int32_t nm, nb; std::memcpy(&nm, b.data(), 4); std::memcpy(&nb, b.data() + 4, 4);
      m.NB = nb; m.kind = "mol logits";
      P.assign(q.nMol, {});
      const char* r = b.data() + 8;
      for (int i = 0; i < nm; ++i) {
        int32_t mol; std::memcpy(&mol, r, 4); r += 4;
        const float* z = reinterpret_cast<const float*>(r); r += 4 * size_t(nb);
        if (mol < 0 || size_t(mol) >= P.size()) continue;
        P[mol].resize(nb);
        for (int k = 0; k < nb; ++k) P[mol][k] = 1.f / (1.f + std::exp(-z[k]));
      }
    } else if (cnn::Model::isCnnFile(dir + "/" + path)) {
      cnn::Model cm;
      if (!cm.load(dir + "/" + path)) { std::fprintf(stderr, "cannot load %s\n", path.c_str()); return 1; }
      m.NB = cm.NB; m.calA = cm.calA; m.calC = cm.calC; m.quality = cm.quality; m.H = cm.cfg.chans; m.kind = "CNN chans";
      P = predictByMolCnn(cm, q);
    } else {
      Model rm;
      if (!rm.load(dir + "/" + path)) { std::fprintf(stderr, "cannot load %s\n", path.c_str()); return 1; }
      m.NB = rm.NB; m.calA = rm.calA; m.calC = rm.calC; m.quality = rm.quality; m.H = rm.H; m.kind = "MLP layers";
      P = predictByMol(rm, q);
    }
    double secPred = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    Fps F = loadFps(dir + "/" + win + "_" + tag + ".bin");
    if (int(F.nBits) != m.NB) { std::fprintf(stderr, "bits mismatch %s: %u vs %d\n", tag.c_str(), F.nBits, m.NB); return 1; }
    t0 = std::chrono::steady_clock::now();
    MS ms;
    ms.name = path;
    ms.s.assign(NV, std::vector<std::vector<float>>(nQ));
    ms.sh.assign(NV, std::vector<std::vector<float>>(nQ));
    ::parallel_for(nQ, [&](int64_t qi) {
      const Win& w = wins[qi];
      for (int pass = 0; pass < 2; ++pass) {
        int src = pass == 0 ? w.mol : wins[(qi + shift) % nQ].mol;
        const std::vector<float>& p = P[src];
        auto& dst = pass == 0 ? ms.s : ms.sh;
        if (p.empty()) { for (int v = 0; v < NV; ++v) dst[v][qi].assign(w.ids.size(), 0.f); continue; }
        double sumP = 0;
        for (float x : p) sumP += x;
        for (int v = 0; v < NV; ++v) {
          std::vector<float> u;
          if (v < 4) u = weights(m, p, v);
          auto& out = dst[v][qi];
          out.resize(w.ids.size());
          for (size_t k = 0; k < w.ids.size(); ++k) {
            uint32_t id = w.ids[k];
            const uint16_t* b = F.bits.data() + F.ptr[id];
            int n = int(F.ptr[id + 1] - F.ptr[id]);
            if (v < 4) { float s = 0; for (int j = 0; j < n; ++j) s += u[b[j]]; out[k] = s; }
            else { double in = 0; for (int j = 0; j < n; ++j) in += p[b[j]]; out[k] = float(in / (n + sumP - in + 1e-9)); }
          }
        }
      }
    });
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("model %s (%s): %d bits, %s", path.c_str(), tag.c_str(), m.NB, m.kind.c_str());
    for (int h : m.H) std::printf(" %d", h);
    std::printf(", calibrated %d, predicted in %.1fs, scored in %.1fs\n", int(m.calA.size()) == m.NB, secPred, sec);
    all.push_back(std::move(ms));
  }
  // ens=1: z-sum with equal weights; ensw=w1,w2,...;w1,w2,... adds one weighted z-sum per weight set
  std::vector<std::vector<float>> wsets;
  if (ens && all.size() > 1) wsets.push_back(std::vector<float>(all.size(), 1.f));
  {
    std::stringstream ws(get("ensw", ""));
    std::string set;
    while (std::getline(ws, set, ';')) {
      std::vector<float> w;
      std::stringstream s2(set);
      std::string t;
      while (std::getline(s2, t, ',')) w.push_back(float(std::atof(t.c_str())));
      if (w.size() == all.size()) wsets.push_back(w);
      else std::fprintf(stderr, "ensw set '%s' ignored (need %zu weights)\n", set.c_str(), all.size());
    }
  }
  const size_t nBase = all.size();
  // shufm=i: for every weight set also an ensemble whose member i alone uses the shuffled-query scores
  // (query q scored with query q+shift's predictions) = control for "what does member i add"
  const int shufm = std::atoi(get("shufm", "-1").c_str());
  for (size_t wi = 0, nw = wsets.size(); wi < nw; ++wi) {
    if (shufm < 0 || size_t(shufm) >= nBase || wsets[wi][shufm] == 0.f) continue;
    std::vector<float> w = wsets[wi];
    w.push_back(-1.f);  // marker: member shufm shuffled
    wsets.push_back(w);
  }
  for (auto& wm : wsets) {
    const bool shufOne = wm.size() == nBase + 1;
    std::vector<float> w(wm.begin(), wm.begin() + nBase);
    MS e;
    e.name = shufOne ? "ENSEMBLE-SHUFm" + std::to_string(shufm) + "(z-sum w=" : "ENSEMBLE(z-sum w=";
    for (size_t i = 0; i < w.size(); ++i) e.name += (i ? "," : "") + std::to_string(w[i]).substr(0, 4);
    e.name += ")";
    e.s.assign(NV, std::vector<std::vector<float>>(nQ));
    e.sh.assign(NV, std::vector<std::vector<float>>(nQ));
    for (int v = 0; v < NV; ++v)
      for (int qi = 0; qi < nQ; ++qi)
        for (int pass = 0; pass < 2; ++pass) {
          auto& d = pass ? e.sh[v][qi] : e.s[v][qi];
          d.assign(wins[qi].ids.size(), 0.f);
          for (size_t mi = 0; mi < nBase; ++mi) {
            const bool useSh = pass || (shufOne && int(mi) == shufm);
            auto x = useSh ? all[mi].sh[v][qi] : all[mi].s[v][qi];
            zs(x);
            for (size_t k = 0; k < d.size(); ++k) d[k] += w[mi] * x[k];
          }
        }
    all.push_back(std::move(e));
  }
  struct R { double in = 0, all = 0, top1 = 0, top25 = 0, top100 = 0, medrank = 0; };
  auto evalScores = [&](const std::function<void(int, std::vector<float>&)>& sc) {
    R r;
    std::vector<int> ranks;
    for (int qi = 0; qi < nQ; ++qi) {
      if (wins[qi].truth < 0) continue;
      std::vector<float> f;
      sc(qi, f);
      int rk = rankOf(f, wins[qi].truth);
      ranks.push_back(rk);
      if (rk <= 25) r.in += 1.0 / rk;
      r.top1 += rk == 1; r.top25 += rk <= 25; r.top100 += rk <= 100;
    }
    int n = std::max<int>(1, int(ranks.size()));
    r.all = r.in / nQ; r.in /= n; r.top1 /= n; r.top25 /= n; r.top100 /= n;
    std::sort(ranks.begin(), ranks.end());
    r.medrank = ranks.empty() ? 0 : ranks[ranks.size() / 2];
    return r;
  };
  auto pr = [&](const char* lab, const R& r) {
    std::printf("  %-34s MRR_in %.4f  MRR_all %.4f  top1 %.3f top25 %.3f top100 %.3f  median rank %.0f\n", lab, r.in,
                r.all, r.top1, r.top25, r.top100, r.medrank);
  };
  std::printf("[%s]\n", qry.c_str());
  pr("popularity only", evalScores([&](int qi, std::vector<float>& f) {
       f.resize(wins[qi].ids.size());
       for (size_t k = 0; k < f.size(); ++k) f[k] = pop[wins[qi].ids[k]];
     }));
  std::ofstream rout;
  if (A.count("out")) rout.open(dir + "/" + A["out"]);
  for (auto& ms : all) {
    std::printf(" model %s\n", ms.name.c_str());
    for (int v = 0; v < NV; ++v) {
      char lab[128];
      std::snprintf(lab, sizeof lab, "%s", VN[v]);
      pr(lab, evalScores([&](int qi, std::vector<float>& f) { f = ms.s[v][qi]; }));
      std::snprintf(lab, sizeof lab, "z(%s)+%.2f pop", VN[v], popW);
      pr(lab, evalScores([&](int qi, std::vector<float>& f) {
           f = ms.s[v][qi]; zs(f);
           for (size_t k = 0; k < f.size(); ++k) f[k] += popW * pop[wins[qi].ids[k]];
         }));
      std::snprintf(lab, sizeof lab, "SHUFFLED %s", VN[v]);
      pr(lab, evalScores([&](int qi, std::vector<float>& f) { f = ms.sh[v][qi]; }));
    }
    if (rout) {
      for (int qi = 0; qi < nQ; ++qi) {
        if (wins[qi].truth < 0) { rout << ms.name << '\t' << qi << "\t-1\n"; continue; }
        rout << ms.name << '\t' << qi << '\t' << rankOf(ms.s[2][qi], wins[qi].truth) << '\n';
      }
    }
  }
  std::fflush(stdout);
  return 0;
}
