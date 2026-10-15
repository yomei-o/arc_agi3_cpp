// lmart: LambdaMART ranker (histogram gradient-boosted regression trees with
// lambda gradients) over phys_lm feature dumps. No external ML library.
//
//   lmart train key=value...
//     train=f1.txt:G,f2.txt:G,...   dump files with a group letter each
//     mix=G:share,...               total weight share per group (each query of
//                                   group g weighs share_g * N / n_g, n_g counting
//                                   all its queries; unlisted groups: share 0)
//     val=f.txt,...                 early stopping on the mean MRR@25 of these
//     test=f.txt:label,...          reported only
//     out=model.txt                 (lmart_model.h format)
//     trees=2000 lr=0.05 leaves=15 depth=6 minleaf=40 minhess=1e-3 l2=1
//     col=0.8 row=0.8 (per-tree feature / query subsampling) es=200 (patience)
//     metric=ndcg|mrr sigma=1 norm=1 bins=255 seed=1 thr=6
//     drop=name,name,...            features left out
//     keep=name,...                 only these features
//     base=hand:1                   initial score = coef * feature (default none)
//     permute=1                     control: training truth replaced by a random
//                                   candidate of the same query
//   lmart eval model=m.txt test=f.txt:label,...
//
// Dump format (phys_lm eval [features.txt]): "Q <name> <truth idx|-1> <n>",
// then n lines "phys mlp rev lib ana inLib name g0..g5 merr aCos aMcos aM6p2
// aM6p4 aNP6 nspec mass pol adduct instr". inLib and name are never features.
//
// MRR@25 is over all queries (no truth among the candidates = 0), reported
// with ties averaged and pessimistic (ties count as above the truth).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "rdkit-min/lm_feat.h"
#include "rdkit-min/lmart_model.h"

namespace {

struct Set {
  std::string path, label;
  char group = 0;
  int nq = 0;
  std::vector<int> qStart;  // row offsets, size nq+1
  std::vector<int> truth;   // per query, local index or -1
  std::vector<std::string> qname;  // per query (molecule key)
  std::vector<int> qset;    // merged training set: index of the source set
  std::vector<float> X;     // rows * F
  std::vector<double> qw;   // per query weight
};

int F = 0;

Set load(const std::string& path) {
  Set s;
  s.path = path;
  std::ifstream f(path);
  if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(1); }
  std::string line;
  s.qStart.push_back(0);
  std::vector<lmf::Row> R;
  lmf::QMeta meta;
  std::vector<float> Xq;
  int rows = 0;
  auto flush = [&]() {
    lmf::build(R, meta, Xq);
    s.X.insert(s.X.end(), Xq.begin(), Xq.end());
    rows += int(R.size());
    s.qStart.push_back(rows);
  };
  bool open = false;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    if (line[0] == 'Q' && line[1] == ' ') {
      if (open) flush();
      std::istringstream is(line.substr(2));
      std::string name;
      int t, n;
      is >> name >> t >> n;
      s.truth.push_back(t);
      s.qname.push_back(name);
      ++s.nq;
      R.clear();
      R.reserve(n);
      open = true;
      continue;
    }
    std::istringstream is(line);
    lmf::Row r{};
    float inLib;
    std::string name;
    is >> r[lmf::PHYS] >> r[lmf::MLP] >> r[lmf::REV] >> r[lmf::LIB] >> r[lmf::ANA] >> inLib >> name;
    for (int k = 0; k < 6; ++k) is >> r[lmf::G0 + k];
    is >> r[lmf::MERR] >> r[lmf::ACOS] >> r[lmf::AMCOS] >> r[lmf::AM6P2] >> r[lmf::AM6P4] >> r[lmf::ANP6];
    if (!is) { std::fprintf(stderr, "%s: short row (phys_lm dump needed)\n", path.c_str()); std::exit(1); }
    is >> meta.nspec >> meta.mass >> meta.pol >> meta.adduct >> meta.instr;
    R.push_back(r);
  }
  if (open) flush();
  s.qw.assign(s.nq, 1.0);
  int withT = 0;
  for (int t : s.truth) withT += t >= 0;
  std::printf("%s: %d queries (%d with truth), %d rows\n", path.c_str(), s.nq, withT, s.qStart.back());
  return s;
}

// MRR@25 of per-row scores over a set: (ties averaged, pessimistic)
std::pair<double, double> mrr(const Set& s, const std::vector<float>& sc) {
  double a = 0, p = 0;
  for (int q = 0; q < s.nq; ++q) {
    int t = s.truth[q];
    if (t < 0) continue;
    int b = s.qStart[q], e = s.qStart[q + 1];
    float v = sc[b + t];
    int better = 0, eq = 0;
    for (int i = b; i < e; ++i)
      if (i != b + t) { better += sc[i] > v; eq += sc[i] == v; }
    if (better + eq < 25) p += 1.0 / (better + eq + 1);
    if (better < 25) a += 1.0 / (better + 1 + 0.5 * eq);
  }
  return {a / std::max(1, s.nq), p / std::max(1, s.nq)};
}

std::vector<float> column(const Set& s, int f) {
  std::vector<float> v(s.qStart.back());
  for (size_t i = 0; i < v.size(); ++i) v[i] = s.X[i * F + f];
  return v;
}

std::map<std::string, std::string> parseArgs(int argc, char** argv, int from) {
  std::map<std::string, std::string> A;
  for (int i = from; i < argc; ++i) {
    std::string a = argv[i];
    size_t e = a.find('=');
    if (e != std::string::npos) A[a.substr(0, e)] = a.substr(e + 1);
  }
  return A;
}
std::string get(const std::map<std::string, std::string>& A, const char* k, const char* d) {
  auto i = A.find(k);
  return i == A.end() ? d : i->second;
}
std::vector<std::string> split(const std::string& s, char c) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, c)) if (!t.empty()) out.push_back(t);
  return out;
}

struct TNode { int feat = -1, bin = 0; float thr = 0.f; int left = -1, right = -1; double value = 0; };

struct Hist {
  std::vector<double> g, h;
  std::vector<int> c;
  void init(int nb) { g.assign(size_t(F) * nb, 0.0); h.assign(size_t(F) * nb, 0.0); c.assign(size_t(F) * nb, 0); }
};

}  // namespace

int main(int argc, char** argv) {
  F = lmf::nf();
  const auto& FN = lmf::names();
  if (argc < 2) { std::fprintf(stderr, "usage: lmart train|eval key=value ...\n"); return 1; }
  std::string mode = argv[1];
  auto A = parseArgs(argc, argv, 2);
  auto featIdx = [&](const std::string& n) {
    for (int i = 0; i < F; ++i) if (FN[i] == n) return i;
    std::fprintf(stderr, "unknown feature %s\n", n.c_str());
    std::exit(1);
  };
  // test sets
  std::vector<Set> tests;
  for (auto& spec : split(get(A, "test", ""), ',')) {
    std::string path = spec, label = spec;
    size_t c = spec.rfind(':');
    if (c != std::string::npos && c > 1) { path = spec.substr(0, c); label = spec.substr(c + 1); }
    tests.push_back(load(path));
    tests.back().label = label;
  }
  if (mode == "eval") {
    lmart::Model m;
    if (!m.load(get(A, "model", "model.txt"), FN)) { std::fprintf(stderr, "cannot load model\n"); return 1; }
    for (auto& s : tests) {
      std::vector<float> sc(s.qStart.back());
      for (size_t i = 0; i < sc.size(); ++i) sc[i] = m.score(&s.X[i * F]);
      auto r = mrr(s, sc);
      auto h = mrr(s, column(s, featIdx("hand")));
      std::printf("eval %-8s model %.4f (pess %.4f)  hand %.4f (pess %.4f)\n", s.label.c_str(), r.first, r.second, h.first, h.second);
    }
    return 0;
  }

  // ---- training data ----
  std::vector<Set> train;
  for (auto& spec : split(get(A, "train", ""), ',')) {
    size_t c = spec.rfind(':');
    train.push_back(load(spec.substr(0, c)));
    train.back().group = spec[c + 1];
    train.back().label = spec.substr(0, c);
  }
  std::vector<Set> vals;
  for (auto& p : split(get(A, "val", ""), ',')) { vals.push_back(load(p)); vals.back().label = p; }
  const int nTrees = std::atoi(get(A, "trees", "2000").c_str());
  const double lr = std::atof(get(A, "lr", "0.05").c_str());
  const int maxLeaves = std::atoi(get(A, "leaves", "15").c_str());
  const int maxDepth = std::atoi(get(A, "depth", "6").c_str());
  const int minLeaf = std::atoi(get(A, "minleaf", "40").c_str());
  const double minHess = std::atof(get(A, "minhess", "1e-3").c_str());
  const double l2 = std::atof(get(A, "l2", "1").c_str());
  const double colFrac = std::atof(get(A, "col", "0.8").c_str());
  const double rowFrac = std::atof(get(A, "row", "0.8").c_str());
  const int patience = std::atoi(get(A, "es", "200").c_str());
  const std::string metric = get(A, "metric", "ndcg");
  const double sigma = std::atof(get(A, "sigma", "1").c_str());
  const bool norm = get(A, "norm", "1") == "1";
  const int maxBins = std::min(255, std::atoi(get(A, "bins", "255").c_str()));
  const unsigned nThr = unsigned(std::atoi(get(A, "thr", "6").c_str()));
  const bool permute = get(A, "permute", "0") == "1";
  std::mt19937_64 rng(std::atoll(get(A, "seed", "1").c_str()));

  std::vector<char> useF(F, 1);
  if (A.count("keep")) { useF.assign(F, 0); for (auto& n : split(A["keep"], ',')) useF[featIdx(n)] = 1; }
  for (auto& n : split(get(A, "drop", ""), ',')) useF[featIdx(n)] = 0;
  int baseF = -1;
  double baseCoef = 0;
  if (A.count("base")) {
    auto p = split(A["base"], ':');
    baseF = featIdx(p[0]);
    baseCoef = p.size() > 1 ? std::atof(p[1].c_str()) : 1.0;
  }

  // class-mix weights
  {
    std::map<char, double> share;
    for (auto& p : split(get(A, "mix", ""), ',')) share[p[0]] = std::atof(p.substr(2).c_str());
    std::map<char, int> ng;
    int N = 0;
    for (auto& s : train) { ng[s.group] += s.nq; N += s.nq; }
    double tot = 0;
    for (auto& kv : share) tot += kv.second;
    for (auto& s : train) {
      double w = share.empty() ? 1.0 : (share.count(s.group) ? share[s.group] / tot * N / ng[s.group] : 0.0);
      s.qw.assign(s.nq, w);
      std::printf("train %s group %c: query weight %.3f\n", s.path.c_str(), s.group, w);
    }
  }
  // merge training sets
  Set T;
  T.qStart.push_back(0);
  for (auto& s : train) {
    int off = T.qStart.back();
    for (int q = 0; q < s.nq; ++q) {
      T.truth.push_back(s.truth[q]);
      T.qname.push_back(s.qname[q]);
      T.qset.push_back(int(&s - &train[0]));
      T.qw.push_back(s.qw[q]);
      T.qStart.push_back(off + s.qStart[q + 1]);
    }
    T.nq += s.nq;
    T.X.insert(T.X.end(), s.X.begin(), s.X.end());
  }
  // valfrac=f: every training query whose index hashes below f is held out
  // (never trained on) and early stopping uses their class-mix-weighted MRR
  const double valFrac = std::atof(get(A, "valfrac", "0").c_str());
  std::vector<char> inVal(T.nq, 0), inTest(T.nq, 0);
  std::vector<int> trueTruth = T.truth;
  // fold=k/K: queries whose molecule name hashes to k mod K are held out of
  // training and validation (grouped by molecule across all training sets)
  // and reported as out-of-fold "FOLDRR" lines for cross-validation
  int foldK = 0, foldN = 0;
  if (A.count("fold")) { auto p = split(A["fold"], '/'); foldK = std::atoi(p[0].c_str()); foldN = std::atoi(p[1].c_str()); }
  auto fnv = [](const std::string& x) { uint64_t h = 1469598103934665603ull; for (unsigned char c : x) { h ^= c; h *= 1099511628211ull; } return h; };
  if (foldN > 0)
    for (int q = 0; q < T.nq; ++q) inTest[q] = int(fnv(T.qname[q]) % uint64_t(foldN)) == foldK;
  if (valFrac > 0)
    for (int q = 0; q < T.nq; ++q) {
      if (inTest[q]) continue;
      uint64_t h = uint64_t(q) * 0x9E3779B97F4A7C15ull;
      h ^= h >> 29;
      inVal[q] = double(h % 1000003) / 1000003.0 < valFrac;
    }
  if (permute) {
    for (int q = 0; q < T.nq; ++q)
      if (T.truth[q] >= 0 && !inVal[q] && !inTest[q]) {
        int n = T.qStart[q + 1] - T.qStart[q];
        T.truth[q] = int(rng() % uint64_t(n));
      }
    std::printf("CONTROL: training truth permuted within each query\n");
  }
  const int NR = T.qStart.back();
  std::printf("training: %d queries, %d rows, %d features used\n", T.nq, NR, int(std::count(useF.begin(), useF.end(), 1)));

  // ---- bins ----
  std::vector<std::vector<float>> upper(F);
  std::vector<uint8_t> B(size_t(NR) * F);
  for (int f = 0; f < F; ++f) {
    std::vector<float> v = column(T, f);
    std::sort(v.begin(), v.end());
    std::vector<float> u = v;
    u.erase(std::unique(u.begin(), u.end()), u.end());
    auto& up = upper[f];
    if (int(u.size()) <= maxBins) up = u;
    else {
      for (int k = 1; k <= maxBins; ++k) {
        float x = v[std::min(v.size() - 1, size_t(double(k) * v.size() / maxBins) - 1)];
        if (up.empty() || x > up.back()) up.push_back(x);
      }
      if (up.back() < u.back()) up.push_back(u.back());
    }
    for (int i = 0; i < NR; ++i) {
      float x = T.X[size_t(i) * F + f];
      int b = int(std::lower_bound(up.begin(), up.end(), x) - up.begin());
      B[size_t(i) * F + f] = uint8_t(std::min(b, int(up.size()) - 1));
    }
  }
  const int NB = 256;

  // ---- boosting ----
  std::vector<double> score(NR, 0.0), grad(NR), hess(NR);
  if (baseF >= 0)
    for (int i = 0; i < NR; ++i) score[i] = baseCoef * T.X[size_t(i) * F + baseF];
  std::vector<std::vector<double>> vscore(vals.size()), tscore(tests.size());
  auto initScores = [&](const Set& s, std::vector<double>& v) {
    v.assign(s.qStart.back(), 0.0);
    if (baseF >= 0) for (size_t i = 0; i < v.size(); ++i) v[i] = baseCoef * s.X[i * F + baseF];
  };
  for (size_t k = 0; k < vals.size(); ++k) initScores(vals[k], vscore[k]);
  for (size_t k = 0; k < tests.size(); ++k) initScores(tests[k], tscore[k]);
  auto toF = [](const std::vector<double>& v) { return std::vector<float>(v.begin(), v.end()); };
  auto valMRR = [&]() {
    if (valFrac > 0) {  // weighted MRR@25 (ties averaged) over the held-out training queries
      double num = 0, den = 0;
      for (int q = 0; q < T.nq; ++q) {
        if (!inVal[q]) continue;
        den += T.qw[q];
        int t = trueTruth[q];
        if (t < 0) continue;
        int b = T.qStart[q], e = T.qStart[q + 1];
        int better = 0, eq = 0;
        for (int i = b; i < e; ++i)
          if (i != b + t) { better += score[i] > score[b + t]; eq += score[i] == score[b + t]; }
        if (better < 25) num += T.qw[q] / (better + 1 + 0.5 * eq);
      }
      return den > 0 ? num / den : 0.0;
    }
    double m = 0;
    for (size_t k = 0; k < vals.size(); ++k) m += mrr(vals[k], toF(vscore[k])).first;
    return vals.empty() ? 0.0 : m / vals.size();
  };
  std::vector<std::vector<TNode>> trees;
  double best = valMRR();
  int bestIt = 0;
  std::printf("iter 0 val %.4f\n", best);
  std::vector<double> gainF(F, 0.0);
  auto tStart = std::chrono::steady_clock::now();
  std::vector<Hist> thrHist(nThr);

  auto buildHist = [&](const std::vector<int>& rows, const std::vector<int>& feats, Hist& H) {
    H.init(NB);
    unsigned nt = rows.size() > 20000 ? nThr : 1;
    if (nt == 1) {
      for (int r : rows) {
        const uint8_t* b = &B[size_t(r) * F];
        for (int f : feats) { size_t k = size_t(f) * NB + b[f]; H.g[k] += grad[r]; H.h[k] += hess[r]; ++H.c[k]; }
      }
      return;
    }
    std::vector<std::thread> th;
    for (unsigned t = 0; t < nt; ++t)
      th.emplace_back([&, t]() {
        Hist& L = thrHist[t];
        L.init(NB);
        for (size_t j = t; j < rows.size(); j += nt) {
          int r = rows[j];
          const uint8_t* b = &B[size_t(r) * F];
          for (int f : feats) { size_t k = size_t(f) * NB + b[f]; L.g[k] += grad[r]; L.h[k] += hess[r]; ++L.c[k]; }
        }
      });
    for (auto& x : th) x.join();
    for (unsigned t = 0; t < nt; ++t)
      for (size_t k = 0; k < H.g.size(); ++k) { H.g[k] += thrHist[t].g[k]; H.h[k] += thrHist[t].h[k]; H.c[k] += thrHist[t].c[k]; }
  };

  for (int it = 1; it <= nTrees; ++it) {
    // lambda gradients
    std::fill(grad.begin(), grad.end(), 0.0);
    std::fill(hess.begin(), hess.end(), 0.0);
    std::vector<char> qUse(T.nq, 0);
    for (int q = 0; q < T.nq; ++q) qUse[q] = !inVal[q] && !inTest[q] && T.truth[q] >= 0 && T.qw[q] > 0 && std::uniform_real_distribution<>(0, 1)(rng) < rowFrac;
    {
      std::vector<std::thread> th;
      std::atomic<int> nx(0);
      for (unsigned t = 0; t < nThr; ++t)
        th.emplace_back([&]() {
          std::vector<int> o;
          std::vector<int> rk;
          for (;;) {
            int q = nx++;
            if (q >= T.nq) break;
            if (!qUse[q]) continue;
            int b = T.qStart[q], n = T.qStart[q + 1] - b, t0 = T.truth[q];
            o.resize(n); rk.resize(n);
            std::iota(o.begin(), o.end(), 0);
            std::sort(o.begin(), o.end(), [&](int x, int y) { return score[b + x] > score[b + y] || (score[b + x] == score[b + y] && x < y); });
            for (int k = 0; k < n; ++k) rk[o[k]] = k + 1;
            auto gain = [&](int r) { return metric == "mrr" ? 1.0 / r : 1.0 / std::log2(1.0 + r); };
            double gt = gain(rk[t0]), st = score[b + t0], sumL = 0;
            for (int j = 0; j < n; ++j) {
              if (j == t0) continue;
              double d = std::fabs(gt - gain(rk[j]));
              if (d <= 0) continue;
              double rho = 1.0 / (1.0 + std::exp(sigma * (st - score[b + j])));
              double l = sigma * d * rho, hh = sigma * sigma * d * rho * (1.0 - rho);
              grad[b + t0] -= l; grad[b + j] += l;
              hess[b + t0] += hh; hess[b + j] += hh;
              sumL += 2 * l;
            }
            double nf = (norm && sumL > 0) ? std::log2(1.0 + sumL) / sumL : 1.0;
            nf *= T.qw[q];
            for (int j = 0; j < n; ++j) { grad[b + j] *= nf; hess[b + j] *= nf; }
          }
        });
      for (auto& x : th) x.join();
    }
    std::vector<int> feats;
    for (int f = 0; f < F; ++f)
      if (useF[f] && std::uniform_real_distribution<>(0, 1)(rng) < colFrac) feats.push_back(f);
    if (feats.empty()) for (int f = 0; f < F; ++f) if (useF[f]) feats.push_back(f);
    // rows of the bagged queries
    std::vector<int> rows0;
    for (int q = 0; q < T.nq; ++q)
      if (qUse[q]) for (int i = T.qStart[q]; i < T.qStart[q + 1]; ++i) rows0.push_back(i);
    // leaf-wise growth
    struct Leaf { std::vector<int> rows; Hist H; int node, depth; double G, Hs; int bf = -1, bb = 0; double gain = 0; };
    std::vector<TNode> nodes(1);
    std::vector<Leaf> leaves;
    auto findSplit = [&](Leaf& L) {
      L.bf = -1; L.gain = 0;
      if (L.depth >= maxDepth || int(L.rows.size()) < 2 * minLeaf) return;
      double parent = L.G * L.G / (L.Hs + l2);
      for (int f : feats) {
        double gl = 0, hl = 0;
        int cl = 0;
        const int nb = int(upper[f].size());
        for (int b = 0; b < nb - 1; ++b) {
          size_t k = size_t(f) * NB + b;
          gl += L.H.g[k]; hl += L.H.h[k]; cl += L.H.c[k];
          int cr = int(L.rows.size()) - cl;
          if (cl < minLeaf) continue;
          if (cr < minLeaf) break;
          double hr = L.Hs - hl;
          if (hl < minHess || hr < minHess) continue;
          double gr = L.G - gl;
          double g = gl * gl / (hl + l2) + gr * gr / (hr + l2) - parent;
          if (g > L.gain) { L.gain = g; L.bf = f; L.bb = b; }
        }
      }
    };
    {
      Leaf root;
      root.rows = std::move(rows0);
      root.node = 0; root.depth = 0;
      buildHist(root.rows, feats, root.H);
      root.G = 0; root.Hs = 0;
      for (int r : root.rows) { root.G += grad[r]; root.Hs += hess[r]; }
      findSplit(root);
      leaves.push_back(std::move(root));
    }
    int nLeaves = 1;
    while (nLeaves < maxLeaves) {
      int bi = -1;
      for (int i = 0; i < int(leaves.size()); ++i)
        if (leaves[i].bf >= 0 && (bi < 0 || leaves[i].gain > leaves[bi].gain)) bi = i;
      if (bi < 0) break;
      Leaf P = std::move(leaves[bi]);
      leaves.erase(leaves.begin() + bi);
      gainF[P.bf] += P.gain;
      Leaf Lc, Rc;
      for (int r : P.rows) (B[size_t(r) * F + P.bf] <= P.bb ? Lc.rows : Rc.rows).push_back(r);
      const int li = int(nodes.size()), ri = li + 1;
      {
        TNode& nd = nodes[P.node];
        nd.feat = P.bf; nd.bin = P.bb; nd.thr = upper[P.bf][P.bb];
        nd.left = li; nd.right = ri;
      }
      nodes.emplace_back(); nodes.emplace_back();
      Lc.node = li; Rc.node = ri;
      Lc.depth = Rc.depth = P.depth + 1;
      Leaf& small = Lc.rows.size() <= Rc.rows.size() ? Lc : Rc;
      Leaf& large = Lc.rows.size() <= Rc.rows.size() ? Rc : Lc;
      buildHist(small.rows, feats, small.H);
      large.H = std::move(P.H);
      for (size_t k = 0; k < large.H.g.size(); ++k) { large.H.g[k] -= small.H.g[k]; large.H.h[k] -= small.H.h[k]; large.H.c[k] -= small.H.c[k]; }
      for (Leaf* c : {&Lc, &Rc}) {
        c->G = 0; c->Hs = 0;
        for (int r : c->rows) { c->G += grad[r]; c->Hs += hess[r]; }
        findSplit(*c);
      }
      leaves.push_back(std::move(Lc));
      leaves.push_back(std::move(Rc));
      ++nLeaves;
    }
    for (Leaf& L : leaves) nodes[L.node].value = -lr * L.G / (L.Hs + l2);
    // apply (raw thresholds, exactly as lmart_model.h)
    auto pred = [&](const float* x) {
      int k = 0;
      while (nodes[k].feat >= 0) k = x[nodes[k].feat] <= nodes[k].thr ? nodes[k].left : nodes[k].right;
      return nodes[k].value;
    };
    for (int i = 0; i < NR; ++i) score[i] += pred(&T.X[size_t(i) * F]);
    for (size_t k = 0; k < vals.size(); ++k)
      for (size_t i = 0; i < vscore[k].size(); ++i) vscore[k][i] += pred(&vals[k].X[i * F]);
    for (size_t k = 0; k < tests.size(); ++k)
      for (size_t i = 0; i < tscore[k].size(); ++i) tscore[k][i] += pred(&tests[k].X[i * F]);
    trees.push_back(nodes);
    double v = valMRR();
    if (v > best + 1e-6 || (vals.empty() && valFrac <= 0)) { best = v; bestIt = it; }
    if (it % 50 == 0 || it == 1) {
      std::printf("iter %d val %.4f (best %.4f @%d) train %.4f", it, v, best, bestIt, mrr(T, toF(score)).first);
      for (size_t k = 0; k < tests.size(); ++k) std::printf(" %s %.4f", tests[k].label.c_str(), mrr(tests[k], toF(tscore[k])).first);
      std::printf("  %.1fs\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - tStart).count());
      std::fflush(stdout);
    }
    if ((!vals.empty() || valFrac > 0) && it - bestIt >= patience) break;
  }
  trees.resize(bestIt);
  std::printf("best iteration %d, val %.4f, %.1fs\n", bestIt, best,
              std::chrono::duration<double>(std::chrono::steady_clock::now() - tStart).count());
  // save
  std::string out = get(A, "out", "lmart_model.txt");
  {
    FILE* fo = std::fopen(out.c_str(), "w");
    std::fprintf(fo, "LMART1\nfeatures %d", F);
    for (auto& n : FN) std::fprintf(fo, " %s", n.c_str());
    std::fprintf(fo, "\nbase %s %.9g\ntrees %zu\n", baseF >= 0 ? FN[baseF].c_str() : "-", baseCoef, trees.size());
    for (auto& t : trees) {
      std::fprintf(fo, "tree %zu\n", t.size());
      for (auto& nd : t) std::fprintf(fo, "%d %.9g %d %d %.9g\n", nd.feat, nd.thr, nd.left, nd.right, nd.value);
    }
    std::fclose(fo);
    std::printf("wrote %s\n", out.c_str());
  }
  // final report with the saved model (reloaded: checks the file path end to end)
  lmart::Model m;
  if (!m.load(out, FN)) { std::fprintf(stderr, "reload failed\n"); return 1; }
  auto report = [&](const Set& s, const char* kind) {
    std::vector<float> sc(s.qStart.back());
    for (size_t i = 0; i < sc.size(); ++i) sc[i] = m.score(&s.X[i * F]);
    auto r = mrr(s, sc);
    auto h = mrr(s, column(s, featIdx("hand")));
    std::printf("RESULT %-5s %-24s model %.4f (pess %.4f)  hand %.4f (pess %.4f)  diff %+.4f\n", kind, s.label.c_str(), r.first,
                r.second, h.first, h.second, r.first - h.first);
  };
  if (foldN > 0) {
    // out-of-fold reciprocal ranks per source set (ties averaged), model and hand
    std::vector<double> rm(train.size(), 0), rh(train.size(), 0);
    std::vector<int> nq(train.size(), 0);
    const int hf = featIdx("hand");
    for (int q = 0; q < T.nq; ++q) {
      if (!inTest[q]) continue;
      int k = T.qset[q];
      ++nq[k];
      int t = trueTruth[q];
      if (t < 0) continue;
      int b = T.qStart[q], e = T.qStart[q + 1];
      for (int pass = 0; pass < 2; ++pass) {
        auto val = [&](int i) { return pass == 0 ? m.score(&T.X[size_t(i) * F]) : T.X[size_t(i) * F + hf]; };
        float v = val(b + t);
        int better = 0, eq = 0;
        for (int i = b; i < e; ++i)
          if (i != b + t) { float x = val(i); better += x > v; eq += x == v; }
        if (better < 25) (pass == 0 ? rm : rh)[k] += 1.0 / (better + 1 + 0.5 * eq);
      }
    }
    // oof=file: per held-out query "set name rankModel rankHand q_maxlib truthLib" (rank 0 = truth absent)
    if (A.count("oof")) {
      FILE* fo = std::fopen(A["oof"].c_str(), "a");
      const int ql = featIdx("q_maxlib"), lf = featIdx("lib");
      for (int q = 0; q < T.nq; ++q) {
        if (!inTest[q]) continue;
        int t = trueTruth[q], b = T.qStart[q], e = T.qStart[q + 1];
        int rk[2] = {0, 0};
        if (t >= 0)
          for (int pass = 0; pass < 2; ++pass) {
            auto val = [&](int i) { return pass == 0 ? m.score(&T.X[size_t(i) * F]) : T.X[size_t(i) * F + hf]; };
            float v = val(b + t);
            int better = 0;
            for (int i = b; i < e; ++i) if (i != b + t && val(i) >= v) ++better;
            rk[pass] = better + 1;
          }
        std::fprintf(fo, "%s\t%s\t%d\t%d\t%.4f\t%.4f\n", train[T.qset[q]].label.c_str(), T.qname[q].c_str(), rk[0], rk[1],
                     T.X[size_t(b) * F + ql], t >= 0 ? T.X[size_t(b + t) * F + lf] : -1.f);
      }
      std::fclose(fo);
    }
    for (size_t k = 0; k < train.size(); ++k)
      std::printf("FOLDRR %s %d %.6f %.6f\n", train[k].label.c_str(), nq[k], rm[k], rh[k]);
  }
  for (auto& s : train) report(s, "train");
  for (auto& s : vals) report(s, "val");
  for (auto& s : tests) report(s, "test");
  std::vector<int> o(F);
  std::iota(o.begin(), o.end(), 0);
  std::sort(o.begin(), o.end(), [&](int a, int b) { return gainF[a] > gainF[b]; });
  double gs = std::accumulate(gainF.begin(), gainF.end(), 0.0) + 1e-12;
  std::printf("gain:");
  for (int k = 0; k < F && k < 20; ++k) if (gainF[o[k]] > 0) std::printf(" %s %.3f", FN[o[k]].c_str(), gainF[o[k]] / gs);
  std::printf("\n");
  return 0;
}
