// Learn the fusion weights of phys_rank's scores instead of hand-gridding them.
//
// Per candidate, phys_rank (eval ... [features.txt]) writes the physics
// log-likelihood, the forward MLP's cosine, the reverse model's ECFP
// log-likelihood, the best similarity to the candidate's own library
// spectra, the analog-propagation score and whether it is a train structure.
// Here those become a feature vector x (per-query z-scores for the three
// model scores, sharp powers of the library similarity, the analog score,
// the train-structure flag) and the weights maximise the conditional
// log-likelihood of the true candidate within its own query - a
// conditional logit, the same objective a listwise ranker optimises:
//   L(b) = sum_q w_q [ b.x_true - log sum_i exp(b.x_i) ] - lambda |b|^2
//
//   fuse_fit <train1.txt>[:weight],<train2.txt>[:weight],... [eval1.txt,eval2.txt,...]
// prints the hand-tuned baseline, the fitted weights, 5-fold cross-validated
// MRR@25 on the training sets and MRR@25 on each eval set.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// 8 spectral features + COCONUT metadata (coco_meta.txt: InChIKey14
// collections dois synonyms organisms np_likeness) as popularity-like priors
constexpr int NBASE = 14;  // fixed features; extra columns after the name are appended
int NF = NBASE;
std::vector<std::string> FNAME = {"z(phys)", "z(mlp)", "z(rev)", "lib^32", "lib^8", "lib", "analog", "inLib",
                         "inCOCONUT", "log1p(coll)", "log1p(dois)", "log1p(syn)", "log1p(org)", "npl/3"};
std::unordered_map<std::string, std::array<float, 5>> META;
std::vector<int> DISABLED;  // feature indices forced to 0 (ablation)
bool COCONUT_ONLY = false;  // [5th arg "coconly"]
std::string EXTRA_MODE;  // [6th arg] one char per extra column: z or r

struct Query {
  int truth = -1;
  std::vector<std::vector<float>> x;
  double weight = 1.0;
  int set = 0;
};

std::vector<float> zscore(const std::vector<float>& v) {
  double m = 0, s = 0;
  for (float a : v) m += a;
  m /= double(std::max<size_t>(1, v.size()));
  for (float a : v) s += (a - m) * (a - m);
  s = std::sqrt(s / double(std::max<size_t>(1, v.size()))) + 1e-6;
  std::vector<float> z(v.size());
  for (size_t i = 0; i < v.size(); ++i) z[i] = float((v[i] - m) / s);
  return z;
}

std::vector<Query> load(const std::string& path, double weight, int set) {
  std::vector<Query> out;
  std::ifstream f(path);
  if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(1); }
  std::string tag, name;
  int truth, n;
  while (f >> tag >> name >> truth >> n) {
    std::vector<float> ph(n), ml(n), rv(n), lb(n), an(n), tr(n);
    std::vector<std::string> nm(n);
    // extra columns after the name: per EXTRA_MODE entry 'z' (z-score within
    // the query) or 'r' (raw)
    std::vector<std::vector<float>> ex(EXTRA_MODE.size(), std::vector<float>(n, 0.f));
    for (int i = 0; i < n; ++i) {
      f >> ph[i] >> ml[i] >> rv[i] >> lb[i] >> an[i] >> tr[i] >> nm[i];
      std::string rest;
      std::getline(f, rest);
      std::istringstream rs(rest);
      for (size_t e = 0; e < EXTRA_MODE.size(); ++e) rs >> ex[e][i];
    }
    for (size_t e = 0; e < EXTRA_MODE.size(); ++e)
      if (EXTRA_MODE[e] == 'z') ex[e] = zscore(ex[e]);
    Query q;
    q.truth = truth;
    q.weight = weight;
    q.set = set;
    auto zp = zscore(ph), zm = zscore(ml), zr = zscore(rv);
    q.x.resize(n);
    for (int i = 0; i < n; ++i) {
      float l = std::max(0.f, lb[i]);
      auto mt = META.find(nm[i]);
      std::array<float, 5> m = mt == META.end() ? std::array<float, 5>{0, 0, 0, 0, 0} : mt->second;
      q.x[i] = {zp[i], zm[i], zr[i], std::pow(l, 32.f), std::pow(l, 8.f), l, an[i], tr[i],
                float(mt != META.end()), std::log1p(m[0]), std::log1p(m[1]), std::log1p(m[2]),
                std::log1p(m[3]), m[4] / 3.f};
      for (size_t e = 0; e < EXTRA_MODE.size(); ++e) q.x[i].push_back(ex[e][i]);
      for (int d : DISABLED) q.x[i][d] = 0.f;
    }
    // COCONUT_ONLY: keep only candidates listed in COCONUT. The natural-
    // product holdouts' answers are all in COCONUT by construction, so over
    // the full pool "has COCONUT metadata" marks the answer; within COCONUT
    // the metadata counts can only help as genuine popularity.
    if (COCONUT_ONLY) {
      std::vector<std::vector<float>> keep;
      int t = -1;
      for (int i = 0; i < n; ++i)
        if (META.count(nm[i])) {
          if (i == q.truth) t = int(keep.size());
          keep.push_back(q.x[i]);
        }
      q.x = std::move(keep);
      q.truth = t;
    }
    out.push_back(std::move(q));
  }
  return out;
}

double score(const std::vector<float>& x, const std::vector<double>& b) {
  double s = 0;
  for (int k = 0; k < NF; ++k) s += b[k] * x[k];
  return s;
}

// MRR@25 with ties counted against the truth; queries without the truth in
// their candidates count as 0.
double mrr(const std::vector<Query>& Q, const std::vector<double>& b, const std::vector<int>* idx = nullptr) {
  double tot = 0;
  size_t cnt = 0;
  auto one = [&](const Query& q) {
    ++cnt;
    if (q.truth < 0) return;
    double st = score(q.x[q.truth], b);
    int better = 0;
    for (size_t i = 0; i < q.x.size(); ++i)
      if (int(i) != q.truth && score(q.x[i], b) >= st) ++better;
    if (better < 25) tot += 1.0 / (better + 1);
  };
  if (idx) for (int i : *idx) one(Q[i]);
  else for (auto& q : Q) one(q);
  return cnt ? tot / cnt : 0;
}

std::vector<double> fit(const std::vector<Query>& Q, const std::vector<int>& idx, double lambda) {
  std::vector<double> b(NF, 0.0), m(NF, 0.0), v(NF, 0.0);
  b[0] = 1.0;  // start at "physics only"
  const double lr = 0.05, b1 = 0.9, b2 = 0.999;
  for (int step = 1; step <= 600; ++step) {
    std::vector<double> g(NF, 0.0);
    double wsum = 0;
    for (int qi : idx) {
      const Query& q = Q[qi];
      if (q.truth < 0) continue;
      double mx = -1e300;
      std::vector<double> s(q.x.size());
      for (size_t i = 0; i < s.size(); ++i) { s[i] = score(q.x[i], b); mx = std::max(mx, s[i]); }
      double z = 0;
      for (double& a : s) { a = std::exp(a - mx); z += a; }
      for (int k = 0; k < NF; ++k) {
        double e = 0;
        for (size_t i = 0; i < s.size(); ++i) e += s[i] / z * q.x[i][k];
        g[k] += q.weight * (q.x[q.truth][k] - e);
      }
      wsum += q.weight;
    }
    for (int k = 0; k < NF; ++k) {
      double gk = g[k] / std::max(1e-9, wsum) - 2 * lambda * b[k];
      m[k] = b1 * m[k] + (1 - b1) * gk;
      v[k] = b2 * v[k] + (1 - b2) * gk * gk;
      double mh = m[k] / (1 - std::pow(b1, step)), vh = v[k] / (1 - std::pow(b2, step));
      b[k] += lr * mh / (std::sqrt(vh) + 1e-8);
    }
  }
  return b;
}

std::vector<std::string> split(const std::string& s, char d) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, d)) if (!t.empty()) out.push_back(t);
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: fuse_fit <train.txt[:w],...> [eval.txt,...] [lambda] [disabled feature idx,...]\n");
    return 1;
  }
  if (argc > 4) for (auto& d : split(argv[4], ',')) DISABLED.push_back(std::atoi(d.c_str()));
  COCONUT_ONLY = argc > 5 && std::string(argv[5]) == "coconly";
  if (argc > 6) EXTRA_MODE = argv[6] == std::string("-") ? "" : argv[6];
  NF = NBASE + int(EXTRA_MODE.size());
  for (size_t e = 0; e < EXTRA_MODE.size(); ++e) FNAME.push_back("x" + std::to_string(e) + "(" + EXTRA_MODE[e] + ")");
  {
    std::ifstream mf("coco_meta.txt");
    std::string k;
    std::array<float, 5> v;
    while (mf >> k >> v[0] >> v[1] >> v[2] >> v[3] >> v[4]) META[k] = v;
    std::printf("COCONUT metadata: %zu structures\n", META.size());
  }
  std::vector<Query> T;
  std::vector<std::string> trainNames;
  for (auto& item : split(argv[1], ',')) {
    auto p = item.find(':');
    std::string path = item.substr(0, p);
    double w = p == std::string::npos ? 1.0 : std::atof(item.substr(p + 1).c_str());
    auto q = load(path, w, int(trainNames.size()));
    std::printf("train %s: %zu queries (weight %g)\n", path.c_str(), q.size(), w);
    trainNames.push_back(path);
    T.insert(T.end(), q.begin(), q.end());
  }
  std::vector<std::vector<Query>> E;
  std::vector<std::string> evalNames;
  if (argc > 2)
    for (auto& path : split(argv[2], ',')) { E.push_back(load(path, 1.0, 0)); evalNames.push_back(path); }
  double lambda = argc > 3 ? std::atof(argv[3]) : 0.001;

  // hand-tuned weights of v17: phys + 0.25 mlp + 1.0 rev + 8 lib^32 + 8 analog
  std::vector<double> hand = {1, 0.25, 1, 8, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0};  // v20
  hand.resize(NF, 0.0);
  auto report = [&](const char* label, const std::vector<double>& b) {
    std::printf("%-8s", label);
    for (size_t s = 0; s < trainNames.size(); ++s) {
      std::vector<int> idx;
      for (size_t i = 0; i < T.size(); ++i) if (T[i].set == int(s)) idx.push_back(int(i));
      std::printf("  %s %.4f", trainNames[s].c_str(), mrr(T, b, &idx));
    }
    for (size_t e = 0; e < E.size(); ++e) std::printf("  %s %.4f", evalNames[e].c_str(), mrr(E[e], b));
    std::printf("\n");
  };
  report("hand", hand);

  std::vector<int> all(T.size());
  for (size_t i = 0; i < T.size(); ++i) all[i] = int(i);
  auto b = fit(T, all, lambda);
  report("fitted", b);
  std::printf("weights:");
  for (int k = 0; k < NF; ++k) std::printf("  %s %.3f", FNAME[k].c_str(), b[k]);
  std::printf("\n");

  // 5-fold CV over training queries (fold by index, interleaved)
  std::vector<double> rrSum(trainNames.size(), 0.0);
  std::vector<size_t> rrCnt(trainNames.size(), 0);
  for (int fold = 0; fold < 5; ++fold) {
    std::vector<int> tr, te;
    for (size_t i = 0; i < T.size(); ++i) (int(i % 5) == fold ? te : tr).push_back(int(i));
    auto bf = fit(T, tr, lambda);
    for (size_t s = 0; s < trainNames.size(); ++s) {
      std::vector<int> idx;
      for (int i : te) if (T[i].set == int(s)) idx.push_back(i);
      rrSum[s] += mrr(T, bf, &idx) * idx.size();
      rrCnt[s] += idx.size();
    }
  }
  std::printf("5-fold CV:");
  for (size_t s = 0; s < trainNames.size(); ++s) std::printf("  %s %.4f", trainNames[s].c_str(), rrSum[s] / std::max<size_t>(1, rrCnt[s]));
  std::printf("\n");
  return 0;
}
