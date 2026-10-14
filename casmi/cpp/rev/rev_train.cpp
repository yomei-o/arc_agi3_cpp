// Train / evaluate / apply the reverse model (rev_model.hpp).
//
//   rev_train train <dir> <lib> <epochs> <batch> <lr> <out.bin> [dropout] [posWeight] [evalQry] [ppm]
//   rev_train eval  <dir> <qry> <model.bin> <ppm>
//   rev_train rank  <dir> <qry> <model.bin> <ppm> <topk> <out.csv>
//
// Train on dlib (train minus every spectrum of the 400 denovo-holdout
// molecules) and evaluate on dqry: the holdout the reverse model has never
// seen a single spectrum of, and the one whose construction matches the
// real test (the answer has no reference spectrum anywhere). The earlier
// forward-model MLP was trained on all of train - including qry's and
// dqry's own molecules - so its holdout numbers were measured on molecules
// it had trained on; this one is not.
//
// A candidate's score is the fingerprint log-likelihood under the predicted
// bit probabilities: sum_all log(1-p) + sum_on [log p - log(1-p)], i.e. a
// per-query constant plus the sum of logit(p_i) over the candidate's own
// on-bits. Candidates come from the pool (train + COCONUT + PubChem, 4.37M)
// in a ppm window of each spectrum's adduct-corrected neutral mass, exactly
// as pool_candidate_rank selects them.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "rev_model.hpp"

using namespace rev;

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
  std::vector<std::string> name, smiles;
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
    size_t t = line.find('\t');
    s.name.push_back(line.substr(0, t));
    s.smiles.push_back(t == std::string::npos ? "" : line.substr(t + 1));
  }
  std::printf("%s: %zu spectra, %zu molecules\n", stem.c_str(), s.head.size(), s.name.size());
  return s;
}

// Fingerprints as a flat CSR: bits of molecule i are bits[ptr[i] .. ptr[i+1]).
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

struct Pool {
  std::vector<float> sortedMass;
  std::vector<int32_t> order;           // sorted position -> mol id
  std::vector<std::string> name, smiles;
};
Pool loadPool(const std::string& dir) {
  Pool p;
  std::vector<char> hb = slurp(dir + "/pool_head.bin");
  size_t n = hb.size() / 12;
  p.sortedMass.resize(n); p.order.resize(n);
  for (size_t i = 0; i < n; ++i) {
    std::memcpy(&p.sortedMass[i], hb.data() + 12 * i, 4);
    std::memcpy(&p.order[i], hb.data() + 12 * i + 4, 4);
  }
  std::ifstream f(dir + "/pool_mols.txt");
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    size_t t = line.find('\t');
    p.name.push_back(line.substr(0, t));
    p.smiles.push_back(t == std::string::npos ? "" : line.substr(t + 1));
  }
  std::printf("pool: %zu candidates\n", p.name.size());
  return p;
}

uint64_t fpHash(const Fps& fps, int id) {
  uint64_t h = 1469598103934665603ull;
  for (uint32_t k = fps.ptr[id]; k < fps.ptr[id + 1]; ++k) h = (h ^ fps.bits[k]) * 1099511628211ull;
  return h;
}

// Predicted mean probabilities per query molecule (averaged over its spectra).
std::unordered_map<int, std::vector<float>> predictByMol(Model& m, const Store& q);
struct Ensemble { std::vector<Model*> ms; int NB = 2048; };
std::unordered_map<int, std::vector<float>> predictByMol(Ensemble& e, const Store& q) {
  std::unordered_map<int, std::vector<float>> acc;
  for (Model* m : e.ms) {
    auto p = predictByMol(*m, q);
    for (auto& kv : p) {
      auto& v = acc[kv.first];
      if (v.empty()) v.assign(kv.second.size(), 0.f);
      for (size_t i = 0; i < v.size(); ++i) v[i] += kv.second[i] / float(e.ms.size());
    }
  }
  return acc;
}
std::unordered_map<int, std::vector<float>> predictByMol(Model& m, const Store& q) {
  std::unordered_map<int, std::vector<float>> sum;
  std::unordered_map<int, int> cnt;
  const int B = 256;
  std::vector<std::vector<std::pair<int, float>>> rows;
  std::vector<float> P;
  for (size_t s0 = 0; s0 < q.head.size(); s0 += B) {
    size_t s1 = std::min(q.head.size(), s0 + B);
    rows.assign(s1 - s0, {});
    for (size_t s = s0; s < s1; ++s) featurize(q.head[s], q.mzs(q.head[s]), q.its(q.head[s]), rows[s - s0]);
    m.setBatch(rows, false);
    m.forward(int(s1 - s0), false, 0.f, 0);
    download(m.P, P, int64_t(s1 - s0) * m.NB);
    for (size_t s = s0; s < s1; ++s) {
      int mol = q.head[s].mol;
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

// Rank pool candidates for each query molecule. Returns (mol, ranked pool ids).
template <class M>
std::vector<std::pair<int, std::vector<std::pair<float, int>>>> rankAll(
    M& m, const Store& q, const Pool& pool, const Fps& fps, float ppm) {
  auto probs = predictByMol(m, q);
  std::unordered_map<int, std::vector<int>> specOf;
  for (size_t s = 0; s < q.head.size(); ++s) specOf[q.head[s].mol].push_back(int(s));
  std::vector<int> mols;
  for (auto& kv : specOf) mols.push_back(kv.first);
  std::sort(mols.begin(), mols.end());
  std::vector<std::pair<int, std::vector<std::pair<float, int>>>> out(mols.size());
  std::vector<float> logit(m.NB);
  for (size_t mi = 0; mi < mols.size(); ++mi) {
    int mol = mols[mi];
    const auto& p = probs[mol];
    for (int i = 0; i < m.NB; ++i) {
      float pi = std::min(1.f - 1e-6f, std::max(1e-6f, p[i]));
      logit[i] = std::log(pi / (1.f - pi));
    }
    std::set<int> cand;
    for (int s : specOf[mol]) {
      const Head& h = q.head[s];
      int a = h.tag < 0 ? -1 : (h.tag % 16 == 15 ? h.tag / 16 + 1 : h.tag / 16);
      float neutral = h.precursor - ((a >= 0 && a < 10) ? ADDUCT_SHIFT[a] : 1.007276f);
      float w = neutral * ppm * 1e-6f;
      size_t lo = std::lower_bound(pool.sortedMass.begin(), pool.sortedMass.end(), neutral - w) - pool.sortedMass.begin();
      size_t hi = std::upper_bound(pool.sortedMass.begin(), pool.sortedMass.end(), neutral + w) - pool.sortedMass.begin();
      for (size_t k = lo; k < hi; ++k) cand.insert(pool.order[k]);
    }
    auto& ranked = out[mi].second;
    out[mi].first = mol;
    for (int c : cand) {
      uint32_t b0 = fps.ptr[c], b1 = fps.ptr[c + 1];
      if (b1 == b0) continue;
      float sc = 0.f;
      for (uint32_t k = b0; k < b1; ++k) sc += logit[fps.bits[k]];
      ranked.emplace_back(sc, c);
    }
    // Deterministic order (score, then pool id) so the CPU and GPU builds
    // agree, then drop candidates whose fingerprint duplicates a higher-
    // ranked one. ECFP here ignores stereo, and the metric is InChIKey14 -
    // the connectivity block, also stereo-blind - so PubChem's many
    // stereoisomers of one skeleton are one answer occupying several of
    // the 25 slots. Measured before this: GPU vs CPU top-1 agreed on only
    // 386/400 test molecules, almost entirely from exact score ties among
    // such duplicates being ordered arbitrarily.
    std::sort(ranked.begin(), ranked.end(), [](auto& x, auto& y) {
      return x.first != y.first ? x.first > y.first : x.second < y.second;
    });
    std::vector<std::pair<float, int>> dedup;
    std::set<uint64_t> seenFp;
    for (auto& e : ranked) {
      if (seenFp.insert(fpHash(fps, e.second)).second) dedup.push_back(e);
    }
    ranked.swap(dedup);
  }
  return out;
}

template <class M>
void evaluate(M& m, const Store& q, const Pool& pool, const Fps& fps, float ppm) {
  auto ranked = rankAll(m, q, pool, fps, ppm);
  // Truth = any candidate with the true structure's fingerprint (stereo-
  // blind, like the metric's InChIKey14); after dedup that is one entry.
  std::unordered_map<std::string, uint64_t> truthFp;
  for (size_t i = 0; i < pool.name.size(); ++i) truthFp.emplace(pool.name[i], fpHash(fps, int(i)));
  int n = 0, top1 = 0, top5 = 0, top25 = 0, present = 0;
  double mrr = 0.0, candSum = 0.0;
  for (auto& [mol, list] : ranked) {
    ++n;
    candSum += double(list.size());
    auto it = truthFp.find(q.name[mol]);
    int rank = -1;
    if (it != truthFp.end())
      for (size_t r = 0; r < list.size(); ++r)
        if (fpHash(fps, list[r].second) == it->second) { rank = int(r) + 1; break; }
    if (rank < 0) continue;
    ++present;
    if (rank <= 25) { mrr += 1.0 / rank; ++top25; }
    if (rank == 1) ++top1;
    if (rank <= 5) ++top5;
  }
  std::printf("  eval %d mols, truth in candidates %d, mean cands %.0f | top1 %.1f%% top5 %.1f%% top25 %.1f%% MRR@25 %.3f\n",
              n, present, candSum / std::max(1, n), 100.0 * top1 / n, 100.0 * top5 / n, 100.0 * top25 / n, mrr / n);
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: rev_train train|eval|rank ...\n"); return 1; }
  std::string mode = argv[1];
  std::printf("backend: %s\n", bk::backend_name());

  if (mode == "train") {
    std::string dir = argv[2], lib = argv[3];
    int epochs = std::atoi(argv[4]), B = std::atoi(argv[5]);
    float lr = float(std::atof(argv[6]));
    std::string outPath = argv[7];
    float dropout = argc > 8 ? float(std::atof(argv[8])) : 0.2f;
    float posWeight = argc > 9 ? float(std::atof(argv[9])) : 1.0f;
    std::string evalQry = argc > 10 ? argv[10] : "";
    float ppm = argc > 11 ? float(std::atof(argv[11])) : 5.0f;
    int hidden = argc > 12 ? std::atoi(argv[12]) : 1024;
    uint64_t seed = argc > 13 ? uint64_t(std::atoll(argv[13])) : 7;

    Store lib_s = loadSpectra(dir, lib);
    Fps fps = loadFps(dir + "/" + lib + "_ecfp.bin");
    Store q; Pool pool; Fps poolFps;
    if (!evalQry.empty()) {
      q = loadSpectra(dir, evalQry);
      pool = loadPool(dir);
      poolFps = loadFps(dir + "/pool_ecfp.bin");
    }

    std::vector<int> idx;
    for (size_t s = 0; s < lib_s.head.size(); ++s) {
      int mol = lib_s.head[s].mol;
      if (lib_s.head[s].n > 0 && fps.ptr[mol + 1] > fps.ptr[mol]) idx.push_back(int(s));
    }
    std::printf("training spectra: %zu\n", idx.size());

    Model m;
    m.H1 = m.H2 = hidden;
    m.init(seed);
    m.reserve(B);
    int stepsPerEpoch = int(idx.size() / size_t(B));
    int totalSteps = stepsPerEpoch * epochs, step = 0;
    uint64_t rng = 12345 + seed * 7919;
    std::vector<std::vector<std::pair<int, float>>> rows(B);
    std::vector<float> T(size_t(B) * m.NB), P;
    auto t0 = std::chrono::steady_clock::now();
    for (int ep = 0; ep < epochs; ++ep) {
      for (size_t i = idx.size(); i > 1; --i) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        std::swap(idx[i - 1], idx[rng % i]);
      }
      double lossAcc = 0.0; int lossN = 0;
      for (int st = 0; st < stepsPerEpoch; ++st, ++step) {
        std::fill(T.begin(), T.end(), 0.f);
        for (int b = 0; b < B; ++b) {
          const Head& h = lib_s.head[idx[size_t(st) * B + b]];
          featurize(h, lib_s.mzs(h), lib_s.its(h), rows[b]);
          for (uint32_t k = fps.ptr[h.mol]; k < fps.ptr[h.mol + 1]; ++k)
            T[size_t(b) * m.NB + fps.bits[k]] = 1.f;
        }
        m.setBatch(rows, true);
        upload(m.T, T);
        m.forward(B, true, dropout, uint64_t(step));
        m.backward(B, posWeight);
        float t = float(step) / float(std::max(1, totalSteps));
        float lrNow = step < 500 ? lr * float(step + 1) / 500.f : lr * 0.5f * (1.f + std::cos(3.14159265f * t));
        for (Param* p : m.params()) p->adam(lrNow, step + 1, 0.f);
        if (st % 500 == 0) {
          download(m.P, P, int64_t(B) * m.NB);
          double l = 0.0;
          for (size_t i = 0; i < P.size(); ++i) {
            float pp = std::min(1.f - 1e-7f, std::max(1e-7f, P[i]));
            l -= T[i] > 0.f ? std::log(pp) : std::log(1.f - pp);
          }
          lossAcc += l / double(B); ++lossN;
          double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
          std::printf("ep %d step %d/%d  loss/spectrum %.3f  lr %.5f  %.0fs\n", ep, st, stepsPerEpoch, l / B, lrNow, sec);
          std::fflush(stdout);
        }
      }
      m.save(dir + "/" + outPath);
      if (!evalQry.empty()) evaluate(m, q, pool, poolFps, ppm);
    }
    std::printf("wrote %s\n", outPath.c_str());
    return 0;
  }

  if (mode == "eval" || mode == "rank") {
    std::string dir = argv[2], qs = argv[3], modelPath = argv[4];
    float ppm = float(std::atof(argv[5]));
    Store q = loadSpectra(dir, qs);
    Pool pool = loadPool(dir);
    Fps poolFps = loadFps(dir + "/pool_ecfp.bin");
    // Comma-separated model list -> ensemble, bit probabilities averaged.
    Ensemble m;
    std::vector<std::unique_ptr<Model>> owned;
    for (size_t a = 0;;) {
      size_t c = modelPath.find(',', a);
      std::string one = modelPath.substr(a, c == std::string::npos ? std::string::npos : c - a);
      owned.emplace_back(new Model());
      if (!owned.back()->load(dir + "/" + one)) { std::fprintf(stderr, "cannot load %s\n", one.c_str()); return 1; }
      m.ms.push_back(owned.back().get());
      m.NB = owned.back()->NB;
      if (c == std::string::npos) break;
      a = c + 1;
    }
    std::printf("ensemble of %zu model(s)\n", m.ms.size());
    if (mode == "eval") { evaluate(m, q, pool, poolFps, ppm); return 0; }
    int topk = std::atoi(argv[6]);
    std::string outPath = argv[7];
    auto ranked = rankAll(m, q, pool, poolFps, ppm);
    std::ofstream o(dir + "/" + outPath);
    o << "molecule_id,smiles\n";
    for (auto& [mol, list] : ranked) {
      o << q.name[mol] << ',';
      int k = 0;
      for (auto& sc : list) {
        if (k >= topk) break;
        if (k) o << ';';
        o << pool.smiles[sc.second];
        ++k;
      }
      o << '\n';
    }
    std::printf("wrote %s\n", outPath.c_str());
    return 0;
  }
  std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
  return 1;
}
