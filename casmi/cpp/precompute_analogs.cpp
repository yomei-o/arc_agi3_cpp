// For every pool candidate with no reference spectrum of its own, find the K
// structurally most similar candidates that DO have one.
//
// The obvious design - restrict to donors within +-200 Da of the query mass -
// turned out not to filter anything: measured, this library's 275,810
// molecules sit so densely across the whole 50-650 Da range that a +-200 Da
// window around a typical test precursor already holds ~250k of them (90%).
// Mass-shift is not a useful gate here.
//
// What is useful: structural similarity by itself, independent of mass. A
// candidate's chemical neighbours don't change per query, so this is
// precomputed once - 453,579 candidates x 275,810 donors is ~1.25e11 Tanimoto
// comparisons, which a popcount is fast enough to get through in minutes, not
// hours, with Tanimoto done on already-packed 1024-bit Morgan fingerprints.
//
// Build:
//   g++ -O3 -std=c++17 -pthread -mpopcnt -o precompute_analogs precompute_analogs.cpp
//
// Usage:
//   precompute_analogs <dir> <topk>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int FP_WORDS = 16;   // 1024 bits / 64

struct Pool {
  int n = 0;
  std::vector<float> mass;
  std::vector<int32_t> mol_id;
  std::vector<int32_t> has_spectrum;
};

Pool load_pool(const std::string& dir) {
  Pool p;
  std::ifstream f(dir + "/pool_head.bin", std::ios::binary | std::ios::ate);
  if (!f) { std::fprintf(stderr, "cannot read pool_head.bin\n"); std::exit(1); }
  size_t bytes = f.tellg();
  f.seekg(0);
  p.n = int(bytes / 12);
  p.mass.resize(p.n);
  p.mol_id.resize(p.n);
  p.has_spectrum.resize(p.n);
  for (int i = 0; i < p.n; ++i) {
    float m; int32_t id, hs;
    f.read(reinterpret_cast<char*>(&m), 4);
    f.read(reinterpret_cast<char*>(&id), 4);
    f.read(reinterpret_cast<char*>(&hs), 4);
    p.mass[i] = m;
    p.mol_id[i] = id;
    p.has_spectrum[i] = hs;
  }
  return p;
}

std::vector<uint64_t> load_fp(const std::string& dir, int n) {
  std::ifstream f(dir + "/pool_fp.bin", std::ios::binary);
  if (!f) { std::fprintf(stderr, "cannot read pool_fp.bin\n"); std::exit(1); }
  std::vector<uint64_t> fp(size_t(n) * FP_WORDS);
  f.read(reinterpret_cast<char*>(fp.data()), fp.size() * 8);
  return fp;
}

inline int popcount64(uint64_t x) { return __builtin_popcountll(x); }

inline float tanimoto(const uint64_t* a, const uint64_t* b, int pa, int pb) {
  int inter = 0;
  for (int w = 0; w < FP_WORDS; ++w) inter += popcount64(a[w] & b[w]);
  int uni = pa + pb - inter;
  return uni > 0 ? float(inter) / float(uni) : 0.0f;
}

}  // namespace

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : ".";
  int topk = argc > 2 ? std::atoi(argv[2]) : 50;
  // A hard cap on how many targets to process, for timing a small slice
  // before committing to the full run - the full run is O(targets x donors)
  // and nothing about that cost was measured before the first attempt.
  size_t limit = argc > 3 ? size_t(std::atoll(argv[3])) : SIZE_MAX;

  Pool pool = load_pool(dir);
  std::vector<uint64_t> fp = load_fp(dir, pool.n);
  std::printf("pool: %d candidates\n", pool.n);

  // pool_head.bin is mass-sorted, but pool_fp.bin and pool_mols.txt are not -
  // they are in build_pool.py's original insertion order, which is what
  // pool_head.bin's mol_id field points back to. So every index used to read
  // a fingerprint has to be mol_id, never the row position i in this file;
  // mixing the two up compares fingerprints that have nothing to do with the
  // molecules the loop thinks it is looking at.
  std::vector<int> donors;
  for (int i = 0; i < pool.n; ++i) if (pool.has_spectrum[i]) donors.push_back(pool.mol_id[i]);
  std::vector<int> targets;
  for (int i = 0; i < pool.n; ++i) if (!pool.has_spectrum[i]) targets.push_back(pool.mol_id[i]);
  std::printf("donors: %zu   targets (no spectrum): %zu   topk: %d\n",
              donors.size(), targets.size(), topk);

  std::vector<int> donor_pc(donors.size());
  for (size_t d = 0; d < donors.size(); ++d) {
    const uint64_t* f = fp.data() + size_t(donors[d]) * FP_WORDS;
    int pc = 0;
    for (int w = 0; w < FP_WORDS; ++w) pc += popcount64(f[w]);
    donor_pc[d] = pc;
  }

  std::vector<int32_t> out_donor(size_t(pool.n) * topk, -1);
  std::vector<float> out_sim(size_t(pool.n) * topk, 0.0f);

  unsigned nthread = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> workers;
  std::atomic<size_t> next(0);
  std::atomic<size_t> done(0);
  auto t0 = std::chrono::steady_clock::now();

  for (unsigned t = 0; t < nthread; ++t) {
    workers.emplace_back([&]() {
      std::vector<std::pair<float, int> > top;  // (sim, donor pool-index)
      for (;;) {
        size_t ti = next++;
        if (ti >= targets.size() || ti >= limit) break;
        int cand = targets[ti];
        const uint64_t* cf = fp.data() + size_t(cand) * FP_WORDS;
        int cpc = 0;
        for (int w = 0; w < FP_WORDS; ++w) cpc += popcount64(cf[w]);

        top.clear();
        for (size_t d = 0; d < donors.size(); ++d) {
          const uint64_t* df = fp.data() + size_t(donors[d]) * FP_WORDS;
          float s = tanimoto(cf, df, cpc, donor_pc[d]);
          if (s <= 0.0f) continue;
          if (int(top.size()) < topk) {
            top.emplace_back(s, donors[d]);
            if (int(top.size()) == topk)
              std::make_heap(top.begin(), top.end(), std::greater<>());
          } else if (s > top.front().first) {
            std::pop_heap(top.begin(), top.end(), std::greater<>());
            top.back() = {s, donors[d]};
            std::push_heap(top.begin(), top.end(), std::greater<>());
          }
        }
        std::sort(top.begin(), top.end(), std::greater<>());
        for (size_t k = 0; k < top.size(); ++k) {
          out_donor[size_t(cand) * topk + k] = top[k].second;
          out_sim[size_t(cand) * topk + k] = top[k].first;
        }

        size_t d = ++done;
        size_t report_every = std::max<size_t>(1, std::min<size_t>(2000, limit / 10));
        if (d % report_every == 0) {
          double el = std::chrono::duration<double>(
              std::chrono::steady_clock::now() - t0).count();
          std::printf("  %zu / %zu  (%.0fs, %.0f/s)\n",
                      d, targets.size(), el, d / std::max(el, 1e-6));
        }
      }
    });
  }
  for (auto& w : workers) w.join();

  std::ofstream fd(dir + "/analog_donor.bin", std::ios::binary);
  fd.write(reinterpret_cast<const char*>(out_donor.data()), out_donor.size() * 4);
  std::ofstream fs(dir + "/analog_sim.bin", std::ios::binary);
  fs.write(reinterpret_cast<const char*>(out_sim.data()), out_sim.size() * 4);
  std::printf("wrote analog_donor.bin / analog_sim.bin (%d x %d)\n", pool.n, topk);
  return 0;
}
