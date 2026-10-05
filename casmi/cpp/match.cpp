// Identify a molecule by finding the training spectra that look like it.
//
// The shape of the problem, measured before any of this was written: 400 test
// molecules, 1,213 spectra, all from one instrument; 2.54 million training
// spectra over 276 thousand molecules. Every test precursor mass exists in the
// training set, but within 10 ppm the median mass has 105 distinct molecules
// behind it and the worst has 558 - so the mass narrows the field to about a
// hundred and the spectrum has to do the rest. Only 38 of the 400 have few
// enough candidates that mass alone would fill the answer.
//
// That is a billion peak-list comparisons, which is why this is not Python.
//
// Build:
//   g++ -O3 -std=c++17 -pthread -o match match.cpp

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

struct Head {            // one per spectrum, as written by export_bin.py
  float precursor;
  int32_t mol;
  int32_t off;           // index into the peak arrays
  int32_t n;             // how many peaks
  int32_t tag;           // adduct * 16 + instrument, -1 where unknown
};

struct Store {
  std::vector<Head> head;
  std::vector<float> mz, inten;
  std::vector<std::string> name, smiles;

  const float* mzs(const Head& h) const { return mz.data() + h.off; }
  const float* its(const Head& h) const { return inten.data() + h.off; }
  int adduct(const Head& h) const { return h.tag < 0 ? -1 : h.tag / 16; }
};

std::vector<char> slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(1); }
  std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<char> buf((size_t(n)));
  f.read(buf.data(), n);
  return buf;
}

Store load(const std::string& dir, const std::string& stem) {
  Store s;
  std::vector<char> hb = slurp(dir + "/" + stem + "_head.bin");
  s.head.resize(hb.size() / sizeof(Head));
  std::memcpy(s.head.data(), hb.data(), s.head.size() * sizeof(Head));

  std::vector<char> pb = slurp(dir + "/" + stem + "_peaks.bin");
  // Each spectrum wrote its m/z block then its intensity block, so the pairs
  // are interleaved per spectrum rather than globally. Split them out once.
  size_t total = 0;
  for (const Head& h : s.head) total += size_t(h.n);
  s.mz.resize(total);
  s.inten.resize(total);
  const float* p = reinterpret_cast<const float*>(pb.data());
  size_t cur = 0;
  for (const Head& h : s.head) {
    std::memcpy(s.mz.data() + cur, p, size_t(h.n) * sizeof(float));
    p += h.n;
    std::memcpy(s.inten.data() + cur, p, size_t(h.n) * sizeof(float));
    p += h.n;
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
  std::printf("%s: %zu spectra, %zu molecules, %zu peaks\n",
              stem.c_str(), s.head.size(), s.name.size(), total);
  return s;
}

// Cosine over matched peaks, the standard way these libraries are searched.
//
// Peaks match when their m/z agree within a tolerance; both lists are sorted,
// so one walk of each is enough. Intensities are square-rooted first, which is
// what every spectral library search does - it stops one tall peak from
// deciding the whole score, and tall peaks are the least informative part of a
// fragmentation pattern.
float cosine(const float* amz, const float* ait, int an,
             const float* bmz, const float* bit, int bn, float tol) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (int i = 0; i < an; ++i) na += double(ait[i]);
  for (int j = 0; j < bn; ++j) nb += double(bit[j]);
  if (na <= 0.0 || nb <= 0.0) return 0.0;

  int j = 0;
  for (int i = 0; i < an; ++i) {
    while (j < bn && bmz[j] < amz[i] - tol) ++j;
    int k = j;
    float best = 0.0f;
    while (k < bn && bmz[k] <= amz[i] + tol) {
      best = std::max(best, bit[k]);
      ++k;
    }
    if (best > 0.0f) dot += std::sqrt(double(ait[i])) * std::sqrt(double(best));
  }
  double norma = 0.0, normb = 0.0;
  for (int i = 0; i < an; ++i) norma += double(ait[i]);
  for (int j2 = 0; j2 < bn; ++j2) normb += double(bit[j2]);
  return float(dot / std::sqrt(norma * normb + 1e-12));
}

}  // namespace

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : ".";
  float ppm = argc > 2 ? float(std::atof(argv[2])) : 10.0f;
  float tol = argc > 3 ? float(std::atof(argv[3])) : 0.01f;
  int topk = argc > 4 ? std::atoi(argv[4]) : 25;
  // Swap in the holdout by name, so the thing that is measured is the thing
  // that is submitted and not a second implementation of it.
  std::string lib_stem = argc > 5 ? argv[5] : "train";
  std::string qry_stem = argc > 6 ? argv[6] : "test";

  Store tr = load(dir, lib_stem);
  Store te = load(dir, qry_stem);

  // Training spectra sorted by precursor mass, so a candidate window is a
  // binary search rather than a scan of two and a half million rows.
  std::vector<int> order(tr.head.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = int(i);
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return tr.head[a].precursor < tr.head[b].precursor; });
  std::vector<float> keys(order.size());
  for (size_t i = 0; i < order.size(); ++i) keys[i] = tr.head[order[i]].precursor;

  // Group the test spectra by molecule: several spectra of one molecule are
  // evidence about the same answer and their scores add.
  std::unordered_map<int, std::vector<int> > by_mol;
  for (size_t i = 0; i < te.head.size(); ++i) by_mol[te.head[i].mol].push_back(int(i));

  std::vector<int> mol_ids;
  for (const auto& kv : by_mol) mol_ids.push_back(kv.first);
  std::sort(mol_ids.begin(), mol_ids.end());

  std::vector<std::string> answer(mol_ids.size());
  std::vector<int> hit_rank(mol_ids.size(), -1);
  unsigned nthread = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> pool;
  std::atomic<size_t> next(0);

  for (unsigned t = 0; t < nthread; ++t) {
    pool.emplace_back([&]() {
      std::unordered_map<int, float> best;      // train molecule -> best score
      for (;;) {
        size_t idx = next++;
        if (idx >= mol_ids.size()) break;
        best.clear();

        for (int si : by_mol[mol_ids[idx]]) {
          const Head& q = te.head[si];
          float window = q.precursor * ppm * 1e-6f;
          size_t lo = std::lower_bound(keys.begin(), keys.end(), q.precursor - window)
                      - keys.begin();
          size_t hi = std::upper_bound(keys.begin(), keys.end(), q.precursor + window)
                      - keys.begin();
          for (size_t p = lo; p < hi; ++p) {
            const Head& c = tr.head[order[p]];
            if (q.tag >= 0 && c.tag >= 0 && te.adduct(q) != tr.adduct(c)) continue;
            float s = cosine(te.mzs(q), te.its(q), q.n,
                             tr.mzs(c), tr.its(c), c.n, tol);
            auto it = best.find(c.mol);
            if (it == best.end()) best.emplace(c.mol, s);
            else it->second = std::max(it->second, s);
          }
        }

        std::vector<std::pair<float, int> > rank;
        rank.reserve(best.size());
        for (const auto& kv : best) rank.emplace_back(kv.second, kv.first);
        std::sort(rank.begin(), rank.end(),
                  [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                    return a.first > b.first;
                  });

        std::string out;
        int placed = 0;
        for (size_t r = 0; r < rank.size() && placed < topk; ++r) {
          const std::string& smi = tr.smiles[rank[r].second];
          if (smi.empty()) continue;          // a molecule with no structure
          if (!out.empty()) out += ';';
          out += smi;
          ++placed;
          // On a holdout the query carries its own answer, so note where it
          // landed. The run then scores itself and no submission is spent
          // finding out whether a change helped.
          if (hit_rank[idx] < 0 && tr.name[rank[r].second] == te.name[mol_ids[idx]])
            hit_rank[idx] = placed;
        }
        answer[idx] = out;
      }
    });
  }
  for (std::thread& th : pool) th.join();

  std::string path = dir + "/submission.csv";
  std::ofstream out(path);
  out << "molecule_id,smiles\n";
  for (size_t i = 0; i < mol_ids.size(); ++i)
    out << te.name[mol_ids[i]] << ',' << answer[i] << '\n';
  std::printf("wrote %s for %zu molecules\n", path.c_str(), mol_ids.size());

  int found = 0;
  for (int r : hit_rank) found += (r > 0);
  if (found) {
    double mrr = 0.0;
    int t1 = 0, t5 = 0;
    for (int r : hit_rank) {
      if (r <= 0) continue;
      mrr += 1.0 / r;
      t1 += (r == 1);
      t5 += (r <= 5);
    }
    size_t n = mol_ids.size();
    std::printf("top1 %.1f%%  top5 %.1f%%  top%d %.1f%%  MRR %.3f  (n=%zu)\n",
                100.0 * t1 / n, 100.0 * t5 / n, topk, 100.0 * found / n,
                mrr / n, n);
  }
  return 0;
}
