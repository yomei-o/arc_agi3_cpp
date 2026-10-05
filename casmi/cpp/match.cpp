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
//
// Usage:
//   match <dir> <ppm> <peak_tol> <topk> <lib_stem> <qry_stem> [sigma_ppm] [alpha]

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
  std::vector<float> ent;        // Shannon entropy of each spectrum, after weighting

  const float* mzs(const Head& h) const { return mz.data() + h.off; }
  const float* its(const Head& h) const { return inten.data() + h.off; }
  int adduct(const Head& h) const { return h.tag < 0 ? -1 : h.tag / 16; }
  int instr(const Head& h) const { return h.tag < 0 ? -1 : h.tag % 16; }
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
  // Entropy weighting, Li et al. 2021.
  //
  // A plain dot product treats a three-peak spectrum and an eighty-peak one as
  // equally trustworthy. They are not: a low-entropy spectrum carries few
  // independent constraints and matches far too many things by accident. The
  // fix is to flatten the confident-looking ones before comparing, raising the
  // intensities to w = 0.25 + 0.25 S when the entropy S is below 3, and to
  // score with entropy similarity instead of cosine. The one published
  // measurement on this competition puts Class-1 MRR at 0.919 with it against
  // 0.895 with sqrt intensities.
  s.ent.resize(s.head.size());
  for (size_t i = 0; i < s.head.size(); ++i) {
    const Head& h = s.head[i];
    float* it = s.inten.data() + h.off;
    double sum = 0.0;
    for (int k = 0; k < h.n; ++k) sum += double(it[k]);
    if (sum <= 0.0) { s.ent[i] = 0.0f; continue; }
    for (int k = 0; k < h.n; ++k) it[k] = float(double(it[k]) / sum);
    double S = 0.0;
    for (int k = 0; k < h.n; ++k)
      if (it[k] > 0.0f) S -= double(it[k]) * std::log(double(it[k]));
    if (S < 3.0) {
      double w = 0.25 + 0.25 * S, s2 = 0.0;
      for (int k = 0; k < h.n; ++k) { it[k] = float(std::pow(double(it[k]), w)); s2 += it[k]; }
      if (s2 > 0.0) {
        S = 0.0;
        for (int k = 0; k < h.n; ++k) {
          it[k] = float(double(it[k]) / s2);
          if (it[k] > 0.0f) S -= double(it[k]) * std::log(double(it[k]));
        }
      }
    }
    s.ent[i] = float(S);
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

// Entropy similarity. Both spectra are already normalised to sum 1 and
// weighted, so their merge is the average of the two and
//     sim = 1 - (2 S_AB - S_A - S_B) / ln 4
// which is 1 for identical spectra and 0 for disjoint ones.
float entropy_sim(const float* amz, const float* ait, int an, float sa,
                  const float* bmz, const float* bit, int bn, float sb,
                  float tol) {
  double sab = 0.0;
  int i = 0, j = 0;
  auto add = [&sab](double q) { if (q > 0.0) sab -= q * std::log(q); };
  while (i < an && j < bn) {
    if (bmz[j] < amz[i] - tol) { add(0.5 * double(bit[j])); ++j; }
    else if (bmz[j] > amz[i] + tol) { add(0.5 * double(ait[i])); ++i; }
    else { add(0.5 * (double(ait[i]) + double(bit[j]))); ++i; ++j; }
  }
  for (; i < an; ++i) add(0.5 * double(ait[i]));
  for (; j < bn; ++j) add(0.5 * double(bit[j]));
  double v = 1.0 - (2.0 * sab - double(sa) - double(sb)) / std::log(4.0);
  return float(v < 0.0 ? 0.0 : v > 1.0 ? 1.0 : v);
}

// A molecule's several query spectra merged into one before matching, instead
// of matching each separately and taking the best candidate score across
// them. Close peaks (within tol) are summed rather than kept apart, then the
// merge is renormalised and entropy-weighted exactly as load() treats a
// stored spectrum - a merged spectrum is just a spectrum.
//
// The tutorial notebook reports feeding the wrong one of these two costs
// ~0.02 MRR; which one is wrong is exactly what this flag is for measuring.
struct MergedSpec { std::vector<float> mz, it; float ent = 0.0f; };

MergedSpec merge_query(const Store& te, const std::vector<int>& sis, float tol) {
  std::vector<std::pair<float, float> > peaks;
  for (int si : sis) {
    const Head& q = te.head[si];
    const float* mz = te.mzs(q);
    const float* it = te.its(q);
    for (int k = 0; k < q.n; ++k) peaks.emplace_back(mz[k], it[k]);
  }
  std::sort(peaks.begin(), peaks.end());
  MergedSpec out;
  for (auto& pr : peaks) {
    if (!out.mz.empty() && pr.first - out.mz.back() <= tol) out.it.back() += pr.second;
    else { out.mz.push_back(pr.first); out.it.push_back(pr.second); }
  }
  double sum = 0.0;
  for (float x : out.it) sum += double(x);
  if (sum <= 0.0) return out;
  for (float& x : out.it) x = float(double(x) / sum);
  double S = 0.0;
  for (float x : out.it) if (x > 0.0f) S -= double(x) * std::log(double(x));
  if (S < 3.0) {
    double w = 0.25 + 0.25 * S, s2 = 0.0;
    for (float& x : out.it) { x = float(std::pow(double(x), w)); s2 += double(x); }
    if (s2 > 0.0) {
      S = 0.0;
      for (float& x : out.it) {
        x = float(double(x) / s2);
        if (x > 0.0f) S -= double(x) * std::log(double(x));
      }
    }
  }
  out.ent = float(S);
  return out;
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
  // How far the precursor is allowed to be wrong, and how much being wrong
  // costs. See the scoring loop.
  float sigma = argc > 7 ? float(std::atof(argv[7])) : 3.0f;
  float alpha = argc > 8 ? float(std::atof(argv[8])) : 0.1f;
  int simmode = argc > 9 ? std::atoi(argv[9]) : 1;   // 1 entropy, 0 sqrt cosine
  // 0 off, 1 soft bonus for a same-instrument hit, 2 hard-require same
  // instrument when both are known. Cross-instrument fragmentation patterns
  // differ systematically (declustering voltage, collision cell physics), so
  // this is worth a controlled measurement rather than an assumption either
  // way: a hard filter can delete the only reference a Class-1 molecule has.
  int instr_mode = argc > 10 ? std::atoi(argv[10]) : 0;
  float instr_bonus = argc > 11 ? float(std::atof(argv[11])) : 0.1f;
  // Match each of a molecule's spectra separately and keep the best candidate
  // score across them (0), or merge the spectra into one before matching (1).
  int merge_mode = argc > 12 ? std::atoi(argv[12]) : 0;

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

        MergedSpec merged;
        if (merge_mode) merged = merge_query(te, by_mol[mol_ids[idx]], tol);

        // One window, and being near the middle of it is worth something.
        //
        // The window used to start at 0.05 ppm and widen by 3x until 25
        // candidates existed, with a flat penalty per widening. That was tuned
        // on a holdout carved out of train, where it looked twenty times
        // better than a 1 ppm window - because train's precursor_mz is
        // calculated rather than measured (63.5% of (molecule, adduct) groups
        // are bit-identical, p99 spread under 1 ppm), so a query matched its
        // own library rows to the last digit. Re-run against queries carrying
        // 3 ppm of instrument error, which is what a real test spectrum has,
        // and the staged scheme drops from 0.481 MRR to 0.158.
        //
        // So the mass is evidence, not a gate: a candidate 1 ppm away is
        // better than one 8 ppm away, and the cost should grow smoothly rather
        // than in steps of a whole point at arbitrary 3x boundaries.
        // Gaussian in the ppm error, which is what the error actually is.
        float ppm_used = ppm;
        for (int pass = 0; pass < 10 && int(best.size()) < topk; ++pass) {
          for (int si : by_mol[mol_ids[idx]]) {
            const Head& q = te.head[si];
            float window = q.precursor * ppm_used * 1e-6f;
            size_t lo = std::lower_bound(keys.begin(), keys.end(),
                                         q.precursor - window) - keys.begin();
            size_t hi = std::upper_bound(keys.begin(), keys.end(),
                                         q.precursor + window) - keys.begin();
            for (size_t p = lo; p < hi; ++p) {
              const Head& c = tr.head[order[p]];
              if (q.tag >= 0 && c.tag >= 0 && te.adduct(q) != tr.adduct(c)) continue;
              bool same_instr = q.tag >= 0 && c.tag >= 0 && te.instr(q) == tr.instr(c);
              if (instr_mode == 2 && q.tag >= 0 && c.tag >= 0 && !same_instr) continue;
              const float* qmz = merge_mode ? merged.mz.data() : te.mzs(q);
              const float* qit = merge_mode ? merged.it.data() : te.its(q);
              int qn = merge_mode ? int(merged.mz.size()) : q.n;
              float qent = merge_mode ? merged.ent : te.ent[si];
              float s = simmode
                  ? entropy_sim(qmz, qit, qn, qent,
                                tr.mzs(c), tr.its(c), c.n, tr.ent[order[p]], tol)
                  : cosine(qmz, qit, qn,
                           tr.mzs(c), tr.its(c), c.n, tol);
              if (instr_mode == 1 && same_instr) s += instr_bonus;
              float dppm = (c.precursor - q.precursor) / q.precursor * 1e6f;
              float z = dppm / sigma;
              s -= alpha * z * z;
              auto it = best.find(c.mol);
              if (it == best.end()) best.emplace(c.mol, s);
              else it->second = std::max(it->second, s);
            }
          }
          ppm_used *= 3.0f;
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
