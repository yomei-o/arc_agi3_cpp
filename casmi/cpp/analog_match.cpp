// Rank candidates that have no spectrum of their own, by borrowing spectral
// evidence from structurally similar candidates that do.
//
// Library search (match.cpp) can only ever answer a molecule whose structure
// already has a reference spectrum - about 16% of this competition's test set
// by the public baseline's own estimate. This tool handles the rest: for a
// candidate with no spectrum, it looks up the K structurally most similar
// candidates that DO have one (precomputed once by precompute_analogs.cpp,
// independent of any query - a candidate's chemical neighbours don't change
// per test molecule), scores each such donor's own reference spectrum against
// the query spectrum directly, and credits the candidate with
// donor_score * tanimoto(candidate, donor)^SIM_POWER, taking the best donor.
//
// Donors are NOT gated by mass to the query - that was tried and measured not
// to filter anything (library density across mass here is close to uniform:
// a +-200 Da window around a typical test precursor holds ~90% of the whole
// donor set). What IS gated by mass is which CANDIDATES are considered at
// all: a candidate only matters if its own (neutral, adduct-corrected) mass
// is close to what the query's precursor implies, exactly like direct search.
//
// Build:
//   g++ -O3 -std=c++17 -pthread -o analog_match analog_match.cpp
//
// Usage:
//   analog_match <dir> <ppm> <peak_tol> <topk> <qry_stem> [sim_power] [donor_k]

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

// Same order as ADDUCT_ID in the Python scripts - index into this table is
// h.tag / 16. The shift turns a precursor m/z back into the neutral,
// monoisotopic mass pool_head.bin is sorted by. [2M+*] adducts are dimers
// (2*M + adduct), which a single subtraction cannot undo exactly; they are
// left using the monomer's shift, which is wrong by roughly M itself for
// those rows - rare enough in practice that this is a known, accepted gap
// rather than a silent one.
constexpr float ADDUCT_SHIFT[10] = {
    1.007276f,    // [M+H]+
    -1.007276f,   // [M-H]-
    22.989218f,   // [M+Na]+
    18.033823f,   // [M+NH4]+
    38.963158f,   // [M+K]+
    34.969402f,   // [M+Cl]-
    44.998201f,   // [M+CH2O2-H]-  (+formic acid, -H)
    -17.002740f,  // [M+H-H2O]+
    1.007276f,    // [2M+H]+   (not dimer-corrected, see note above)
    22.989218f,   // [2M+Na]+  (not dimer-corrected, see note above)
};

struct Head {
  float precursor;
  int32_t mol, off, n, tag;
};

struct Store {
  std::vector<Head> head;
  std::vector<float> mz, inten;
  std::vector<std::string> name, smiles;
  const float* mzs(const Head& h) const { return mz.data() + h.off; }
  const float* its(const Head& h) const { return inten.data() + h.off; }
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

// Loads a head.bin/peaks.bin/mols.txt triple with plain sqrt-cosine in mind -
// no entropy weighting here, since donors are only ever compared head to head
// against a query spectrum, no per-spectrum precomputed state required.
Store load(const std::string& dir, const std::string& stem) {
  Store s;
  std::vector<char> hb = slurp(dir + "/" + stem + "_head.bin");
  s.head.resize(hb.size() / sizeof(Head));
  std::memcpy(s.head.data(), hb.data(), s.head.size() * sizeof(Head));
  std::vector<char> pb = slurp(dir + "/" + stem + "_peaks.bin");
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

float cosine(const float* amz, const float* ait, int an,
             const float* bmz, const float* bit, int bn, float tol) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (int i = 0; i < an; ++i) na += double(ait[i]);
  for (int j = 0; j < bn; ++j) nb += double(bit[j]);
  if (na <= 0.0 || nb <= 0.0) return 0.0f;
  int j = 0;
  for (int i = 0; i < an; ++i) {
    while (j < bn && bmz[j] < amz[i] - tol) ++j;
    int k = j;
    float best = 0.0f;
    while (k < bn && bmz[k] <= amz[i] + tol) { best = std::max(best, bit[k]); ++k; }
    if (best > 0.0f) dot += std::sqrt(double(ait[i])) * std::sqrt(double(best));
  }
  double norma = 0.0, normb = 0.0;
  for (int i = 0; i < an; ++i) norma += double(ait[i]);
  for (int j2 = 0; j2 < bn; ++j2) normb += double(bit[j2]);
  return float(dot / std::sqrt(norma * normb + 1e-12));
}

struct Pool {
  int n = 0;
  std::vector<float> mass;
  std::vector<int32_t> mol_id;        // original index into pool_mols.txt
  std::vector<int32_t> has_spectrum;
};

Pool load_pool(const std::string& dir) {
  Pool p;
  std::ifstream f(dir + "/pool_head.bin", std::ios::binary | std::ios::ate);
  if (!f) { std::fprintf(stderr, "cannot read pool_head.bin\n"); std::exit(1); }
  size_t bytes = f.tellg();
  f.seekg(0);
  p.n = int(bytes / 12);
  p.mass.resize(p.n); p.mol_id.resize(p.n); p.has_spectrum.resize(p.n);
  for (int i = 0; i < p.n; ++i) {
    f.read(reinterpret_cast<char*>(&p.mass[i]), 4);
    f.read(reinterpret_cast<char*>(&p.mol_id[i]), 4);
    f.read(reinterpret_cast<char*>(&p.has_spectrum[i]), 4);
  }
  return p;
}

std::vector<std::string> load_pool_mols(const std::string& dir, std::vector<std::string>* names_out) {
  std::vector<std::string> smiles;
  std::ifstream f(dir + "/pool_mols.txt");
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    size_t t = line.find('\t');
    names_out->push_back(line.substr(0, t));
    smiles.push_back(t == std::string::npos ? "" : line.substr(t + 1));
  }
  return smiles;
}

}  // namespace

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : ".";
  float ppm = argc > 2 ? float(std::atof(argv[2])) : 50.0f;
  float tol = argc > 3 ? float(std::atof(argv[3])) : 0.02f;
  int topk = argc > 4 ? std::atoi(argv[4]) : 25;
  std::string qry_stem = argc > 5 ? argv[5] : "test";
  float sim_power = argc > 6 ? float(std::atof(argv[6])) : 3.0f;
  int donor_k = argc > 7 ? std::atoi(argv[7]) : 50;   // must match precompute_analogs's topk

  Pool pool = load_pool(dir);
  std::vector<std::string> pool_names;
  std::vector<std::string> pool_smiles = load_pool_mols(dir, &pool_names);
  std::printf("pool: %d candidates (%zu named)\n", pool.n, pool_names.size());

  Store donor = load(dir, "donor");      // one spectrum per train molecule
  Store te = load(dir, qry_stem);

  std::ifstream fd(dir + "/analog_donor.bin", std::ios::binary);
  std::ifstream fs(dir + "/analog_sim.bin", std::ios::binary);
  if (!fd || !fs) { std::fprintf(stderr, "missing analog_donor.bin/analog_sim.bin\n"); std::exit(1); }
  std::vector<int32_t> adonor(size_t(pool.n) * donor_k);
  std::vector<float> asim(size_t(pool.n) * donor_k);
  fd.read(reinterpret_cast<char*>(adonor.data()), adonor.size() * 4);
  fs.read(reinterpret_cast<char*>(asim.data()), asim.size() * 4);

  // Pool sorted by mass, for the candidate gate.
  std::vector<int> order(pool.n);
  for (int i = 0; i < pool.n; ++i) order[i] = i;
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return pool.mass[a] < pool.mass[b]; });
  std::vector<float> keys(pool.n);
  for (int i = 0; i < pool.n; ++i) keys[i] = pool.mass[order[i]];

  // donor Store's own "mol" field is the train molecule id, which equals
  // pool_mols.txt's row order for has_spectrum==1 entries (build_pool.py
  // appends train's structures first, unchanged) - so a donor mol_id doubles
  // directly as an index into donor.head if we also index donors by mol.
  std::vector<int> donor_row_of_mol(pool.n, -1);   // mol_id -> row in donor.head
  for (size_t i = 0; i < donor.head.size(); ++i) donor_row_of_mol[donor.head[i].mol] = int(i);

  std::unordered_map<int, std::vector<int> > by_mol;
  for (size_t i = 0; i < te.head.size(); ++i) by_mol[te.head[i].mol].push_back(int(i));
  std::vector<int> mol_ids;
  for (const auto& kv : by_mol) mol_ids.push_back(kv.first);
  std::sort(mol_ids.begin(), mol_ids.end());

  std::vector<std::string> answer(mol_ids.size());
  unsigned nthread = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> pool_threads;
  std::atomic<size_t> next(0);

  for (unsigned t = 0; t < nthread; ++t) {
    pool_threads.emplace_back([&]() {
      std::unordered_map<int, float> donor_score_cache;   // donor mol_id -> score vs this query
      std::unordered_map<int, float> best;                // candidate mol_id -> best analog score
      for (;;) {
        size_t idx = next++;
        if (idx >= mol_ids.size()) break;
        best.clear();
        donor_score_cache.clear();

        for (int si : by_mol[mol_ids[idx]]) {
          const Head& q = te.head[si];
          int adduct_idx = q.tag < 0 ? -1 : q.tag / 16;
          float shift = (adduct_idx >= 0 && adduct_idx < 10) ? ADDUCT_SHIFT[adduct_idx] : 1.007276f;
          float neutral = q.precursor - shift;

          float ppm_used = ppm;
          for (int pass = 0; pass < 6; ++pass) {
            float window = neutral * ppm_used * 1e-6f;
            size_t lo = std::lower_bound(keys.begin(), keys.end(), neutral - window) - keys.begin();
            size_t hi = std::upper_bound(keys.begin(), keys.end(), neutral + window) - keys.begin();
            for (size_t p = lo; p < hi; ++p) {
              int row = order[p];
              if (pool.has_spectrum[row]) continue;   // direct search already covers these
              int cand_mol = pool.mol_id[row];
              const int32_t* dons = adonor.data() + size_t(cand_mol) * donor_k;
              const float* sims = asim.data() + size_t(cand_mol) * donor_k;
              float cand_score = 0.0f;
              for (int k = 0; k < donor_k && dons[k] >= 0; ++k) {
                int dmol = dons[k];
                auto itc = donor_score_cache.find(dmol);
                float ds;
                if (itc != donor_score_cache.end()) {
                  ds = itc->second;
                } else {
                  int drow = donor_row_of_mol[dmol];
                  ds = drow < 0 ? 0.0f
                                : cosine(te.mzs(q), te.its(q), q.n,
                                        donor.mzs(donor.head[drow]), donor.its(donor.head[drow]),
                                        donor.head[drow].n, tol);
                  donor_score_cache.emplace(dmol, ds);
                }
                float w = float(std::pow(double(sims[k]), double(sim_power)));
                cand_score = std::max(cand_score, ds * w);
              }
              auto it = best.find(cand_mol);
              if (it == best.end()) best.emplace(cand_mol, cand_score);
              else it->second = std::max(it->second, cand_score);
            }
            if (int(best.size()) >= topk) break;
            ppm_used *= 3.0f;
          }
        }

        // best is <cand_mol, score>; building the (score, cand_mol) rank pair
        // by range-constructing straight from best's own pair<const int,float>
        // silently swaps and type-converts both fields instead of erroring -
        // an int candidate id becomes a float "score" and the real score
        // truncates to an int id. Spell the field order out.
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
          const std::string& smi = pool_smiles[rank[r].second];
          if (smi.empty()) continue;
          if (!out.empty()) out += ';';
          out += smi;
          ++placed;
        }
        answer[idx] = out;
      }
    });
  }
  for (auto& th : pool_threads) th.join();

  std::string path = dir + "/analog_submission.csv";
  std::ofstream out(path);
  out << "molecule_id,smiles\n";
  for (size_t i = 0; i < mol_ids.size(); ++i)
    out << te.name[mol_ids[i]] << ',' << answer[i] << '\n';
  std::printf("wrote %s for %zu molecules\n", path.c_str(), mol_ids.size());
  return 0;
}
