// Updates pool_recall.py's question (COCONUT: how much of the denovo-holdout
// 84% does an auxiliary structure pool actually reach, before any ranking
// code touches it?) for the now-PubChem-expanded pool, canonical-SMILES
// equality instead of InChIKey14 (new pool entries have no InChIKey - see
// RESUME.md). For each denovo-holdout molecule, gather every pool candidate
// within a ppm window of its own neutral mass (same windowing forward_holdout
// uses) and check whether the true structure's canonical SMILES is among
// them - a ceiling check, no scoring.
//
// Usage: pool_recall_check <dir> <dqry_stem> <ppm>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/PeriodicTable.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>

namespace {

struct Head { float precursor; int32_t mol, off, n, tag; };

std::vector<char> slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(1); }
  std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<char> buf((size_t(n)));
  f.read(buf.data(), n);
  return buf;
}

struct QryStore {
  std::vector<Head> head;
  std::vector<std::string> name, smiles;
};

QryStore loadQry(const std::string& dir, const std::string& stem) {
  QryStore s;
  std::vector<char> hb = slurp(dir + "/" + stem + "_head.bin");
  s.head.resize(hb.size() / sizeof(Head));
  std::memcpy(s.head.data(), hb.data(), s.head.size() * sizeof(Head));
  std::ifstream f(dir + "/" + stem + "_mols.txt");
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    size_t t = line.find('\t');
    s.name.push_back(line.substr(0, t));
    s.smiles.push_back(t == std::string::npos ? "" : line.substr(t + 1));
  }
  return s;
}

struct Pool {
  std::vector<float> mass;      // by mol_id
  std::vector<std::string> smiles;  // by mol_id
  std::vector<bool> hasSpec;    // by mol_id - true for train's own 275,810 structures
};

Pool loadPool(const std::string& dir) {
  Pool p;
  std::ifstream f(dir + "/pool_head.bin", std::ios::binary | std::ios::ate);
  size_t bytes = f.tellg(); f.seekg(0);
  int n = int(bytes / 12);
  std::vector<int32_t> molId(n), hasSpecFlag(n);
  std::vector<float> massSorted(n);
  for (int i = 0; i < n; ++i) {
    f.read(reinterpret_cast<char*>(&massSorted[i]), 4);
    f.read(reinterpret_cast<char*>(&molId[i]), 4);
    f.read(reinterpret_cast<char*>(&hasSpecFlag[i]), 4);
  }
  std::ifstream mf(dir + "/pool_mols.txt");
  std::string line;
  while (std::getline(mf, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    size_t t = line.find('\t');
    p.smiles.push_back(t == std::string::npos ? "" : line.substr(t + 1));
  }
  p.mass.resize(p.smiles.size());
  p.hasSpec.resize(p.smiles.size());
  for (int i = 0; i < n; ++i) {
    p.mass[molId[i]] = massSorted[i];
    p.hasSpec[molId[i]] = hasSpecFlag[i] != 0;
  }
  std::fprintf(stderr, "pool: %zu candidates\n", p.smiles.size());
  return p;
}

constexpr float ADDUCT_SHIFT[10] = {
    1.007276f, -1.007276f, 22.989218f, 18.033823f, 38.963158f,
    34.969402f, 44.998201f, -17.002740f, 1.007276f, 22.989218f,
};

RDKit::RWMol* safeParse(const std::string& smi) {
  try { return RDKit::SmilesToMol(smi); } catch (...) { return nullptr; }
}

std::string canon(const std::string& smi) {
  std::unique_ptr<RDKit::RWMol> m(safeParse(smi));
  if (!m) return "";
  return RDKit::MolToSmiles(*m);
}

}  // namespace

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : ".";
  std::string dqryStem = argc > 2 ? argv[2] : "dqry";
  float ppm = argc > 3 ? float(std::atof(argv[3])) : 20.0f;

  QryStore dqry = loadQry(dir, dqryStem);
  Pool pool = loadPool(dir);
  std::vector<float> sortedMass = pool.mass;
  std::vector<int> order(pool.mass.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = int(i);
  std::sort(order.begin(), order.end(), [&](int a, int b) { return pool.mass[a] < pool.mass[b]; });
  for (size_t i = 0; i < order.size(); ++i) sortedMass[i] = pool.mass[order[i]];

  RDKit::PeriodicTable::getTable();

  std::unordered_map<int, std::vector<int>> byMol;
  for (size_t i = 0; i < dqry.head.size(); ++i) byMol[dqry.head[i].mol].push_back(int(i));
  std::vector<int> molIds;
  for (auto& kv : byMol) molIds.push_back(kv.first);

  std::atomic<size_t> next(0);
  std::atomic<int> nTested(0), nFound(0), nFoundNonTrain(0);
  std::mutex printMutex;

  unsigned nthread = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> threads;
  for (unsigned t = 0; t < nthread; ++t) {
    threads.emplace_back([&]() {
      for (;;) {
        size_t mi = next++;
        if (mi >= molIds.size()) break;
        int molId = molIds[mi];
        const std::string& truthSmi = dqry.smiles[molId];
        if (truthSmi.empty()) continue;
        std::string truthCanon = canon(truthSmi);
        if (truthCanon.empty()) continue;

        std::set<int> candIdx;
        for (int qi : byMol[molId]) {
          const Head& q = dqry.head[qi];
          int adductIdx = q.tag < 0 ? -1 : q.tag / 16;
          float shift = (adductIdx >= 0 && adductIdx < 10) ? ADDUCT_SHIFT[adductIdx] : 1.007276f;
          float neutral = q.precursor - shift;
          float window = neutral * ppm * 1e-6f;
          size_t lo = std::lower_bound(sortedMass.begin(), sortedMass.end(), neutral - window) - sortedMass.begin();
          size_t hi = std::upper_bound(sortedMass.begin(), sortedMass.end(), neutral + window) - sortedMass.begin();
          for (size_t p = lo; p < hi; ++p) candIdx.insert(order[p]);
        }
        if (candIdx.empty()) continue;
        ++nTested;

        bool found = false, foundNonTrain = false;
        for (int idx : candIdx) {
          if (canon(pool.smiles[idx]) == truthCanon) {
            found = true;
            if (!pool.hasSpec[idx]) foundNonTrain = true;
          }
        }
        if (found) ++nFound;
        if (foundNonTrain) ++nFoundNonTrain;
        std::lock_guard<std::mutex> lk(printMutex);
        std::printf("  [mol %d] candidates=%zu found=%s found_non_train=%s\n", molId, candIdx.size(),
                    found ? "yes" : "no", foundNonTrain ? "yes" : "no");
      }
    });
  }
  for (auto& th : threads) th.join();

  std::printf("\ndenovo-holdout: %d/%zu molecules had any mass-window candidate\n",
              nTested.load(), molIds.size());
  std::printf("true structure present in pool incl. train (ppm=%.1f): %d/%d  (%.1f%%)"
              " - trivially ~100%% since dqry IS a subset of train's own structures\n",
              ppm, nFound.load(), nTested.load(),
              nTested.load() ? 100.0 * nFound.load() / nTested.load() : 0.0);
  std::printf("true structure ALSO present via a non-train (COCONUT/PubChem) entry: %d/%d  (%.1f%%)"
              " - the real ceiling-raise question\n",
              nFoundNonTrain.load(), nTested.load(),
              nTested.load() ? 100.0 * nFoundNonTrain.load() / nTested.load() : 0.0);
  return 0;
}
