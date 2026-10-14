// forward_rerank.cpp, but scored with the trained MLP
// (forward_model_nn::predictSpectrumNN) instead of forward_model.h's fixed
// exp(-decay*k^2) rule. The rule-based version was submitted as v8 and
// REGRESSED the real public LB (0.121 -> 0.102) despite a clean holdout
// number - see RESUME.md. This version exists because that result alone
// doesn't tell us whether reranking same-formula groups is a bad idea in
// general, or whether the specific rule-based scorer was the problem; the
// trained model (holdout MRR 0.538 rule-based -> 0.666 trained, same pool/
// ppm) is the thing actually worth spending a submission on next.
//
// Usage:
//   forward_rerank_nn <dir> <qry_stem> <mlp.bin> <in.csv> <out.csv>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "forward_model_nn.h"

namespace {

struct Head { float precursor; int32_t mol, off, n, tag; };

struct Store {
  std::vector<Head> head;
  std::vector<float> mz, inten;
  std::vector<std::string> name;
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
  for (const Head& h : s.head) {
    std::memcpy(s.mz.data() + cur, p, size_t(h.n) * sizeof(float)); p += h.n;
    std::memcpy(s.inten.data() + cur, p, size_t(h.n) * sizeof(float)); p += h.n;
    cur += size_t(h.n);
  }
  std::ifstream f(dir + "/" + stem + "_mols.txt");
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    size_t t = line.find('\t');
    s.name.push_back(line.substr(0, t));
  }
  std::printf("%s: %zu spectra, %zu molecules\n", stem.c_str(), s.head.size(), s.name.size());
  return s;
}

float cosine(const std::vector<std::pair<double, double>>& pred,
             const float* mz, const float* it, int n, float tol) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (auto& pk : pred) na += pk.second;
  for (int i = 0; i < n; ++i) nb += double(it[i]);
  if (na <= 0.0 || nb <= 0.0) return 0.0f;
  for (auto& pk : pred) {
    double best = 0.0;
    for (int i = 0; i < n; ++i)
      if (std::fabs(double(mz[i]) - pk.first) <= tol) best = std::max(best, double(it[i]));
    if (best > 0.0) dot += std::sqrt(pk.second) * std::sqrt(best);
  }
  return float(dot / std::sqrt(na * nb + 1e-12));
}

// Exact formula key: sorted (atomic_num, count) pairs including total H.
std::string formulaKey(const RDKit::ROMol& mol) {
  std::map<int, int> counts;
  int totalH = 0;
  for (const auto& a : mol.atoms()) {
    counts[a->getAtomicNum()]++;
    totalH += a->getTotalNumHs(false);
  }
  counts[1] += totalH;
  std::ostringstream key;
  for (auto& kv : counts) key << kv.first << ":" << kv.second << ";";
  return key.str();
}

RDKit::RWMol* safeParse(const std::string& smi) {
  try { return RDKit::SmilesToMol(smi); } catch (...) { return nullptr; }
}

std::vector<std::string> splitCandidates(const std::string& cell) {
  std::vector<std::string> out;
  size_t start = 0;
  while (start <= cell.size()) {
    size_t sep = cell.find(';', start);
    if (sep == std::string::npos) { out.push_back(cell.substr(start)); break; }
    out.push_back(cell.substr(start, sep - start));
    start = sep + 1;
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : ".";
  std::string qryStem = argc > 2 ? argv[2] : "test";
  std::string mlpPath = argc > 3 ? argv[3] : "fwd_mlp_rank.bin";
  std::string inPath = argc > 4 ? argv[4] : "submission.csv";
  std::string outPath = argc > 5 ? argv[5] : "submission_reranked_nn.csv";

  Store qry = loadSpectra(dir, qryStem);
  std::unordered_map<std::string, int> nameToMol;
  for (size_t i = 0; i < qry.name.size(); ++i) nameToMol[qry.name[i]] = int(i);
  std::unordered_map<int, std::vector<int>> byMol;
  for (size_t i = 0; i < qry.head.size(); ++i) byMol[qry.head[i].mol].push_back(int(i));

  std::ifstream in(dir + "/" + inPath);
  if (!in) { std::fprintf(stderr, "cannot read %s\n", inPath.c_str()); return 1; }
  std::string line;
  std::getline(in, line);  // header
  std::vector<std::pair<std::string, std::string>> rows;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    size_t comma = line.find(',');
    if (comma == std::string::npos) continue;
    rows.emplace_back(line.substr(0, comma), line.substr(comma + 1));
  }
  std::printf("candidate list: %zu molecules\n", rows.size());

  RDKit::PeriodicTable::getTable();
  forward_model_nn::MLP mlp = forward_model_nn::MLP::load(dir + "/" + mlpPath);
  std::printf("loaded %s (inDim=%d hidden=%d)\n", mlpPath.c_str(), mlp.inDim, mlp.hidden);

  std::vector<std::string> outRows(rows.size());
  std::atomic<size_t> next(0);
  std::atomic<int> nGroupsReranked(0), nMoleculesTouched(0);
  std::mutex printMutex;

  unsigned nthread = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < nthread; ++t) {
    pool.emplace_back([&]() {
      for (;;) {
        size_t ri = next++;
        if (ri >= rows.size()) break;
        const std::string& molName = rows[ri].first;
        std::vector<std::string> cands = splitCandidates(rows[ri].second);

        auto itMol = nameToMol.find(molName);
        if (itMol == nameToMol.end() || cands.size() < 2) {
          outRows[ri] = molName + "," + rows[ri].second;
          continue;
        }
        const auto& spectra = byMol[itMol->second];

        std::vector<std::unique_ptr<RDKit::RWMol>> mols(cands.size());
        std::unordered_map<std::string, std::vector<size_t>> groups;
        for (size_t c = 0; c < cands.size(); ++c) {
          if (cands[c].empty()) continue;
          mols[c].reset(safeParse(cands[c]));
          if (!mols[c]) continue;
          groups[formulaKey(*mols[c])].push_back(c);
        }

        bool touched = false;
        for (auto& kv : groups) {
          if (kv.second.size() < 2) continue;
          touched = true;
          ++nGroupsReranked;
          std::vector<std::pair<float, size_t>> scored;
          for (size_t c : kv.second) {
            auto frags = forward_model_nn::enumerateFragments(*mols[c]);
            float best = 0.0f;
            for (int qi : spectra) {
              const Head& q = qry.head[qi];
              forward_model_nn::Context ctx;
              if (q.tag >= 0) { ctx.adductIdx = q.tag / 16; ctx.instrumentIdx = q.tag % 16; }
              auto pred = forward_model_nn::scoreFragmentsWithContext(frags, mlp, ctx);
              best = std::max(best, cosine(pred, qry.mzs(q), qry.its(q), q.n, 0.01f));
            }
            scored.emplace_back(best, c);
          }
          std::sort(scored.begin(), scored.end(), [](auto& a, auto& b) { return a.first > b.first; });
          std::vector<std::string> original;
          original.reserve(kv.second.size());
          for (size_t pos : kv.second) original.push_back(cands[pos]);
          std::unordered_map<size_t, size_t> origIndexOf;
          for (size_t k = 0; k < kv.second.size(); ++k) origIndexOf[kv.second[k]] = k;
          for (size_t k = 0; k < kv.second.size(); ++k)
            cands[kv.second[k]] = original[origIndexOf[scored[k].second]];
        }
        if (touched) ++nMoleculesTouched;

        std::string cell;
        for (size_t c = 0; c < cands.size(); ++c) {
          if (c) cell += ';';
          cell += cands[c];
        }
        outRows[ri] = molName + "," + cell;

        if (touched) {
          std::lock_guard<std::mutex> lk(printMutex);
          std::printf("  [%s] reranked %zu same-formula groups\n", molName.c_str(), groups.size());
        }
      }
    });
  }
  for (auto& th : pool) th.join();

  std::ofstream out(dir + "/" + outPath);
  out << "molecule_id,smiles\n";
  for (auto& r : outRows) out << r << "\n";
  std::printf("\nwrote %s: %d/%zu molecules had a same-formula group reranked (%d groups total)\n",
              outPath.c_str(), nMoleculesTouched.load(), rows.size(), nGroupsReranked.load());
  return 0;
}
