// Cascade stage 2: re-score cand_rank's shortlist (reverse model + popularity
// over whole PubChem + pool) with the forward-model MLP, and fuse.
//
// Why a cascade: measured leak-free on the denovo holdout, the forward MLP
// ranks far better than the reverse model (MRR@25 0.675 vs 0.471 on the same
// pool) - and the real LB agreed (v10 forward 0.181 > v12 reverse 0.154).
// But the forward model enumerates fragments per candidate, too slow for
// whole-PubChem windows; the reverse model is a dot product. So the reverse
// model + popularity pick a few hundred per molecule, the forward model
// decides among them.
//
// fused = a*z(fwd cosine) + b*z(rev loglik) + c*popularity, z within molecule.
//   fwd_rescore eval <dir> <qry> <mlp.bin> <cands.csv> [truth from <qry>_ecfp.bin]
//       -> prints MRR@25 for a grid of (a, b, c)
//   fwd_rescore rank <dir> <qry> <mlp.bin> <cands.csv> <out.csv> <a> <b> <c>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <GraphMol/GraphMol.h>
#include <GraphMol/PeriodicTable.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include "../rev/ecfp.h"
#include "forward_model_nn.h"

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

std::vector<std::string> split(const std::string& s, char d) {
  std::vector<std::string> out;
  size_t a = 0;
  for (;;) {
    size_t b = s.find(d, a);
    out.push_back(s.substr(a, b == std::string::npos ? std::string::npos : b - a));
    if (b == std::string::npos) break;
    a = b + 1;
  }
  return out;
}

RDKit::RWMol* safeParse(const std::string& smi) {
  try { return RDKit::SmilesToMol(smi); } catch (...) { return nullptr; }
}

float cosine(const std::vector<std::pair<double, double>>& pred, const float* mz, const float* it, int n, float tol) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (auto& pk : pred) na += pk.second;
  for (int i = 0; i < n; ++i) nb += double(it[i]);
  if (na <= 0.0 || nb <= 0.0) return 0.0f;
  int j = 0;
  for (auto& pk : pred) {
    while (j < n && double(mz[j]) < pk.first - tol) ++j;
    double best = 0.0;
    for (int k = j; k < n && double(mz[k]) <= pk.first + tol; ++k) best = std::max(best, double(it[k]));
    if (best > 0.0) dot += std::sqrt(pk.second) * std::sqrt(best);
  }
  return float(dot / std::sqrt(na * nb + 1e-12));
}

uint64_t hashBits(const std::vector<uint16_t>& b) {
  uint64_t h = 1469598103934665603ull;
  for (uint16_t x : b) h = (h ^ x) * 1099511628211ull;
  return h;
}

struct Row { std::string id; std::vector<std::string> smi; std::vector<float> rev, pop; };
struct Scored { std::vector<float> fwd, rev, pop; std::vector<uint64_t> fp; };

std::vector<float> zscore(const std::vector<float>& v) {
  if (v.empty()) return v;
  double m = 0, s = 0;
  for (float x : v) m += x;
  m /= v.size();
  for (float x : v) s += (x - m) * (x - m);
  s = std::sqrt(s / v.size()) + 1e-6;
  std::vector<float> z(v.size());
  for (size_t i = 0; i < v.size(); ++i) z[i] = float((v[i] - m) / s);
  return z;
}

std::vector<size_t> fusedOrder(const Scored& s, float a, float b, float c) {
  auto zf = zscore(s.fwd), zr = zscore(s.rev);
  std::vector<float> f(s.fwd.size());
  for (size_t i = 0; i < f.size(); ++i) f[i] = a * zf[i] + b * zr[i] + c * s.pop[i];
  std::vector<size_t> o(f.size());
  for (size_t i = 0; i < o.size(); ++i) o[i] = i;
  std::stable_sort(o.begin(), o.end(), [&](size_t x, size_t y) { return f[x] > f[y]; });
  return o;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6) { std::fprintf(stderr, "usage: fwd_rescore eval|rank <dir> <qry> <mlp> <cands.csv> ...\n"); return 1; }
  std::string mode = argv[1], dir = argv[2], qs = argv[3], mlpPath = argv[4], candPath = argv[5];

  // query spectra
  std::vector<Head> head;
  std::vector<float> mz, it;
  std::vector<std::string> names;
  {
    std::vector<char> hb = slurp(dir + "/" + qs + "_head.bin");
    head.resize(hb.size() / sizeof(Head));
    std::memcpy(head.data(), hb.data(), hb.size());
    std::vector<char> pb = slurp(dir + "/" + qs + "_peaks.bin");
    const float* p = reinterpret_cast<const float*>(pb.data());
    for (Head& h : head) {
      h.off = int32_t(mz.size());
      mz.insert(mz.end(), p, p + h.n); p += h.n;
      it.insert(it.end(), p, p + h.n); p += h.n;
    }
    std::ifstream f(dir + "/" + qs + "_mols.txt");
    std::string line;
    while (std::getline(f, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      names.push_back(line.substr(0, line.find('\t')));
    }
  }
  std::unordered_map<std::string, std::vector<int>> specOf;
  for (size_t s = 0; s < head.size(); ++s) specOf[names[head[s].mol]].push_back(int(s));

  std::vector<Row> rows;
  {
    std::ifstream f(dir + "/" + candPath);
    std::string line;
    std::getline(f, line);
    while (std::getline(f, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      auto c = split(line, ',');
      if (c.size() < 4) continue;
      Row r; r.id = c[0];
      if (!c[1].empty()) {
        r.smi = split(c[1], ';');
        for (auto& x : split(c[2], ';')) r.rev.push_back(float(std::atof(x.c_str())));
        for (auto& x : split(c[3], ';')) r.pop.push_back(float(std::atof(x.c_str())));
      }
      rows.push_back(std::move(r));
    }
  }
  std::printf("%zu molecules, %zu query spectra\n", rows.size(), head.size());

  RDKit::PeriodicTable::getTable();
  forward_model_nn::MLP mlp = forward_model_nn::MLP::load(dir + "/" + mlpPath);
  std::vector<Scored> sc(rows.size());
  std::atomic<size_t> next(0);
  unsigned nt = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> th;
  for (unsigned t = 0; t < nt; ++t)
    th.emplace_back([&]() {
      for (;;) {
        size_t ri = next++;
        if (ri >= rows.size()) break;
        const Row& r = rows[ri];
        Scored& s = sc[ri];
        const auto& specs = specOf[r.id];
        for (size_t k = 0; k < r.smi.size(); ++k) {
          std::unique_ptr<RDKit::RWMol> m(safeParse(r.smi[k]));
          float best = 0.f;
          uint64_t fp = 0;
          if (m) {
            fp = hashBits(ecfp::compute(*m, 2, 2048));
            auto frags = forward_model_nn::enumerateFragments(*m);
            for (int qi : specs) {
              const Head& q = head[qi];
              forward_model_nn::Context ctx;
              int a = q.tag < 0 ? -1 : q.tag / 16, ins = q.tag < 0 ? -1 : q.tag % 16;
              if (q.tag >= 0 && ins == 15) { a = q.tag / 16 + 1; ins = -1; }  // tag-encoding fix
              ctx.adductIdx = (a >= 0 && a < 10) ? a : -1;
              ctx.instrumentIdx = (ins >= 0 && ins < 5) ? ins : -1;
              auto pred = forward_model_nn::scoreFragmentsWithContext(frags, mlp, ctx);
              best = std::max(best, cosine(pred, mz.data() + q.off, it.data() + q.off, q.n, 0.01f));
            }
          }
          s.fwd.push_back(best);
          s.rev.push_back(k < r.rev.size() ? r.rev[k] : 0.f);
          s.pop.push_back(k < r.pop.size() ? r.pop[k] : 0.f);
          s.fp.push_back(fp);
        }
      }
    });
  for (auto& x : th) x.join();

  if (mode == "eval") {
    // truth fingerprints from <qry>_ecfp.bin, by molecule index
    std::vector<char> b = slurp(dir + "/" + qs + "_ecfp.bin");
    const char* p = b.data();
    uint32_t n; std::memcpy(&n, p, 4); p += 8;
    std::unordered_map<std::string, uint64_t> truth;
    for (uint32_t i = 0; i < n; ++i) {
      uint16_t c; std::memcpy(&c, p, 2); p += 2;
      std::vector<uint16_t> bits(c);
      std::memcpy(bits.data(), p, 2 * size_t(c)); p += 2 * size_t(c);
      truth[names[i]] = hashBits(bits);
    }
    int present = 0;
    for (size_t ri = 0; ri < rows.size(); ++ri)
      for (uint64_t f : sc[ri].fp) if (f == truth[rows[ri].id]) { ++present; break; }
    std::printf("truth in shortlist: %d / %zu\n", present, rows.size());
    float A[] = {0, 1}, Bv[] = {0, 0.25f, 0.5f, 1}, C[] = {0, 0.05f, 0.1f, 0.2f, 0.4f};
    for (float a : A) for (float bb : Bv) for (float c : C) {
      if (a == 0 && bb == 0 && c == 0) continue;
      double mrr = 0; int t1 = 0;
      for (size_t ri = 0; ri < rows.size(); ++ri) {
        auto o = fusedOrder(sc[ri], a, bb, c);
        uint64_t t = truth[rows[ri].id];
        for (size_t r = 0; r < o.size() && r < 25; ++r)
          if (sc[ri].fp[o[r]] == t) { mrr += 1.0 / (r + 1); t1 += r == 0; break; }
      }
      std::printf("a=%.2f b=%.2f c=%.2f  top1 %.1f%%  MRR@25 %.3f\n", a, bb, c, 100.0 * t1 / rows.size(), mrr / rows.size());
    }
    return 0;
  }

  float a = float(std::atof(argv[7])), bb = float(std::atof(argv[8])), c = float(std::atof(argv[9]));
  std::ofstream o(dir + "/" + std::string(argv[6]));
  o << "molecule_id,smiles\n";
  for (size_t ri = 0; ri < rows.size(); ++ri) {
    auto ord = fusedOrder(sc[ri], a, bb, c);
    o << rows[ri].id << ',';
    for (size_t r = 0; r < ord.size() && r < 25; ++r) o << (r ? ";" : "") << rows[ri].smi[ord[r]];
    o << '\n';
  }
  std::printf("wrote %s\n", argv[6]);
  return 0;
}
