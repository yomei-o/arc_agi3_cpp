// fwd2_enum.cpp: how much of real timsTOF [M+H]+ spectra can a deterministic fragment enumeration explain?
// (upper bound for an own learned-intensity forward model "fwd2" trained on train.parquet: fixed fragments, learned
// intensities, ICEBERG-IntenGNN-like network). Fragments = ICEBERG's generation space without the learned selection:
// repeated single heavy-atom removal (depth <= D), connected components with >= 2 atoms, exact atom-set dedup;
// peaks = fragment mass + (h - 6) H (h allowed by broken bonds / fragment H, as ICEBERG) + adduct shift (both the
// [M+H]+ shift and the electron shift, as ICEBERG's 2 x 13 outputs).
//   fwd2_enum <dir> <lib stem> <n spectra> [max depth=3] [cap=20000] [tol_da=0.01]
#include <chrono>
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <unordered_set>

#include "ice_model.hpp"

struct Head {
  float prec;
  int32_t mol, off, n, tag;
};

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: fwd2_enum dir lib_stem n [depth=3] [cap=20000] [tol=0.01]\n");
    return 1;
  }
  std::string dir = argv[1], stem = argv[2];
  int NS = std::atoi(argv[3]), DEPTH = argc > 4 ? std::atoi(argv[4]) : 3, CAP = argc > 5 ? std::atoi(argv[5]) : 20000;
  double TOL = argc > 6 ? std::atof(argv[6]) : 0.01;
  std::vector<Head> head;
  {
    std::ifstream f(dir + "/" + stem + "_head.bin", std::ios::binary | std::ios::ate);
    size_t n = (size_t)f.tellg();
    f.seekg(0);
    head.resize(n / sizeof(Head));
    f.read((char*)head.data(), n);
  }
  std::vector<float> peaks;
  {
    std::ifstream f(dir + "/" + stem + "_peaks.bin", std::ios::binary | std::ios::ate);
    size_t n = (size_t)f.tellg();
    f.seekg(0);
    peaks.resize(n / 4);
    f.read((char*)peaks.data(), n);
  }
  std::vector<size_t> off(head.size());
  size_t cur = 0;
  for (size_t i = 0; i < head.size(); i++) {
    off[i] = cur;
    cur += 2 * (size_t)head[i].n;
  }
  std::vector<std::string> smiles;
  {
    std::ifstream f(dir + "/" + stem + "_mols.txt");
    std::string l;
    while (std::getline(f, l)) {
      if (!l.empty() && l.back() == '\r') l.pop_back();
      size_t t = l.find('\t');
      smiles.push_back(t == std::string::npos ? "" : l.substr(t + 1));
    }
  }
  std::vector<size_t> cand;
  for (size_t i = 0; i < head.size(); i++)
    if (head[i].tag == 0) cand.push_back(i);  // [M+H]+ x timsTOF
  std::printf("%s: %zu spectra, %zu timsTOF [M+H]+\n", stem.c_str(), head.size(), cand.size());
  float CHEM[18];
  for (int e = 0; e < 18; e++) CHEM[e] = (float)gl::ELEMENT_MASS[e];
  const double HM = gl::ELEMENT_MASS[7], PROTON = 1.007276452, ELEC = 0.00054858;
  // per depth: covered intensity share, fragment counts
  std::vector<double> cov(DEPTH + 1, 0), covpk(DEPTH + 1, 0), nfr(DEPTH + 1, 0), ctrl(DEPTH + 1, 0), nctrl(DEPTH + 1, 0);
  std::vector<std::vector<double>> prevpm(DEPTH + 1);
  double cov_noH = 0;
  int done = 0, capped = 0;
  auto t0 = std::chrono::steady_clock::now();
  size_t stride = std::max<size_t>(1, cand.size() / std::max(1, NS));
  for (size_t ci = 0; ci < cand.size() && done < NS; ci += stride) {
    const Head& h = head[cand[ci]];
    if (h.mol < 0 || (size_t)h.mol >= smiles.size()) continue;
    ice::Root R;
    if (!ice::root_from_smiles(smiles[h.mol], R) || R.n < 3) continue;
    // observed peaks (relative >= 1 %, below the precursor + 1)
    const float* mz = &peaks[off[cand[ci]]];
    const float* it = mz + h.n;
    float mx = 0;
    for (int k = 0; k < h.n; k++) mx = std::max(mx, it[k]);
    std::vector<std::pair<double, double>> obs;
    double tot = 0;
    for (int k = 0; k < h.n; k++)
      if (it[k] >= 0.01f * mx && mz[k] <= h.prec + 1.0) {
        obs.push_back({mz[k], it[k]});
        tot += it[k];
      }
    if (obs.empty()) continue;
    // enumerate
    struct F {
      std::vector<int> atoms;
      float broken;
      int depth;
    };
    std::vector<F> frs;
    std::set<std::vector<int>> seen;
    std::vector<int> all(R.n);
    std::iota(all.begin(), all.end(), 0);
    frs.push_back({all, 0.f, 0});
    seen.insert(all);
    bool cap = false;
    for (size_t q = 0; q < frs.size() && !cap; q++) {
      if (frs[q].depth >= DEPTH) continue;
      F P = frs[q];
      std::vector<char> in(R.n, 0);
      for (int a : P.atoms) in[a] = 1;
      for (int rm : P.atoms) {
        in[rm] = 0;
        std::vector<int> comp(R.n, -1);
        for (int a : P.atoms) {
          if (!in[a] || comp[a] >= 0) continue;
          std::vector<int> st{a}, mem;
          comp[a] = a;
          while (!st.empty()) {
            int u = st.back();
            st.pop_back();
            mem.push_back(u);
            for (auto [v, t] : R.nb[u])
              if (in[v] && comp[v] < 0) {
                comp[v] = a;
                st.push_back(v);
              }
          }
          if (mem.size() < 2) continue;
          std::sort(mem.begin(), mem.end());
          if (seen.count(mem)) continue;
          float br = 0;
          for (auto [v, t] : R.nb[rm])
            if (in[v] && comp[v] == a) br += (float)t;
          seen.insert(mem);
          frs.push_back({mem, P.broken + br, P.depth + 1});
          if ((int)frs.size() >= CAP) {
            cap = true;
            break;
          }
        }
        in[rm] = 1;
        if (cap) break;
      }
    }
    capped += cap;
    // peaks by max depth, coverage
    for (int D = 0; D <= DEPTH; D++) {
      std::vector<double> pm;
      int nf = 0;
      for (const F& f : frs) {
        if (f.depth > D) continue;
        nf++;
        ice::Frag tmp;
        ice::frag_formula(R, f.atoms, CHEM, tmp);
        float nh = tmp.form[7];
        int maxadd = (int)std::min(f.broken, 6.f), maxrem = (int)std::min(std::min(f.broken, nh), 6.f);
        for (int hh = -maxrem; hh <= maxadd; hh++) {
          pm.push_back(tmp.mass + hh * HM + PROTON);
          pm.push_back(tmp.mass + hh * HM - ELEC);
        }
      }
      std::sort(pm.begin(), pm.end());
      double c = 0, cp = 0;
      for (auto& [m, v] : obs) {
        auto itp = std::lower_bound(pm.begin(), pm.end(), m - TOL);
        if (itp != pm.end() && *itp <= m + TOL) {
          c += v;
          cp += 1;
        }
      }
      cov[D] += c / tot;
      covpk[D] += cp / obs.size();
      nfr[D] += nf;
      // control: the same spectrum against the previous molecule's fragment peaks (chance matches)
      if (!prevpm[D].empty()) {
        double cc = 0;
        for (auto& [m, v] : obs) {
          auto itp = std::lower_bound(prevpm[D].begin(), prevpm[D].end(), m - TOL);
          if (itp != prevpm[D].end() && *itp <= m + TOL) cc += v;
        }
        ctrl[D] += cc / tot;
        nctrl[D]++;
      }
      prevpm[D] = pm;
    }
    {  // no H shifts at all (pure fragment + proton)
      std::vector<double> pm;
      for (const F& f : frs) {
        ice::Frag tmp;
        ice::frag_formula(R, f.atoms, CHEM, tmp);
        pm.push_back(tmp.mass + PROTON);
      }
      std::sort(pm.begin(), pm.end());
      double c = 0;
      for (auto& [m, v] : obs) {
        auto itp = std::lower_bound(pm.begin(), pm.end(), m - TOL);
        if (itp != pm.end() && *itp <= m + TOL) c += v;
      }
      cov_noH += c / tot;
    }
    done++;
  }
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("spectra %d (capped at %d fragments: %d), %.1fs\n", done, CAP, capped, sec);
  for (int D = 0; D <= DEPTH; D++)
    std::printf("depth <= %d: fragments/mol %.0f  explained intensity %.3f (control: another molecule's fragments %.3f)"
                "  explained peaks %.3f\n",
                D, nfr[D] / done, cov[D] / done, ctrl[D] / std::max(1.0, nctrl[D]), covpk[D] / done);
  std::printf("all depths, no H shifts: explained intensity %.3f\n", cov_noH / done);
  return 0;
}
