// ice_check.cpp: own C++ ICEBERG (ice_model.hpp) vs the Python reference dumps of ice_ref.py.
//   ice_check <ice_gen_w.bin> <ice_inten_w.bin> <ref dir> <n cases>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "ice_model.hpp"

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: ice_check gen_w.bin inten_w.bin refdir ncases\n");
    return 1;
  }
  ice::Net G(argv[1], 6, false), I(argv[2], 4, true);
  float CHEM[18];
  for (int e = 0; e < 18; e++) CHEM[e] = (float)gl::ELEMENT_MASS[e];
  std::printf("gen H %d in %d mlp %d; inten H %d in %d trans %zu\n", G.gnn.H, G.gnn.inp.in, G.mlp_in.in, I.gnn.H,
              I.gnn.inp.in, I.trans.size());
  int nc = std::atoi(argv[4]);
  for (int k = 0; k < nc; k++) {
    std::string base = std::string(argv[3]) + "/iref_" + std::to_string(k);
    std::ifstream tf(base + ".txt");
    std::string line, smi, ad, ins, ces;
    if (!std::getline(tf, line)) continue;
    std::stringstream ss(line);
    std::getline(ss, smi, '\t');
    std::getline(ss, ad, '\t');
    std::getline(ss, ins, '\t');
    std::getline(ss, ces, '\t');
    auto Rf = glw_load(base + ".bin");
    ice::Root R;
    if (!ice::root_from_smiles(smi, R)) {
      std::printf("case %d: root failed %s\n", k, smi.c_str());
      continue;
    }
    gl::Cond C;
    gl::cond_for(ad, ins, (float)std::atof(ces.c_str()), C);
    ice::IceTrace tr;
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> spec = ice::predict(G, I, R, C, CHEM, &tr);
    double tp = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double dl = 0;
    const auto& rl = Rf["leave0"].v;
    for (size_t i = 0; i < rl.size() && i < tr.leave0.size(); i++) dl = std::max(dl, (double)std::fabs(rl[i] - tr.leave0[i]));
    // fragments: compare as sets of atom masks
    const auto& rfr = Rf["frags"];
    int nfr = (int)rfr.d(0), n = (int)rfr.d(1);
    std::map<std::vector<int>, int> refset;
    for (int f = 0; f < nfr; f++) {
      std::vector<int> m;
      for (int i = 0; i < n; i++)
        if (rfr.v[(size_t)f * n + i] > 0.5f) m.push_back(i);
      refset[m] = f;
    }
    int match = 0;
    double dp = 0, db = 0, dm = 0;
    for (const auto& F : tr.frags) {
      auto it = refset.find(F.atoms);
      if (it == refset.end()) continue;
      match++;
      int f = it->second;
      dp = std::max(dp, (double)std::fabs(F.prob - Rf["probs"].v[f]));
      db = std::max(db, (double)std::fabs(F.broken - Rf["brokens"].v[f]));
      dm = std::max(dm, (double)std::fabs(F.mass - Rf["masses"].v[f]));
    }
    std::printf("case %d: %s n %d  leave0 maxdiff %.3g | frags %zu vs ref %d, common %d (prob %.3g broken %.3g mass %.3g)\n",
                k, smi.c_str(), R.n, dl, tr.frags.size(), nfr, match, dp, db, dm);
    // raw rows per common fragment (26 = 2 groups x 13 H shifts)
    {
      const auto& rs = Rf["spec"].v;
      double dmass = 0, dint = 0;
      int worst_f = -1, worst_r = -1;
      for (int my = 0; my < (int)tr.frags.size(); my++) {
        auto it = refset.find(tr.frags[my].atoms);
        if (it == refset.end()) continue;
        int f = it->second;
        for (int r = 0; r < 26; r++) {
          double a = spec[((size_t)my * 26 + r) * 2], b = rs[((size_t)f * 26 + r) * 2];
          double c = spec[((size_t)my * 26 + r) * 2 + 1], d = rs[((size_t)f * 26 + r) * 2 + 1];
          dmass = std::max(dmass, std::fabs(a - b));
          if (std::fabs(c - d) > dint) {
            dint = std::fabs(c - d);
            worst_f = my;
            worst_r = r;
          }
        }
      }
      std::printf("   raw rows: mass maxdiff %.3g  inten maxdiff %.3g", dmass, dint);
      if (worst_f >= 0) {
        int f = refset[tr.frags[worst_f].atoms];
        std::printf("  (frag %d row %d: mine %.4g ref %.4g, broken %g, nH %g)", worst_f, worst_r,
                    spec[((size_t)worst_f * 26 + worst_r) * 2 + 1], Rf["spec"].v[((size_t)f * 26 + worst_r) * 2 + 1],
                    tr.frags[worst_f].broken, tr.frags[worst_f].form[7]);
      }
      std::printf("\n");
    }
    // spectrum: compare post() outputs
    std::vector<double> pm, pi;
    gl::post(spec, pm, pi);
    const auto& rpm = Rf["post_mz"].v;
    const auto& rpi = Rf["post_it"].v;
    // match peaks by m/z within 1e-3
    int pk = 0;
    double di = 0;
    for (size_t i = 0; i < pm.size(); i++)
      for (size_t j = 0; j < rpm.size(); j++)
        if (std::fabs(pm[i] - rpm[j]) < 1e-3) {
          pk++;
          di = std::max(di, std::fabs(pi[i] - rpi[j]));
          break;
        }
    std::printf("   time: gnn %.3f mlp %.3f inten %.3f; nodes per depth", tr.t_gnn, tr.t_mlp, tr.t_inten);
    for (int x : tr.nodes_per_depth) std::printf(" %d", x);
    std::printf("\n");
    std::printf("   post peaks %zu vs ref %zu, matched %d, inten maxdiff %.3g   (%.3fs)\n", pm.size(), rpm.size(), pk, di, tp);
  }
  return 0;
}
