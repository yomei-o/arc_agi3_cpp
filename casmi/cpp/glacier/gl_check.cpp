// gl_check.cpp: stage-by-stage comparison of the own C++ GLACIER inference (gl_feat.hpp + gl_model.hpp) with the
// Python reference dumps of gl_ref.py.
//   gl_check <glacier_w.bin> <ref dir> <n cases>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "gl_feat.hpp"

using namespace gl;

static double maxdiff(const std::vector<float>& a, const std::vector<float>& b, size_t* where = nullptr) {
  if (a.size() != b.size()) return 1e30;
  double m = 0;
  for (size_t i = 0; i < a.size(); i++) {
    double d = std::fabs((double)a[i] - (double)b[i]);
    if (d > m) {
      m = d;
      if (where) *where = i;
    }
  }
  return m;
}
static std::vector<float> asf(const std::vector<int>& v) { return std::vector<float>(v.begin(), v.end()); }

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: gl_check glacier_w.bin refdir ncases\n");
    return 1;
  }
  auto t0 = std::chrono::steady_clock::now();
  Model M(argv[1]);
  double tl = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("model: %zu tensors, graphormer %zu, slot dec %zu, inten dec %zu, inten enc %zu (load %.1fs)\n",
              M.W.size(), M.gr.size(), M.slot_dec.size(), M.inten_dec.size(), M.inten_enc.size(), tl);
  int nc = std::atoi(argv[3]);
  for (int k = 0; k < nc; k++) {
    std::string base = std::string(argv[2]) + "/ref_" + std::to_string(k);
    std::ifstream tf(base + ".txt");
    std::string line;
    if (!std::getline(tf, line)) continue;
    std::stringstream ss(line);
    std::string smi, ad, ins, ces;
    std::getline(ss, smi, '\t');
    std::getline(ss, ad, '\t');
    std::getline(ss, ins, '\t');
    std::getline(ss, ces, '\t');
    auto R = glw_load(base + ".bin");
    std::string c = canon(smi);
    MolFeat F;
    if (!featurize(c, F)) {
      std::printf("case %d: featurize failed (%s)\n", k, smi.c_str());
      continue;
    }
    std::printf("case %d: %s  canon %s %s  n %d\n", k, smi.c_str(), c.c_str(), c == smi ? "(same)" : "(DIFFERS)", F.n);
    size_t w = 0;
    double d = maxdiff(F.x, R["in_x"].v, &w);
    std::printf("  x          maxdiff %.3g (at row %zu col %zu)\n", d, w / F.nfeat, w % F.nfeat);
    std::printf("  degree     maxdiff %.3g\n", maxdiff(asf(F.degree), R["in_degree"].v));
    std::printf("  spatial    maxdiff %.3g\n", maxdiff(asf(F.spatial), R["in_spatial_pos"].v));
    std::printf("  edge_input maxdiff %.3g\n", maxdiff(F.edge_input, R["in_edge_input"].v));
    std::printf("  adj        maxdiff %.3g\n", maxdiff(F.adj_bo, R["in_adj_matrices"].v));
    std::printf("  masses     maxdiff %.3g  atom_hs %.3g  total_hs %d vs %g\n", maxdiff(F.masses, R["in_masses"].v),
                maxdiff(asf(F.atom_hs), R["in_atom_hs"].v), F.total_hs, R["in_total_hs"].v[0]);
    std::printf("  root_form  maxdiff %.3g  atom_form %.3g\n", maxdiff(F.root_form, R["in_root_form_vecs"].v),
                maxdiff(F.atom_form, R["in_atom_form_vecs"].v));
    Cond C;
    if (!cond_for(ad, ins, (float)std::atof(ces.c_str()), C)) {
      std::printf("  unknown adduct %s\n", ad.c_str());
      continue;
    }
    std::printf("  cond adduct %d (ref %g) instr %d (ref %g) ce %g (ref %g) shifts %.9g %.9g (ref %.9g %.9g)\n", C.adduct,
                R["in_adducts"].v[0], C.instrument, R["in_instruments"].v[0], C.ce, R["in_collision_engs"].v[0],
                C.adduct_mass, C.electron, R["in_adduct_mass_shifts"].v[0], R["in_adduct_mass_shifts"].v[1]);
    Trace tr;
    auto t1 = std::chrono::steady_clock::now();
    std::vector<float> out = predict(M, F, C, &tr);
    double tp = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    std::printf("  node_emb   maxdiff %.3g   root_tok %.3g\n", maxdiff(tr.node_emb, R["node_emb"].v),
                maxdiff(tr.root_tok, R["root_tok"].v));
    std::printf("  frag_logit maxdiff %.3g   card %.3g\n", maxdiff(tr.frag_logits, R["frag_logits"].v),
                maxdiff(tr.card_logits, R["card_logits"].v));
    // breakpoints
    const auto& rb = R["breakpoints"].v;
    int bdiff = 0;
    for (int a = 0; a < NSLOT; a++) {
      std::vector<int> b;
      for (int i = 0; i < F.n; i++)
        if (rb[(size_t)a * F.n + i] > 0.5f) b.push_back(i);
      if (b != tr.breakpoints[a]) bdiff++;
    }
    const auto& rf = R["fragments"];
    int nfr = (int)rf.d(0), fdiff = 0;
    if (nfr == (int)tr.fragments.size()) {
      for (int f = 0; f < nfr; f++)
        for (int i = 0; i < F.n; i++)
          if ((rf.v[(size_t)f * F.n + i] > 0.5f) != (tr.fragments[f][i] != 0)) {
            fdiff++;
            break;
          }
    }
    std::printf("  breakpoints: %d / %d slots differ;  fragments %zu vs ref %d, %d differ\n", bdiff, NSLOT,
                tr.fragments.size(), nfr, fdiff);
    const auto& ro = R["full_out"].v;
    if (ro.size() == out.size()) {
      std::vector<float> om, oi, rm, ri;
      for (size_t r = 0; r < out.size(); r += 2) {
        om.push_back(out[r]);
        oi.push_back(out[r + 1]);
        rm.push_back(ro[r]);
        ri.push_back(ro[r + 1]);
      }
      std::printf("  output rows %zu: mass maxdiff %.3g  inten maxdiff %.3g\n", out.size() / 2, maxdiff(om, rm),
                  maxdiff(oi, ri));
    } else {
      std::printf("  output rows %zu vs ref %zu\n", out.size() / 2, ro.size() / 2);
    }
    std::vector<double> pm, pi;
    post(out, pm, pi);
    const auto& rpm = R["post_mz"].v;
    const auto& rpi = R["post_it"].v;
    double dm = pm.size() == rpm.size() ? 0 : 1e30, di = dm;
    if (dm == 0)
      for (size_t i = 0; i < pm.size(); i++) {
        dm = std::max(dm, std::fabs(pm[i] - rpm[i]));
        di = std::max(di, std::fabs(pi[i] - rpi[i]));
      }
    std::printf("  post %zu peaks (ref %zu): mz maxdiff %.3g  it maxdiff %.3g   predict %.3fs\n", pm.size(), rpm.size(),
                dm, di, tp);
  }
  return 0;
}
