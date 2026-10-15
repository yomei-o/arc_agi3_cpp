# mkpclm.py <phys_pc.cpp> <phys_pclm.cpp>: the phys_lm patches applied to phys_pc
import sys
src, p = sys.argv[1], sys.argv[2]
s = open(src, encoding='utf-8').read().replace('\r\n', '\n')
BS = chr(92)


def rep(a, b):
    global s
    a = a.replace('@N', BS + 'n')
    b = b.replace('@N', BS + 'n')
    assert s.count(a) == 1, (a, s.count(a))
    s = s.replace(a, b)


rep('#include "gen_ops.h"\n', '#include "gen_ops.h"\n#include "lm_feat.h"\n#include "lmart_model.h"\n')
rep('  std::string det;                       // generated-candidate detail lines\n',
    '  std::string det;                       // generated-candidate detail lines\n'
    '  std::vector<float> merr;               // ppm distance of the candidate mass to the nearest window centre\n'
    '  lmf::QMeta meta;                       // query-level features for the ranker\n')
rep('''        std::vector<std::pair<double, double>> win;
''', '''        std::vector<std::pair<double, double>> win;
        {
          const Head& h0 = head[q.spectra[0]];
          int a0 = adductIdx(h0.tag);
          float sh0 = (a0 >= 0 && a0 < 10) ? ADDUCT_SHIFT[a0] : 1.007276f;
          q.meta.nspec = int(q.spectra.size());
          q.meta.mass = (a0 == 8 || a0 == 9) ? 0.5f * (h0.precursor - sh0) : h0.precursor - sh0;
          q.meta.pol = (a0 >= 0 && a0 < 10) ? int(ADDUCT_NEG[a0]) : -1;
          q.meta.adduct = a0;
          q.meta.instr = (h0.tag < 0 || h0.tag % 16 == 15) ? -1 : h0.tag % 16;
        }
''')
rep('''          q.candN.push_back(int(m->getNumAtoms()));
''', '''          q.candN.push_back(int(m->getNumAtoms()));
          {
            double cm = gen::molMass(*m), best = 1e9;
            for (auto& w : win) {
              double c0 = 0.5 * (w.first + w.second);
              best = std::min(best, std::fabs(cm - c0) / std::max(1.0, c0) * 1e6);
            }
            q.merr.push_back(float(best));
          }
''')
rep('''    std::ofstream ff(dir + "/" + featPath);
''', '''    std::ofstream ff(dir + "/" + featPath);
    ff.precision(9);  // full float precision: lmart rebuilds the exact features phys_lm computes
''')
rep('''<< " " << q.gF[i][5] << "@N";
    }''', '''<< " " << q.gF[i][5]
           // phys_lm extras: mass error, other analog variants, query meta
           << " " << q.merr[i] << " " << (haveLib ? q.anaV[1][1][i] : 0.f) << " " << (haveLib ? q.anaV[2][1][i] : 0.f)
           << " " << (haveLib ? q.anaV[4][0][i] : 0.f) << " " << (haveLib ? q.anaV[4][2][i] : 0.f)
           << " " << (haveLib ? q.anaV[6][1][i] : 0.f) << " " << q.meta.nspec << " " << q.meta.mass << " "
           << q.meta.pol << " " << q.meta.adduct << " " << q.meta.instr << "@N";
    }''')
LM = '''  // phys_lm: [lmart model] (argv[25], relative to dir) replaces the output
  // order with the LambdaMART ranker over lm_feat.h features (raw scores, so
  // independent of the hand fusion above); FPNet, if given, is added after.
  // (phys_pclm: placed after scoresNoFp, so the PubChem score-mode insertion
  // still compares the picks with the HAND-fused scores of the main list,
  // now taken in the ranker's order.)
  std::string lmPath = argc > 25 ? argv[25] : "";
  if (!lmPath.empty()) {
    lmart::Model lm;
    if (!lm.load(dir + "/" + lmPath, lmf::names())) { std::fprintf(stderr, "cannot load lmart model %s@N", lmPath.c_str()); return 1; }
    double tot = 0, totAvg = 0;
    std::vector<float> X;
    for (size_t qi = 0; qi < Q.size(); ++qi) {
      const Query& q = Q[qi];
      std::vector<lmf::Row> R(q.cand.size());
      for (size_t i = 0; i < q.cand.size(); ++i) {
        lmf::Row& r = R[i];
        r[lmf::PHYS] = physRaw[qi][i];
        r[lmf::MLP] = haveMlp ? q.mlpCos[i] : 0.f;
        r[lmf::REV] = haveRev ? q.revLL[i] : 0.f;
        r[lmf::LIB] = haveLib ? q.libSim[i] : 0.f;
        r[lmf::ANA] = haveLib ? q.anaP[i] : 0.f;
        for (int k = 0; k < NG; ++k) r[lmf::G0 + k] = q.gF[i][k];
        r[lmf::MERR] = q.merr[i];
        r[lmf::ACOS] = haveLib ? q.anaV[1][1][i] : 0.f;
        r[lmf::AMCOS] = haveLib ? q.anaV[2][1][i] : 0.f;
        r[lmf::AM6P2] = haveLib ? q.anaV[4][0][i] : 0.f;
        r[lmf::AM6P4] = haveLib ? q.anaV[4][2][i] : 0.f;
        r[lmf::ANP6] = haveLib ? q.anaV[6][1][i] : 0.f;
      }
      lmf::build(R, q.meta, X);
      const int F = lmf::nf();
      std::vector<float> sc(q.cand.size());
      for (size_t i = 0; i < sc.size(); ++i) sc[i] = lm.score(&X[i * F]);
      if (q.truth >= 0) {
        int better = 0, eq = 0;
        for (size_t i = 0; i < sc.size(); ++i)
          if (int(i) != q.truth) { better += sc[i] > sc[q.truth]; eq += sc[i] == sc[q.truth]; }
        if (better + eq < 25) tot += 1.0 / (better + eq + 1);
        if (better < 25) totAvg += 1.0 / (better + 1 + 0.5 * eq);
      }
      scores[qi] = std::move(sc);
    }
    std::printf("output uses lmart %s (%zu trees): MRR@25 %.4f (ties averaged %.4f)@N", lmPath.c_str(), lm.trees.size(),
                tot / Q.size(), totAvg / Q.size());
  }
'''
anchor = '  const std::vector<std::vector<float>> scoresNoFp = scores;\n'
if anchor in s:
    rep(anchor, anchor + LM)
else:
    rep('  if (!fpPath.empty() && fuseF != 0.f) {', LM + '  if (!fpPath.empty() && fuseF != 0.f) {')
rep('''  double m = evaluate(P, &scores);
''', '''  double m = evaluate(P, &scores);
  const std::vector<std::vector<float>> physRaw = scores;  // the fusion below overwrites scores
''')
s = ('''// phys_pclm / phys_lm: + LambdaMART ranker (lmart.cpp, lm_feat.h, lmart_model.h).
//   argv[25] = lmart model file (relative to dir): the output order comes from
//   the ranker instead of the hand / learned linear fusion. The features dump
//   ([features.txt]) gains 11 columns after the 6 generation columns:
//   merr aCos aMcos aM6p2 aM6p4 aNP6 nspec mass pol adduct instr.
//
''') + s
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('ok', p)
