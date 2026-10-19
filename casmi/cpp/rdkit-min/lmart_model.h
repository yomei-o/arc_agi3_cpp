// lmart_model.h - header-only scorer for lmart.cpp's LambdaMART models.
//
// Model file (text):
//   LMART1
//   features <F> <name_0> ... <name_F-1>     (names the trainer saw; mapped to
//                                           lm_feat.h's order at load time)
//   [opt <key> <value>]...                 (optional, e.g. gate / blend / libfloor)
//   base <feature name or -> <coef>          (score = coef * x[base] + trees)
//   trees <T>
//   then per tree: "tree <nNodes>" and nNodes lines
//     "<feat> <threshold> <left> <right> <value>"  (feat -1 = leaf with value;
//     internal: x[feat] <= threshold goes left)
#pragma once
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace lmart {

struct Node { int feat; float thr; int left, right; float value; };
struct Tree { std::vector<Node> nodes; };

struct Model {
  std::vector<int> fmap;  // model feature k -> index in the caller's feature vector
  int baseFeat = -1;
  float baseCoef = 0.f;
  std::vector<Tree> trees;
  std::map<std::string, double> opts;  // "opt" lines

  // names: the caller's feature order (lmf::names()). Returns false on a
  // missing feature or a malformed file.
  bool load(const std::string& path, const std::vector<std::string>& names) {
    std::ifstream f(path);
    if (!f) return false;
    std::string tag;
    f >> tag;
    if (tag != "LMART1") return false;
    int F = 0;
    f >> tag >> F;
    if (tag != "features") return false;
    std::vector<std::string> mn(F);
    for (auto& s : mn) f >> s;
    auto idx = [&](const std::string& s) {
      for (size_t i = 0; i < names.size(); ++i) if (names[i] == s) return int(i);
      return -1;
    };
    fmap.resize(F);
    for (int k = 0; k < F; ++k) {
      fmap[k] = idx(mn[k]);
      if (fmap[k] < 0) { std::fprintf(stderr, "lmart: feature %s unknown\n", mn[k].c_str()); return false; }
    }
    std::string bname;
    f >> tag;
    while (tag == "opt") {  // optional "opt <key> <value>" lines (lmart_v29: gate, blend, libfloor)
      std::string k;
      double v = 0;
      f >> k >> v;
      opts[k] = v;
      f >> tag;
    }
    f >> bname >> baseCoef;
    if (tag != "base") return false;
    baseFeat = -1;
    if (bname != "-") {
      for (int k = 0; k < F; ++k) if (mn[k] == bname) baseFeat = fmap[k];
      if (baseFeat < 0) baseFeat = idx(bname);
      if (baseFeat < 0) return false;
    }
    int T = 0;
    f >> tag >> T;
    if (tag != "trees") return false;
    trees.assign(T, {});
    for (auto& t : trees) {
      int nn = 0;
      f >> tag >> nn;
      if (tag != "tree" || nn <= 0) return false;
      t.nodes.resize(nn);
      for (auto& nd : t.nodes) {
        f >> nd.feat >> nd.thr >> nd.left >> nd.right >> nd.value;
        if (nd.feat >= 0) {
          if (nd.feat >= F) return false;
          nd.feat = fmap[nd.feat];
        }
      }
    }
    return bool(f);
  }
  float score(const float* x) const {
    double s = baseFeat >= 0 ? double(baseCoef) * x[baseFeat] : 0.0;
    for (const Tree& t : trees) {
      int k = 0;
      while (t.nodes[k].feat >= 0) k = x[t.nodes[k].feat] <= t.nodes[k].thr ? t.nodes[k].left : t.nodes[k].right;
      s += t.nodes[k].value;
    }
    return float(s);
  }
};

}  // namespace lmart
