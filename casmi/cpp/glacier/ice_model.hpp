// ice_model.hpp: own C++ inference of ICEBERG 2.1 (ms-pred iceberg FragGNN + IntenGNN, Coley group, MIT; MassSpecGym
// checkpoints), written from the Python source (ms_pred/iceberg/{gen_model,inten_model,joint_model,dag_data}.py,
// nn_utils GGNN / dgl_graph_ops, magma FragmentEngine) and checked against ice_ref.py dumps (ice_check.cpp).
//
// Generation (FragGNN, GGNN 6 steps, 512): from the root molecule, 3 rounds of "remove one heavy atom": every frontier
// fragment gets per-atom leaving probabilities; accumulated probabilities (parent x atom) compete with the kept
// fragments for the top 2 x 100 slots; each selected removal yields the connected components (>= 2 atoms) as new
// fragments (broken-bond order summed over the cut bonds); fragments isomorphic by the 1-step message-passing hash
// are kept once (first = best); the 100 most probable fragments survive.
// Intensities (IntenGNN, GGNN 4 steps, 256 + 3 transformer layers over the fragment set): per fragment 2 x 13 peaks
// (adduct shift or electron, -6..+6 H); peaks are pooled per 0.1 Da bucket (linspace(0, 1500, 15000)) by a softmax
// over the "isomer attention" logits, the bucket value is sigmoid(sum w * out) and each peak gets value x its weight.
#pragma once
#include <GraphMol/GraphMol.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include <chrono>

#include "gl_feat.hpp"
#include "ice_gpu.h"

namespace ice {
using namespace gl;

constexpr int NE = 18, NG = 8, MAXH = 6, BASEF = NE + NG + MAXH;  // 32 base node features
constexpr int PEK = 14, NBT = 4, MAXBR = 6, NSH = 13, MAXN = 100;

struct GGNN {
  Lin inp;
  Lin et[NBT];
  const float *wih, *whh, *bih, *bhh;
  int H = 0, steps = 0;
};

struct Net {
  std::map<std::string, GLArr> W;
  const float *adduct_emb, *coll_den, *coll_merged, *instr_emb, *form_feat, *form_extra;
  int adduct_dim, instr_dim;
  GGNN gnn;  // root_module == gnn (same weights)
  Lin mlp_in, mlp_out;  // output_map (gen) / intermediate_out (inten)
  // inten only
  std::vector<EncLayer> trans;
  Lin out_map, attn_out;
  const float* P(const std::string& n) const { return W.at(n).v.data(); }
  Lin lin(const std::string& n) const {
    const GLArr& a = W.at(n + ".weight");
    Lin l;
    l.w = a.v.data();
    l.out = (int)a.d(0);
    l.in = (int)a.d(1);
    auto it = W.find(n + ".bias");
    l.b = it == W.end() ? nullptr : it->second.v.data();
    return l;
  }
  Net(const std::string& path, int steps, bool inten) : W(glw_load(path)) {
    adduct_emb = P("adduct_embedder");
    adduct_dim = (int)W.at("adduct_embedder").d(1);
    coll_den = P("collision_embedder_denominators");
    coll_merged = P("collision_embed_merged");
    instr_emb = P("instrument_embedder");
    instr_dim = (int)W.at("instrument_embedder").d(1);
    form_feat = P("embedder.int_to_feat_matrix");
    form_extra = P("embedder._extra_embeddings");
    gnn.inp = lin("gnn.input_project");
    gnn.H = gnn.inp.out;
    for (int t = 0; t < NBT; t++) gnn.et[t] = lin("gnn.gnn.model.linears." + std::to_string(t));
    gnn.wih = P("gnn.gnn.model.gru.weight_ih");
    gnn.whh = P("gnn.gnn.model.gru.weight_hh");
    gnn.bih = P("gnn.gnn.model.gru.bias_ih");
    gnn.bhh = P("gnn.gnn.model.gru.bias_hh");
    gnn.steps = steps;
    if (!inten) {
      mlp_in = lin("output_map.input_layer");
      mlp_out = lin("output_map.output_layer");
    } else {
      mlp_in = lin("intermediate_out.input_layer");
      mlp_out = lin("intermediate_out.output_layer");
      for (int i = 0; W.count("trans_layers." + std::to_string(i) + ".norm1.weight"); i++) {
        std::string n = "trans_layers." + std::to_string(i);
        EncLayer e;
        const GLArr& a = W.at(n + ".self_attn.in_proj_weight");
        e.sa.in.w = a.v.data();
        e.sa.in.out = (int)a.d(0);
        e.sa.in.in = (int)a.d(1);
        e.sa.in.b = P(n + ".self_attn.in_proj_bias");
        e.sa.out = lin(n + ".self_attn.out_proj");
        e.l1 = lin(n + ".linear1");
        e.l2 = lin(n + ".linear2");
        e.n1 = LN{P(n + ".norm1.weight"), P(n + ".norm1.bias")};
        e.n2 = LN{P(n + ".norm2.weight"), P(n + ".norm2.bias")};
        trans.push_back(e);
      }
      out_map = lin("output_map");
      attn_out = lin("isomer_attn_out");
    }
  }
  void form_embed(const float* counts, float* out) const {
    for (int e = 0; e < NFORM; e++) {
      float c = counts[e];
      const float* row = c >= 255.f ? form_extra : form_feat + (size_t)std::max(0, (int)c) * FDIM;
      std::memcpy(out + e * FDIM, row, FDIM * sizeof(float));
    }
  }
};

// A batch of disjoint graphs: node features [N, in], directed edges (src, dst, type), graph id per node.
struct GBatch {
  int N = 0, G = 0, in = 0;
  std::vector<float> x;
  std::vector<int> src, dst, typ, gid;
  std::vector<int> gstart, gsize;  // node range per graph
};

inline GpuGgnnW gpu_w(const GGNN& g) {
  GpuGgnnW w;
  w.winp = g.inp.w;
  w.binp = g.inp.b;
  for (int t = 0; t < NBT; t++) {
    w.wet[t] = g.et[t].w;
    w.bet[t] = g.et[t].b;
  }
  w.wih = g.wih;
  w.whh = g.whh;
  w.bih = g.bih;
  w.bhh = g.bhh;
  w.in = g.inp.in;
  w.H = g.H;
  w.steps = g.steps;
  return w;
}
inline GpuGraph gpu_graph(const GBatch& B) {
  return GpuGraph{B.N, (int)B.src.size(), B.G, B.x.data(), B.src.data(), B.dst.data(), B.typ.data(),
                  B.gstart.data(), B.gsize.data(), B.gid.data()};
}

// GGNN (input projection + GatedGraphConv(n steps, 4 edge types) with a GRU cell) -> node states [N, H]
inline std::vector<float> ggnn(const GGNN& g, const GBatch& B) {
  const int H = g.H, N = B.N;
  if (GpuGgnnFn f = gpu_ggnn_hook()) {  // optional device-resident path (gpu_ggnn.cu)
    std::vector<float> hd((size_t)N * H);
    if (f(gpu_w(g), gpu_graph(B), hd.data(), nullptr, nullptr)) return hd;
  }
  std::vector<float> h = linear(B.x, N, g.inp);
  std::vector<float> a((size_t)N * H), gi((size_t)N * 3 * H), gh((size_t)N * 3 * H);
  Lin Lih{g.wih, g.bih, H, 3 * H}, Lhh{g.whh, g.bhh, H, 3 * H};
  // node lists per edge type (sources), messages computed per source node once per type
  std::vector<std::vector<int>> srcs(NBT);
  std::vector<std::vector<int>> pos(NBT, std::vector<int>(N, -1));
  for (size_t e = 0; e < B.src.size(); e++) {
    int t = B.typ[e];
    if (pos[t][B.src[e]] < 0) {
      pos[t][B.src[e]] = (int)srcs[t].size();
      srcs[t].push_back(B.src[e]);
    }
  }
  std::vector<float> xs, ms;
  for (int step = 0; step < g.steps; step++) {
    std::fill(a.begin(), a.end(), 0.f);
    for (int t = 0; t < NBT; t++) {
      int ns = (int)srcs[t].size();
      if (!ns) continue;
      xs.resize((size_t)ns * H);
      for (int i = 0; i < ns; i++) std::memcpy(&xs[(size_t)i * H], &h[(size_t)srcs[t][i] * H], H * sizeof(float));
      ms.resize((size_t)ns * H);
      linear(xs.data(), ns, g.et[t], ms.data());
      for (size_t e = 0; e < B.src.size(); e++) {
        if (B.typ[e] != t) continue;
        const float* m = &ms[(size_t)pos[t][B.src[e]] * H];
        float* d = &a[(size_t)B.dst[e] * H];
        for (int k = 0; k < H; k++) d[k] += m[k];
      }
    }
    linear(a.data(), N, Lih, gi.data());
    linear(h.data(), N, Lhh, gh.data());
    for (int i = 0; i < N; i++) {
      const float* xi = &gi[(size_t)i * 3 * H];
      const float* hi = &gh[(size_t)i * 3 * H];
      float* hv = &h[(size_t)i * H];
      for (int k = 0; k < H; k++) {
        float r = 1.f / (1.f + std::exp(-(xi[k] + hi[k])));
        float z = 1.f / (1.f + std::exp(-(xi[H + k] + hi[H + k])));
        float n = std::tanh(xi[2 * H + k] + r * hi[2 * H + k]);
        hv[k] = (1.f - z) * n + z * hv[k];
      }
    }
  }
  return h;
}
inline std::vector<float> avg_pool(const std::vector<float>& h, const GBatch& B, int H) {
  std::vector<float> p((size_t)B.G * H, 0.f);
  for (int g = 0; g < B.G; g++) {
    float* o = &p[(size_t)g * H];
    for (int i = B.gstart[g]; i < B.gstart[g] + B.gsize[g]; i++)
      for (int k = 0; k < H; k++) o[k] += h[(size_t)i * H + k];
    float inv = 1.f / std::max(1, B.gsize[g]);
    for (int k = 0; k < H; k++) o[k] *= inv;
  }
  return p;
}

// ---- root molecule (FragmentEngine on the kekulised root; atom order of the SMILES)
struct Root {
  int n = 0;
  std::vector<std::array<float, BASEF>> feat;  // per atom: element one-hot | group one-hot | H one-hot
  std::vector<int> hs, elem;
  std::vector<std::vector<std::pair<int, int>>> nb;  // (neighbour, bond type 1..3)
  std::vector<float> root_form;                      // CalcMolFormula counts (incl. H)
};
inline bool root_from_smiles(const std::string& smi, Root& R) {
  std::unique_ptr<RDKit::RWMol> m;
  try {
    m.reset(RDKit::SmilesToMol(smi));
    if (!m) return false;
    RDKit::MolOps::Kekulize(*m, true);
  } catch (...) {
    return false;
  }
  R = Root();
  R.n = (int)m->getNumAtoms();
  if (R.n == 0 || R.n > 128) return false;
  R.feat.resize(R.n);
  R.hs.resize(R.n);
  R.elem.resize(R.n);
  R.nb.assign(R.n, {});
  R.root_form.assign(NFORM, 0.f);
  for (int i = 0; i < R.n; i++) {
    const RDKit::Atom* a = m->getAtomWithIdx(i);
    int e = elem_index(a->getAtomicNum());
    if (e < 0) return false;
    int h = (int)a->getTotalNumHs();
    R.elem[i] = e;
    R.hs[i] = h;
    auto& f = R.feat[i];
    f.fill(0.f);
    f[e] = 1.f;
    f[NE + ELEMENT_GROUP[e]] = 1.f;
    if (h < MAXH) f[NE + NG + h] = 1.f;  // torch.eye(6)[h] (h >= 6 would raise in Python)
    R.root_form[e] += 1.f;
    R.root_form[7] += (float)h;
  }
  for (const RDKit::Bond* b : m->bonds()) {
    int i = (int)b->getBeginAtomIdx(), j = (int)b->getEndAtomIdx();
    int t = b->getBondType() == RDKit::Bond::DOUBLE || b->getBondType() == RDKit::Bond::AROMATIC ? 2
            : b->getBondType() == RDKit::Bond::TRIPLE                                         ? 3
                                                                                               : 1;
    R.nb[i].push_back({j, t});
    R.nb[j].push_back({i, t});
  }
  return true;
}

struct Frag {
  std::vector<int> atoms;  // sorted original atom indices
  float prob = 0.f, broken = 0.f;
  std::vector<int> hash;   // 32-dim message-passing hash (integer-valued)
  float form[NFORM];
  float mass = 0.f;
};

inline std::vector<int> frag_hash(const Root& R, const std::vector<int>& atoms, const std::vector<char>& in) {
  std::vector<int> h(BASEF, 0);
  for (int a : atoms)
    for (auto [b, t] : R.nb[a])
      if (in[b])  // each directed edge b -> a contributes t * feat(b)
        for (int k = 0; k < BASEF; k++) h[k] += t * (int)R.feat[b][k];
  return h;
}
inline void frag_formula(const Root& R, const std::vector<int>& atoms, const float* CHEM, Frag& F) {
  for (int e = 0; e < NFORM; e++) F.form[e] = 0.f;
  for (int a : atoms) {
    F.form[R.elem[a]] += 1.f;
    int h = R.hs[a] < MAXH ? R.hs[a] : 0;
    F.form[7] += (float)h;
  }
  float m = 0.f;
  for (int e = 0; e < NFORM; e++) m += F.form[e] * CHEM[e];
  F.mass = m;
}

// random-walk PE (weights = bond type) of one fragment: diag(P^k), k = 1..14
inline void frag_pe(const Root& R, const std::vector<int>& atoms, const std::vector<int>& loc, float* out, int stride) {
  const int n = (int)atoms.size();
  std::vector<float> P((size_t)n * n, 0.f), O, T((size_t)n * n);
  for (int i = 0; i < n; i++) {
    float s = 0.f;
    for (auto [b, t] : R.nb[atoms[i]])
      if (loc[b] >= 0) {
        P[(size_t)i * n + loc[b]] += (float)t;
        s += (float)t;
      }
    s += 1e-30f;
    for (int j = 0; j < n; j++) P[(size_t)i * n + j] /= s;
  }
  O = P;
  for (int k = 0; k < PEK; k++) {
    if (k > 0) {
      for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
          float s = 0.f;
          for (int t = 0; t < n; t++) s += O[(size_t)i * n + t] * P[(size_t)t * n + j];
          T[(size_t)i * n + j] = s;
        }
      O.swap(T);
    }
    for (int i = 0; i < n; i++) out[(size_t)i * stride + k] = O[(size_t)i * n + i];
  }
}

// condition embedding row (adduct | collision | instrument)
inline std::vector<float> cond_row(const Net& M, const Cond& C) {
  std::vector<float> r(M.adduct_dim + 64 + M.instr_dim);
  std::memcpy(r.data(), M.adduct_emb + (size_t)C.adduct * M.adduct_dim, M.adduct_dim * sizeof(float));
  for (int i = 0; i < 32; i++) {
    float a = C.ce / M.coll_den[i];
    r[M.adduct_dim + i] = std::sin(a);
    r[M.adduct_dim + 32 + i] = std::cos(a);
  }
  if (std::isnan(C.ce))
    for (int i = 0; i < 64; i++) r[M.adduct_dim + i] = M.coll_merged[i];
  std::memcpy(r.data() + M.adduct_dim + 64, M.instr_emb + (size_t)C.instrument * M.instr_dim, M.instr_dim * sizeof(float));
  return r;
}

// batch of fragments of the root: node features [base(32) | PE(14) if pe] | cond
inline GBatch make_batch(const Root& R, const std::vector<const std::vector<int>*>& frags, bool pe,
                         const std::vector<float>& cond) {
  GBatch B;
  const int nf = BASEF + (pe ? PEK : 0);
  B.in = nf + (int)cond.size();
  B.G = (int)frags.size();
  for (const auto* fa : frags) B.N += (int)fa->size();
  B.x.assign((size_t)B.N * B.in, 0.f);
  std::vector<int> loc(R.n, -1);
  int off = 0;
  for (int g = 0; g < B.G; g++) {
    const auto& at = *frags[g];
    for (size_t i = 0; i < at.size(); i++) loc[at[i]] = (int)i;
    B.gstart.push_back(off);
    B.gsize.push_back((int)at.size());
    for (size_t i = 0; i < at.size(); i++) {
      float* r = &B.x[(size_t)(off + i) * B.in];
      std::memcpy(r, R.feat[at[i]].data(), BASEF * sizeof(float));
      std::memcpy(r + nf, cond.data(), cond.size() * sizeof(float));
      B.gid.push_back(g);
      for (auto [b, t] : R.nb[at[i]])
        if (loc[b] >= 0) {
          B.src.push_back(off + loc[b]);
          B.dst.push_back(off + (int)i);
          B.typ.push_back(t);
        }
    }
    if (pe) frag_pe(R, at, loc, &B.x[(size_t)off * B.in + BASEF], B.in);
    for (int a : at) loc[a] = -1;
    off += (int)at.size();
  }
  return B;
}

struct IceTrace {
  std::vector<float> leave0;
  double t_gnn = 0, t_mlp = 0, t_frag = 0, t_inten = 0;
  std::vector<int> nodes_per_depth;
  std::vector<Frag> frags;
};

// one prediction: root + condition -> raw [rows, 2] (mass, intensity)
inline std::vector<float> predict(const Net& Gm, const Net& Im, const Root& R, const Cond& C, const float* CHEM,
                                  IceTrace* tr = nullptr) {
  const float HM = CHEM[7];
  std::vector<float> cond_g = cond_row(Gm, C);
  // ---- generation
  std::vector<int> all_atoms(R.n);
  std::iota(all_atoms.begin(), all_atoms.end(), 0);
  std::vector<char> inall(R.n, 1);
  Frag root;
  root.atoms = all_atoms;
  root.prob = 1.f;
  root.broken = 0.f;
  root.hash = frag_hash(R, all_atoms, inall);
  for (int e = 0; e < NFORM; e++) root.form[e] = R.root_form[e];
  root.mass = 0.f;
  for (int e = 0; e < NFORM; e++) root.mass += R.root_form[e] * CHEM[e];
  std::vector<Frag> kept{root};      // the "all_pred" slots (<= MAXN, sorted by prob desc)
  std::vector<int> frontier{0};      // indices into kept
  // root embedding with PE (root_module == gnn), recomputed identically every depth
  std::vector<const std::vector<int>*> rl{&all_atoms};
  GBatch RB = make_batch(R, rl, true, cond_g);
  std::vector<float> rootemb = avg_pool(ggnn(Gm.gnn, RB), RB, Gm.gnn.H);
  const int HG = Gm.gnn.H;
  for (int depth = 0; depth < 3 && !frontier.empty(); depth++) {
    std::vector<const std::vector<int>*> fl;
    for (int f : frontier) fl.push_back(&kept[f].atoms);
    GBatch B = make_batch(R, fl, true, cond_g);
    auto tq0 = std::chrono::steady_clock::now();
    const int D = Gm.mlp_in.in;
    std::vector<float> fe(NFORM * FDIM), de(NFORM * FDIM), o;
    if (GpuGgnnFn f = gpu_ggnn_hook()) {
      // device GGNN + per-atom MLP: only the per-fragment tail [broken one-hot, form enc, diff enc] goes up
      const int tw = D - 3 * HG;
      std::vector<float> tail((size_t)B.G * tw, 0.f);
      for (int g = 0; g < B.G; g++) {
        const Frag& F = kept[frontier[g]];
        float diff[NFORM];
        for (int e = 0; e < NFORM; e++) diff[e] = R.root_form[e] - F.form[e];
        float* r = &tail[(size_t)g * tw];
        r[(int)std::min(F.broken, (float)MAXBR)] = 1.f;
        Gm.form_embed(F.form, r + MAXBR + 1);
        Gm.form_embed(diff, r + MAXBR + 1 + NFORM * FDIM);
      }
      GpuHead hd{rootemb.data(), tail.data(), Gm.mlp_in.w, Gm.mlp_in.b, Gm.mlp_out.w, Gm.mlp_out.b, D, Gm.mlp_in.out};
      o.resize(B.N);
      if (Gm.mlp_out.out != 1 || !f(gpu_w(Gm.gnn), gpu_graph(B), nullptr, &hd, o.data())) o.clear();
      if (tr && !o.empty()) {
        tr->t_gnn += std::chrono::duration<double>(std::chrono::steady_clock::now() - tq0).count();
        tr->nodes_per_depth.push_back(B.N);
      }
    }
    if (o.empty()) {
    std::vector<float> h = ggnn(Gm.gnn, B);
    auto tq1 = std::chrono::steady_clock::now();
    if (tr) { tr->t_gnn += std::chrono::duration<double>(tq1 - tq0).count(); tr->nodes_per_depth.push_back(B.N); }
    std::vector<float> avg = avg_pool(h, B, HG);
    // per-atom MLP input [root, root - avg, h, broken one-hot, form enc, diff enc]
    std::vector<float> X((size_t)B.N * D, 0.f);
    for (int g = 0; g < B.G; g++) {
      const Frag& F = kept[frontier[g]];
      float diff[NFORM];
      for (int e = 0; e < NFORM; e++) diff[e] = R.root_form[e] - F.form[e];
      Gm.form_embed(F.form, fe.data());
      Gm.form_embed(diff, de.data());
      int bk = (int)std::min(F.broken, (float)MAXBR);
      for (int i = B.gstart[g]; i < B.gstart[g] + B.gsize[g]; i++) {
        float* r = &X[(size_t)i * D];
        for (int k = 0; k < HG; k++) {
          r[k] = rootemb[k];
          r[HG + k] = rootemb[k] - avg[(size_t)g * HG + k];
          r[2 * HG + k] = h[(size_t)i * HG + k];
        }
        r[3 * HG + bk] = 1.f;
        std::memcpy(r + 3 * HG + MAXBR + 1, fe.data(), fe.size() * sizeof(float));
        std::memcpy(r + 3 * HG + MAXBR + 1 + fe.size(), de.data(), de.size() * sizeof(float));
      }
    }
    std::vector<float> hid = linear(X, B.N, Gm.mlp_in);
    for (float& t : hid) t = t > 0.f ? t : 0.f;
    o = linear(hid, B.N, Gm.mlp_out);
    auto tq2 = std::chrono::steady_clock::now();
    if (tr) tr->t_mlp += std::chrono::duration<double>(tq2 - tq1).count();
    }
    // candidates (accumulated prob, frontier slot, atom)
    struct Cand {
      float p;
      int g, atom;
    };
    std::vector<Cand> cands;
    for (int g = 0; g < B.G; g++)
      for (int i = 0; i < B.gsize[g]; i++) {
        float p = 1.f / (1.f + std::exp(-o[B.gstart[g] + i]));
        if (depth == 0 && tr) tr->leave0.push_back(p);
        cands.push_back({kept[frontier[g]].prob * p, g, (*fl[g])[i]});
      }
    // top 2*MAXN over [kept probs (MAXN slots, zero padded) | candidates]; take the candidates among them
    std::vector<std::pair<float, int>> pool;  // (prob, -1-slot for kept / candidate index)
    for (int s = 0; s < MAXN; s++) pool.push_back({s < (int)kept.size() ? kept[s].prob : 0.f, -1 - s});
    for (int c = 0; c < (int)cands.size(); c++) pool.push_back({cands[c].p, c});
    std::stable_sort(pool.begin(), pool.end(), [](auto& a, auto& b) { return a.first > b.first; });
    std::vector<int> sel;
    for (int t = 0; t < (int)pool.size() && t < 2 * MAXN; t++)
      if (pool[t].second >= 0 && pool[t].first > 0.f) sel.push_back(pool[t].second);
    if (sel.empty()) break;
    // removals -> connected components (>= 2 atoms)
    std::vector<Frag> nw;
    std::vector<int> comp(R.n);
    for (int c : sel) {
      const Cand& cd = cands[c];
      const Frag& P = kept[frontier[cd.g]];
      std::vector<char> in(R.n, 0);
      for (int a : P.atoms) in[a] = 1;
      in[cd.atom] = 0;
      std::fill(comp.begin(), comp.end(), -1);
      std::vector<std::pair<int, std::vector<int>>> comps;  // (min atom, atoms)
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
        comps.push_back({mem[0], mem});
      }
      for (auto& [mn, mem] : comps) {
        // only components adjacent to the removed atom (all are, the parent is connected)
        float br = 0.f;
        bool adj = false;
        for (auto [v, t] : R.nb[cd.atom])
          if (in[v] && comp[v] == comp[mn]) {
            br += (float)t;
            adj = true;
          }
        if (!adj) continue;
        Frag F;
        F.atoms = mem;
        F.prob = cd.p;
        F.broken = P.broken + br;
        std::vector<char> fin(R.n, 0);
        for (int a : mem) fin[a] = 1;
        F.hash = frag_hash(R, mem, fin);
        frag_formula(R, mem, CHEM, F);
        nw.push_back(std::move(F));
      }
    }
    if (nw.empty()) break;
    // unique by hash (first occurrence over [kept | new]); duplicates among new -> -1
    std::set<std::vector<int>> seen;
    for (const Frag& F : kept) seen.insert(F.hash);
    for (Frag& F : nw) {
      if (seen.count(F.hash)) {
        F.prob = -1.f;
        continue;
      }
      seen.insert(F.hash);
    }
    // keep the top MAXN of [kept | new] by prob (empty kept slots count as 0)
    std::vector<std::pair<float, int>> order;
    for (int s = 0; s < (int)kept.size(); s++) order.push_back({kept[s].prob, s});
    for (int s = (int)kept.size(); s < MAXN; s++) order.push_back({0.f, -1});
    for (int s = 0; s < (int)nw.size(); s++) order.push_back({nw[s].prob, MAXN + s});
    std::stable_sort(order.begin(), order.end(), [](auto& a, auto& b) { return a.first > b.first; });
    std::vector<Frag> nk;
    std::vector<int> nf;
    for (int t = 0; t < MAXN && t < (int)order.size(); t++) {
      int id = order[t].second;
      if (id < 0 || order[t].first <= 0.f) continue;
      if (id < MAXN)
        nk.push_back(kept[id]);
      else {
        nf.push_back((int)nk.size());
        nk.push_back(nw[id - MAXN]);
      }
    }
    kept.swap(nk);
    frontier.swap(nf);
  }
  if (tr) tr->frags = kept;
  auto tq3 = std::chrono::steady_clock::now();
  // ---- intensities
  const int NF = (int)kept.size(), HI = Im.gnn.H;
  std::vector<float> cond_i = cond_row(Im, C);
  GBatch RI = make_batch(R, rl, false, cond_i);
  std::vector<float> rootI = avg_pool(ggnn(Im.gnn, RI), RI, HI);
  std::vector<const std::vector<int>*> fl;
  for (const Frag& F : kept) fl.push_back(&F.atoms);
  GBatch FB = make_batch(R, fl, false, cond_i);
  std::vector<float> avgI = avg_pool(ggnn(Im.gnn, FB), FB, HI);
  const int D = Im.mlp_in.in;
  std::vector<float> X((size_t)NF * D, 0.f);
  for (int f = 0; f < NF; f++) {
    const Frag& F = kept[f];
    float* r = &X[(size_t)f * D];
    for (int k = 0; k < HI; k++) {
      r[k] = rootI[k];
      r[HI + k] = rootI[k] - avgI[(size_t)f * HI + k];
      r[2 * HI + k] = avgI[(size_t)f * HI + k];
    }
    r[3 * HI + (int)std::min(F.broken, (float)MAXBR)] = 1.f;
    float diff[NFORM];
    for (int e = 0; e < NFORM; e++) diff[e] = R.root_form[e] - F.form[e];
    Im.form_embed(F.form, r + 3 * HI + MAXBR + 1);
    Im.form_embed(diff, r + 3 * HI + MAXBR + 1 + NFORM * FDIM);
  }
  std::vector<float> hid = linear(X, NF, Im.mlp_in);
  for (float& t : hid) t = t > 0.f ? t : 0.f;
  std::vector<float> tok = linear(hid, NF, Im.mlp_out);
  for (const EncLayer& L : Im.trans) encoder_layer(L, tok, NF);
  std::vector<float> out = linear(tok, NF, Im.out_map), att = linear(tok, NF, Im.attn_out);
  // peaks: (f, g, h); masses, validity, buckets
  const int NP = NF * 2 * NSH;
  std::vector<float> mass(NP), val(NP), aw(NP);
  std::vector<int> bucket(NP);
  std::vector<char> valid(NP);
  const float off[2] = {C.adduct_mass, C.electron};
  for (int f = 0; f < NF; f++) {
    const Frag& F = kept[f];
    float nhs = F.form[7];
    float maxadd = F.broken, maxrem = std::min(F.broken, nhs);
    for (int g = 0; g < 2; g++)
      for (int hh = 0; hh < NSH; hh++) {
        int p = (f * 2 + g) * NSH + hh;
        float m = F.mass + (off[g] + (float)(hh - MAXBR) * HM);
        mass[p] = m;
        valid[p] = hh <= maxadd + MAXBR && hh >= -maxrem + MAXBR;
        val[p] = out[(size_t)f * 2 * NSH + g * NSH + hh];
        aw[p] = valid[p] ? att[(size_t)f * 2 * NSH + g * NSH + hh] : -99999.f;
        // bucketize(right=False) over linspace(0, 1500, 15000): first i with edge_i >= m
        double x = (double)m, step = 1500.0 / 14999.0;
        int i = (int)std::ceil(x / step - 1e-9);
        i = std::max(0, std::min(i, 14999));
        while (i > 0 && (double)(i - 1) * step >= x) i--;
        while (i < 14999 && (double)i * step < x) i++;
        bucket[p] = i;
      }
  }
  std::map<int, std::vector<int>> byb;
  for (int p = 0; p < NP; p++) byb[bucket[p]].push_back(p);
  std::vector<float> res((size_t)NP * 2);
  for (auto& [b, ps] : byb) {
    float mx = -INFINITY;
    for (int p : ps) mx = std::max(mx, aw[p]);
    double den = 0;
    for (int p : ps) den += std::exp((double)aw[p] - mx);
    double s = 0;
    bool anyv = false;
    std::vector<double> w(ps.size());
    for (size_t q = 0; q < ps.size(); q++) {
      w[q] = std::exp((double)aw[ps[q]] - mx) / den;
      s += w[q] * val[ps[q]];
      anyv |= valid[ps[q]] != 0;
    }
    float bv = anyv ? (float)(1.0 / (1.0 + std::exp(-s))) : 0.f;
    for (size_t q = 0; q < ps.size(); q++) {
      res[(size_t)ps[q] * 2] = mass[ps[q]];
      res[(size_t)ps[q] * 2 + 1] = bv * (float)w[q];
    }
  }
  if (tr) tr->t_inten += std::chrono::duration<double>(std::chrono::steady_clock::now() - tq3).count();
  return res;
}

}  // namespace ice
