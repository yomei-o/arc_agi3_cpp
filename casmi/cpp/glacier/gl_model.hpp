// gl_model.hpp: own C++ inference of GLACIER (ms-pred glacier.JointModel, Coley group, MIT; MassSpecGym checkpoint),
// written from the Python source (ms_pred/glacier/joint_model.py, graphormer/*, nn_utils SlotDecoder, torch
// nn.Transformer{Encoder,Decoder}Layer semantics: post-norm, ReLU, LayerNorm eps 1e-5) and checked stage by stage
// against gl_ref.py dumps (gl_check.cpp).
//
// One prediction = (molecule graph features, adduct, instrument, collision energy) ->
//   Graphormer encoder (5 layers, 256, 8 heads, GELU, spatial / multi-hop edge attention bias) -> node embeddings
//   + graph token;
//   slot decoder (200 learned slots, 4 layers, memory = [formula-mapped graph token; nodes]) -> per slot: breakpoint
//   logits over atoms (slot . node) and a cardinality k in {0..3}; breakpoints = the top-k atoms (LinSAT's projection
//   is strictly monotone per row, so its top-k = the logits' top-k);
//   fragments = connected components left after removing each slot's breakpoint atoms (deduplicated over slots);
//   intensity decoder (4 layers, fragment token = map(graph token, formula / formula-difference sine features,
//   broken-bond count), cross-attention restricted to the fragment's atoms) + encoder (3 layers) -> sigmoid
//   intensities for 2 x 13 peaks per fragment (adduct shift or electron, -6..+6 hydrogens), masked to the allowed
//   hydrogen range.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "glw.hpp"

namespace gl {

constexpr int H = 256, NH = 8, HD = 32, FF = 1024, NSLOT = 200, NFORM = 18, FDIM = 8, MAXDIST = 10, MBB = 6;
constexpr int OUTSZ = 2 * MBB + 1;  // 13 hydrogen shifts
constexpr double HMASS = 1.007825032;

struct Lin {
  const float* w = nullptr;  // [out, in] row-major
  const float* b = nullptr;
  int in = 0, out = 0;
};
struct LN {
  const float* g = nullptr;
  const float* b = nullptr;
};
struct MHA {  // torch nn.MultiheadAttention (in_proj packed q|k|v) or Graphormer's qkv_proj (same packing)
  Lin in, out;
};
struct DecLayer {
  MHA sa, ca;
  Lin l1, l2;
  LN n1, n2, n3;
};
struct EncLayer {
  MHA sa;
  Lin l1, l2;
  LN n1, n2;
};
struct GrLayer {
  MHA sa;
  LN n_sa, n_fin;
  Lin l1, l2;
};

struct Model {
  std::map<std::string, GLArr> W;
  const float *adduct_emb, *coll_den, *coll_merged, *instr_emb;
  int adduct_dim, instr_dim;
  Lin atom_enc, edge_enc, formula_mapper, token_mapper, card_mapper, output_map;
  const float *degree_emb, *graph_token, *edge_dis, *spatial_emb, *virt_dist;
  const float *form_feat, *form_extra;
  std::vector<GrLayer> gr;
  const float* slots;
  std::vector<DecLayer> slot_dec, inten_dec;
  std::vector<EncLayer> inten_enc;

  const float* P(const std::string& n) const {
    auto it = W.find(n);
    if (it == W.end()) throw std::runtime_error("missing weight " + n);
    return it->second.v.data();
  }
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
  LN ln(const std::string& n) const { return LN{P(n + ".weight"), P(n + ".bias")}; }
  MHA torch_mha(const std::string& n) const {
    MHA m;
    const GLArr& a = W.at(n + ".in_proj_weight");
    m.in.w = a.v.data();
    m.in.out = (int)a.d(0);
    m.in.in = (int)a.d(1);
    m.in.b = P(n + ".in_proj_bias");
    m.out = lin(n + ".out_proj");
    return m;
  }
  DecLayer dec(const std::string& n) const {
    DecLayer d;
    d.sa = torch_mha(n + ".self_attn");
    d.ca = torch_mha(n + ".multihead_attn");
    d.l1 = lin(n + ".linear1");
    d.l2 = lin(n + ".linear2");
    d.n1 = ln(n + ".norm1");
    d.n2 = ln(n + ".norm2");
    d.n3 = ln(n + ".norm3");
    return d;
  }
  EncLayer enc(const std::string& n) const {
    EncLayer e;
    e.sa = torch_mha(n + ".self_attn");
    e.l1 = lin(n + ".linear1");
    e.l2 = lin(n + ".linear2");
    e.n1 = ln(n + ".norm1");
    e.n2 = ln(n + ".norm2");
    return e;
  }

  explicit Model(const std::string& path) : W(glw_load(path)) {
    adduct_emb = P("adduct_embedder");
    adduct_dim = (int)W.at("adduct_embedder").d(1);
    coll_den = P("collision_embedder_denominators");
    coll_merged = P("collision_embed_merged");
    instr_emb = P("instrument_embedder");
    instr_dim = (int)W.at("instrument_embedder").d(1);
    const std::string R = "root_module.";
    atom_enc = lin(R + "graph_node_feature.atom_encoder");
    degree_emb = P(R + "graph_node_feature.degree_encoder.weight");
    graph_token = P(R + "graph_node_feature.graph_token.weight");
    edge_enc = lin(R + "graph_attn_bias.edge_encoder");
    edge_dis = P(R + "graph_attn_bias.edge_dis_encoder.weight");
    spatial_emb = P(R + "graph_attn_bias.spatial_pos_encoder.weight");
    virt_dist = P(R + "graph_attn_bias.graph_token_virtual_distance.weight");
    for (int i = 0;; i++) {
      std::string n = R + "layers." + std::to_string(i);
      if (!W.count(n + ".self_attn.qkv_proj.weight")) break;
      GrLayer g;
      g.sa.in = lin(n + ".self_attn.qkv_proj");
      g.sa.out = lin(n + ".self_attn.out_proj");
      g.n_sa = ln(n + ".self_attn_layer_norm");
      g.n_fin = ln(n + ".final_layer_norm");
      g.l1 = lin(n + ".mlp.0");
      g.l2 = lin(n + ".mlp.3");
      gr.push_back(g);
    }
    form_feat = P("embedder.int_to_feat_matrix");
    form_extra = P("embedder._extra_embeddings");
    formula_mapper = lin("formula_mapper");
    token_mapper = lin("token_mapper");
    slots = P("fragment_decoder.slots");
    for (int i = 0; W.count("fragment_decoder.decoder_layers." + std::to_string(i) + ".norm1.weight"); i++)
      slot_dec.push_back(dec("fragment_decoder.decoder_layers." + std::to_string(i)));
    card_mapper = lin("frag_card_mapper");
    for (int i = 0; W.count("inten_decoder.layers." + std::to_string(i) + ".norm1.weight"); i++)
      inten_dec.push_back(dec("inten_decoder.layers." + std::to_string(i)));
    for (int i = 0; W.count("inten_encoder.layers." + std::to_string(i) + ".norm1.weight"); i++)
      inten_enc.push_back(enc("inten_encoder.layers." + std::to_string(i)));
    output_map = lin("output_map");
  }
};

// ---------------------------------------------------------------------------------------------- primitives -------
// y[n, out] = x[n, in] W^T + b
inline void linear(const float* x, int n, const Lin& L, float* y) {
  for (int r = 0; r < n; r++) {
    const float* xr = x + (size_t)r * L.in;
    float* yr = y + (size_t)r * L.out;
    for (int o = 0; o < L.out; o++) {
      const float* w = L.w + (size_t)o * L.in;
      float s = 0.f;
      for (int i = 0; i < L.in; i++) s += xr[i] * w[i];
      yr[o] = s + (L.b ? L.b[o] : 0.f);
    }
  }
}
inline std::vector<float> linear(const std::vector<float>& x, int n, const Lin& L) {
  std::vector<float> y((size_t)n * L.out);
  linear(x.data(), n, L, y.data());
  return y;
}
inline void layernorm(float* x, int n, int d, const LN& p) {
  for (int r = 0; r < n; r++) {
    float* v = x + (size_t)r * d;
    double m = 0, s = 0;
    for (int i = 0; i < d; i++) m += v[i];
    m /= d;
    for (int i = 0; i < d; i++) s += (v[i] - m) * (v[i] - m);
    float inv = (float)(1.0 / std::sqrt(s / d + 1e-5));
    for (int i = 0; i < d; i++) v[i] = (float)((v[i] - m) * inv) * p.g[i] + p.b[i];
  }
}
inline float gelu(float x) { return 0.5f * x * (1.f + std::erf(x * 0.70710678118654752f)); }

// Multi-head attention: q rows [nq, H], kv rows [nk, H]; bias(h, i, j) additive (may be -inf = blocked).
// Output [nq, H] after out_proj. BiasFn: float(int h, int i, int j).
template <class BiasFn>
std::vector<float> attention(const MHA& A, const std::vector<float>& q_in, int nq, const std::vector<float>& kv_in,
                             int nk, BiasFn bias) {
  // projections: q from rows 0..H-1 of in_proj, k from H..2H-1, v from 2H..3H-1
  std::vector<float> q((size_t)nq * H), k((size_t)nk * H), v((size_t)nk * H);
  Lin Lq{A.in.w, A.in.b, H, H}, Lk{A.in.w + (size_t)H * H, A.in.b + H, H, H},
      Lv{A.in.w + (size_t)2 * H * H, A.in.b + 2 * H, H, H};
  linear(q_in.data(), nq, Lq, q.data());
  linear(kv_in.data(), nk, Lk, k.data());
  linear(kv_in.data(), nk, Lv, v.data());
  const float sc = 1.f / std::sqrt((float)HD);
  std::vector<float> ctx((size_t)nq * H, 0.f), sco(nk);
  for (int h = 0; h < NH; h++) {
    for (int i = 0; i < nq; i++) {
      const float* qi = &q[(size_t)i * H + h * HD];
      float mx = -INFINITY;
      for (int j = 0; j < nk; j++) {
        const float* kj = &k[(size_t)j * H + h * HD];
        float s = 0.f;
        for (int t = 0; t < HD; t++) s += qi[t] * kj[t];
        s = s * sc + bias(h, i, j);
        sco[j] = s;
        if (s > mx) mx = s;
      }
      float* ci = &ctx[(size_t)i * H + h * HD];
      if (mx == -INFINITY) {  // fully masked row: torch gives NaN; only padded rows can hit this
        for (int t = 0; t < HD; t++) ci[t] = 0.f;
        continue;
      }
      float den = 0.f;
      for (int j = 0; j < nk; j++) {
        sco[j] = std::exp(sco[j] - mx);
        den += sco[j];
      }
      float inv = 1.f / den;
      for (int j = 0; j < nk; j++) {
        float p = sco[j] * inv;
        if (p == 0.f) continue;
        const float* vj = &v[(size_t)j * H + h * HD];
        for (int t = 0; t < HD; t++) ci[t] += p * vj[t];
      }
    }
  }
  return linear(ctx, nq, A.out);
}

inline void add_inplace(std::vector<float>& a, const std::vector<float>& b) {
  for (size_t i = 0; i < a.size(); i++) a[i] += b[i];
}
inline std::vector<float> ffn_relu(const std::vector<float>& x, int n, const Lin& l1, const Lin& l2) {
  std::vector<float> h = linear(x, n, l1);
  for (float& t : h) t = t > 0.f ? t : 0.f;
  return linear(h, n, l2);
}

// torch TransformerDecoderLayer (post-norm, ReLU). memAllowed(i, j): may query i attend memory j.
template <class MemMask>
void decoder_layer(const DecLayer& L, std::vector<float>& x, int n, const std::vector<float>& mem, int nm,
                   MemMask memAllowed) {
  auto none = [](int, int, int) { return 0.f; };
  add_inplace(x, attention(L.sa, x, n, x, n, none));
  layernorm(x.data(), n, H, L.n1);
  add_inplace(x, attention(L.ca, x, n, mem, nm,
                           [&](int, int i, int j) { return memAllowed(i, j) ? 0.f : -INFINITY; }));
  layernorm(x.data(), n, H, L.n2);
  add_inplace(x, ffn_relu(x, n, L.l1, L.l2));
  layernorm(x.data(), n, H, L.n3);
}
inline void encoder_layer(const EncLayer& L, std::vector<float>& x, int n) {
  auto none = [](int, int, int) { return 0.f; };
  add_inplace(x, attention(L.sa, x, n, x, n, none));
  layernorm(x.data(), n, H, L.n1);
  add_inplace(x, ffn_relu(x, n, L.l1, L.l2));
  layernorm(x.data(), n, H, L.n2);
}

// ---------------------------------------------------------------------------------------------- inputs ----------
// Per-molecule features (= TreeProcessor.featurize_tree + create_graphormer_input), built by gl_feat.hpp.
struct MolFeat {
  int n = 0;                      // heavy atoms
  int nfeat = 88;                 // node feature columns
  std::vector<float> x;           // [n, nfeat]
  std::vector<int> degree;        // [n]
  std::vector<int> spatial;       // [n, n]
  std::vector<float> edge_input;  // [n, n, MAXDIST, 28]
  std::vector<float> adj_bo;      // [n, n] bond orders (aromatic 1.5)
  std::vector<float> masses;      // [n] atom mass + its hydrogens (float32 as in the collate)
  std::vector<int> atom_hs;       // [n]
  int total_hs = 0;
  std::vector<float> root_form;   // [NFORM] element counts incl. H
  std::vector<float> atom_form;   // [n, NFORM]
};

struct Cond {
  int adduct = 0;        // ion2onehot_pos
  int instrument = 1;    // instrument2onehot_pos (QTOF 1)
  float ce = 20.f;       // collision energy bucket (NaN -> merged embedding)
  float adduct_mass = 1.007276452f;
  float electron = -0.00054858f;  // -ELECTRON_MASS for positive adducts
};

struct Trace {  // optional intermediates for the checker
  std::vector<float> node_emb, root_tok, frag_logits, card_logits;
  std::vector<int> brk_k;
  std::vector<std::vector<int>> breakpoints;
  std::vector<std::vector<uint8_t>> fragments;
};

// formula count vector -> sine features (FourierFeaturizerAbsoluteSines; >= 255 -> the learned extra embedding)
inline void form_embed(const Model& M, const float* counts, float* out) {
  for (int e = 0; e < NFORM; e++) {
    float c = counts[e];
    const float* row;
    if (c >= 255.f)
      row = M.form_extra;
    else {
      int ci = (int)c;
      if (ci < 0) ci += 255;  // torch negative indexing wraps (not reached for valid formulas)
      row = M.form_feat + (size_t)ci * FDIM;
    }
    std::memcpy(out + e * FDIM, row, FDIM * sizeof(float));
  }
}

// ---------------------------------------------------------------------------------------------- forward ---------
// Returns the raw [rows, 2] (mass, intensity) output (= predict_inten_frag_batch spec rows) for one prediction.
inline std::vector<float> predict(const Model& M, const MolFeat& F, const Cond& C, Trace* tr = nullptr) {
  const int n = F.n, T = n + 1;
  // node input: [x | adduct | collision | instrument]
  const int in_dim = M.atom_enc.in;
  std::vector<float> xin((size_t)n * in_dim, 0.f);
  std::vector<float> coll(64);
  for (int i = 0; i < 32; i++) {
    float a = C.ce / M.coll_den[i];
    coll[i] = std::sin(a);
    coll[32 + i] = std::cos(a);
  }
  if (std::isnan(C.ce))
    for (int i = 0; i < 64; i++) coll[i] = M.coll_merged[i];
  for (int i = 0; i < n; i++) {
    float* r = &xin[(size_t)i * in_dim];
    std::memcpy(r, &F.x[(size_t)i * F.nfeat], F.nfeat * sizeof(float));
    std::memcpy(r + F.nfeat, M.adduct_emb + (size_t)C.adduct * M.adduct_dim, M.adduct_dim * sizeof(float));
    std::memcpy(r + F.nfeat + M.adduct_dim, coll.data(), 64 * sizeof(float));
    std::memcpy(r + F.nfeat + M.adduct_dim + 64, M.instr_emb + (size_t)C.instrument * M.instr_dim,
                M.instr_dim * sizeof(float));
  }
  // Graphormer node features: token 0 = graph token
  std::vector<float> x((size_t)T * H);
  std::memcpy(x.data(), M.graph_token, H * sizeof(float));
  linear(xin.data(), n, M.atom_enc, x.data() + H);
  for (int i = 0; i < n; i++) {
    int d = std::min(std::max(F.degree[i], 0), 7);
    for (int t = 0; t < H; t++) x[(size_t)(i + 1) * H + t] += M.degree_emb[(size_t)d * H + t];
  }
  // attention bias [NH, T, T]
  std::vector<float> bias((size_t)NH * T * T, 0.f);
  {
    const int EF = M.edge_enc.in;
    std::vector<float> e(NH), acc(NH);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) {
        int sp = F.spatial[(size_t)i * n + j];
        int spc = sp == 0 ? 1 : sp;
        if (spc > 1) spc -= 1;
        spc = std::min(std::max(spc, 0), MAXDIST);
        std::fill(acc.begin(), acc.end(), 0.f);
        for (int hop = 0; hop < MAXDIST; hop++) {
          const float* ei = &F.edge_input[(((size_t)i * n + j) * MAXDIST + hop) * EF];
          for (int h = 0; h < NH; h++) {
            const float* w = M.edge_enc.w + (size_t)h * EF;
            float s = M.edge_enc.b[h];
            for (int f = 0; f < EF; f++) s += ei[f] * w[f];
            e[h] = s;
          }
          const float* Wd = M.edge_dis + (size_t)hop * NH * NH;  // [NH(in), NH(out)]
          for (int o = 0; o < NH; o++) {
            float s = 0.f;
            for (int h = 0; h < NH; h++) s += e[h] * Wd[h * NH + o];
            acc[o] += s;
          }
        }
        int spi = std::min(sp, 1024);
        for (int h = 0; h < NH; h++)
          bias[((size_t)h * T + i + 1) * T + j + 1] = M.spatial_emb[(size_t)spi * NH + h] + acc[h] / (float)spc;
      }
    for (int h = 0; h < NH; h++) {
      for (int i = 1; i < T; i++) bias[((size_t)h * T + i) * T + 0] += M.virt_dist[h];
      for (int j = 0; j < T; j++) bias[((size_t)h * T + 0) * T + j] += M.virt_dist[h];
    }
  }
  for (const GrLayer& L : M.gr) {
    std::vector<float> a = attention(L.sa, x, T, x, T, [&](int h, int i, int j) {
      return bias[((size_t)h * T + i) * T + j];
    });
    add_inplace(x, a);
    layernorm(x.data(), T, H, L.n_sa);
    std::vector<float> h1 = linear(x, T, L.l1);
    for (float& t : h1) t = gelu(t);
    add_inplace(x, linear(h1, T, L.l2));
    layernorm(x.data(), T, H, L.n_fin);
  }
  std::vector<float> root(x.begin(), x.begin() + H);          // graph_rep
  std::vector<float> nodes(x.begin() + H, x.end());           // [n, H]
  if (tr) {
    tr->node_emb = nodes;
    tr->root_tok = root;
  }
  // ---- breakpoints: slot decoder over memory [formula-mapped root; nodes]
  std::vector<float> rf(H + NFORM * FDIM);
  std::memcpy(rf.data(), root.data(), H * sizeof(float));
  form_embed(M, F.root_form.data(), rf.data() + H);
  std::vector<float> mem((size_t)T * H);
  linear(rf.data(), 1, M.formula_mapper, mem.data());
  std::memcpy(mem.data() + H, nodes.data(), (size_t)n * H * sizeof(float));
  std::vector<float> s(M.slots, M.slots + (size_t)NSLOT * H);
  for (const DecLayer& L : M.slot_dec) decoder_layer(L, s, NSLOT, mem, T, [](int, int) { return true; });
  std::vector<float> card = linear(s, NSLOT, M.card_mapper);  // [NSLOT, 4]
  std::vector<float> logits((size_t)NSLOT * n);
  for (int a = 0; a < NSLOT; a++)
    for (int i = 0; i < n; i++) {
      float v = 0.f;
      for (int t = 0; t < H; t++) v += s[(size_t)a * H + t] * nodes[(size_t)i * H + t];
      logits[(size_t)a * n + i] = v;
    }
  if (tr) {
    tr->frag_logits = logits;
    tr->card_logits = card;
  }
  // ---- fragments: remove the top-k atoms of each slot, connected components of the rest, unique over slots
  std::vector<std::vector<int>> nb(n);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      if (F.adj_bo[(size_t)i * n + j] > 0.f) nb[i].push_back(j);
  std::set<std::vector<uint8_t>> uniq;
  std::vector<int> idx(n);
  for (int a = 0; a < NSLOT; a++) {
    const float* c = &card[(size_t)a * 4];
    int k = (int)(std::max_element(c, c + 4) - c);
    std::vector<uint8_t> removed(n, 0);
    if (k > 0) {
      std::iota(idx.begin(), idx.end(), 0);
      const float* lg = &logits[(size_t)a * n];
      int kk = std::min(k, n);
      std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int p, int q) {
        return lg[p] > lg[q] || (lg[p] == lg[q] && p < q);
      });
      for (int t = 0; t < kk; t++) removed[idx[t]] = 1;
    }
    if (tr) {
      tr->brk_k.push_back(k);
      std::vector<int> b;
      for (int i = 0; i < n; i++)
        if (removed[i]) b.push_back(i);
      tr->breakpoints.push_back(b);
    }
    std::vector<int> comp(n, -1);
    for (int i = 0; i < n; i++) {
      if (removed[i] || comp[i] >= 0) continue;
      std::vector<int> st{i}, members;
      comp[i] = i;
      while (!st.empty()) {
        int u = st.back();
        st.pop_back();
        members.push_back(u);
        for (int v : nb[u])
          if (!removed[v] && comp[v] < 0) {
            comp[v] = i;
            st.push_back(v);
          }
      }
      if (members.size() < 2) continue;  // reachability has no self loops: an isolated atom gives an empty row
      std::vector<uint8_t> m(n, 0);
      for (int u : members) m[u] = 1;
      uniq.insert(m);
    }
  }
  // torch.unique sorts rows lexicographically (0 < 1), which std::set<vector<uint8_t>> also does
  std::vector<std::vector<uint8_t>> frags(uniq.begin(), uniq.end());
  if (frags.empty()) frags.push_back(std::vector<uint8_t>(n, 1));
  if (tr) tr->fragments = frags;
  const int NF = (int)frags.size();
  // ---- intensity tokens
  std::vector<float> tok_in((size_t)NF * M.token_mapper.in, 0.f);
  std::vector<float> fmass(NF), maxadd(NF), maxrem(NF);
  for (int f = 0; f < NF; f++) {
    const auto& m = frags[f];
    float* r = &tok_in[(size_t)f * M.token_mapper.in];
    std::memcpy(r, root.data(), H * sizeof(float));
    std::vector<float> fv(NFORM, 0.f), dv(NFORM);
    float hs = 0.f, broken = 0.f, mass = 0.f;
    for (int i = 0; i < n; i++)
      if (m[i]) {
        for (int e = 0; e < NFORM; e++) fv[e] += F.atom_form[(size_t)i * NFORM + e];
        hs += (float)F.atom_hs[i];
        mass += F.masses[i];
        for (int j = 0; j < n; j++)
          if (!m[j]) broken += F.adj_bo[(size_t)i * n + j];
      }
    for (int e = 0; e < NFORM; e++) dv[e] = F.root_form[e] - fv[e];
    form_embed(M, fv.data(), r + H);
    form_embed(M, dv.data(), r + H + NFORM * FDIM);
    int nbk = (int)std::min(broken, (float)MBB);
    r[H + 2 * NFORM * FDIM + nbk] = 1.f;
    fmass[f] = mass;
    maxrem[f] = std::min(hs, (float)MBB);
    maxadd[f] = std::min((float)F.total_hs - hs, (float)MBB);
  }
  std::vector<float> tok = linear(tok_in, NF, M.token_mapper);
  for (const DecLayer& L : M.inten_dec)
    decoder_layer(L, tok, NF, nodes, n, [&](int f, int j) { return frags[f][j] != 0; });
  for (const EncLayer& L : M.inten_enc) encoder_layer(L, tok, NF);
  std::vector<float> o = linear(tok, NF, M.output_map);  // [NF, 26]
  std::vector<float> out((size_t)NF * 2 * OUTSZ * 2);
  const float ashift[2] = {C.adduct_mass, C.electron};
  for (int f = 0; f < NF; f++)
    for (int sidx = 0; sidx < 2; sidx++)
      for (int hh = 0; hh < OUTSZ; hh++) {
        size_t r = ((size_t)f * 2 + sidx) * OUTSZ + hh;
        float hshift = (float)(hh - MBB) * (float)HMASS;
        float m = fmass[f] + hshift + ashift[sidx];
        if (!(m > 0.f)) m = 0.f;
        bool ok = hh <= maxadd[f] + MBB && hh >= -maxrem[f] + MBB;
        float v = 1.f / (1.f + std::exp(-o[(size_t)f * 2 * OUTSZ + sidx * OUTSZ + hh]));
        out[r * 2] = m;
        out[r * 2 + 1] = ok ? v : 0.f;
      }
  return out;
}

// = gl_runner.post: drop <= 0 intensity / mass -> merge identical m/z (4 dp) by MAX -> top-k -> sort by m/z ->
// max-normalise
inline void post(const std::vector<float>& raw, std::vector<double>& mz, std::vector<double>& it, int k = 100) {
  std::map<double, double> m;
  for (size_t r = 0; r + 1 < raw.size(); r += 2) {
    float a = raw[r], b = raw[r + 1];
    if (!(b > 0.f && a > 0.f)) continue;
    double key = std::nearbyint((double)a * 1e4) / 1e4;
    auto ins = m.emplace(key, (double)b);
    if (!ins.second && (double)b > ins.first->second) ins.first->second = b;
  }
  std::vector<std::pair<double, double>> v(m.begin(), m.end());
  if ((int)v.size() > k) {
    std::vector<size_t> o(v.size());
    std::iota(o.begin(), o.end(), 0);
    std::stable_sort(o.begin(), o.end(), [&](size_t p, size_t q) { return v[p].second > v[q].second; });
    std::vector<std::pair<double, double>> w;
    for (int t = 0; t < k; t++) w.push_back(v[o[t]]);
    std::sort(w.begin(), w.end());
    v = w;
  }
  mz.clear();
  it.clear();
  double mx = 0;
  for (auto& p : v) mx = std::max(mx, p.second);
  for (auto& p : v) {
    mz.push_back((double)(float)p.first);
    it.push_back((double)(float)(p.second / mx));
  }
}

}  // namespace gl
