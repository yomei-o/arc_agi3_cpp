// FPNet (ahmedberatozer casmi26-fpnet-full1, casmi/fpnet.py of casmi26-v4b-models) inference with ggml (CPU backend).
// Spectrum -> 10226 fingerprint-bit logits; molecule logits = 0.5*(mean per-spectrum + mean per-polarity merged-spectrum),
// exactly as probe_core2.molecule_logits (ce unknown -> ce=0, ce_known=0; n_merged = 1 / min(sum,8)).
// One spectrum per graph, no padding -> attention needs no mask (identical to the masked padded batch).
// weights: FPN1 (f32) or FPN2 (2-D weights f16, to_f16.py)
// usage: fpnet_ggml <weights.bin> <data_dir> <stem> <out_prefix> [threads=4] [max_mols=0] [write_spec=0]
// writes <out>_mol.bin: i32 nmol, i32 nbits, then per molecule: i32 mol, f32[nbits]
//        <out>_spec.bin (write_spec): i32 nspec, i32 nbits, then per spectrum (head order): f32[nbits]
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

static std::vector<char> slurp(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) { fprintf(stderr, "cannot read %s\n", p.c_str()); exit(1); }
  std::streamsize n = f.tellg(); f.seekg(0);
  std::vector<char> b((size_t)n); f.read(b.data(), n); return b;
}

// our adduct vocabulary (tag/16) -> index in casmi/chem.py ADDUCT_LIST (60 entries, last = <unk>)
//   [M+H]+ 0, [M-H]- 6, [M+Na]+ 4, [M+NH4]+ 1, [M+K]+ 5, [M+Cl]- 9, [M+CH2O2-H]- 8, [M+H-H2O]+ = [M-H2O+H]+ 2,
//   [2M+H]+ 46, [2M+Na]+ 47
static const int OUR2CHEM[10] = {0, 6, 4, 1, 5, 9, 8, 2, 46, 47};
static const float OUR_MODE[10] = {1, -1, 1, 1, 1, -1, -1, 1, 1, 1};

struct Model {
  int nbits, d, layers, heads;
  ggml_context* ctx = nullptr;
  std::map<std::string, ggml_tensor*> t;
  std::map<std::string, std::vector<float>> host;
  ggml_tensor* get(const std::string& n) {
    auto it = t.find(n); if (it == t.end()) { fprintf(stderr, "missing %s\n", n.c_str()); exit(1); } return it->second;
  }
};

static void load_model(Model& M, const std::string& path) {
  std::vector<char> b = slurp(path);
  const char* p = b.data();
  // FPN1: f32 tensor data; FPN2: 2-D weight matrices stored f16 (1-D tensors stay f32), widened to f32 here
  const bool f16 = !memcmp(p, "FPN2", 4);
  if (!f16 && memcmp(p, "FPN1", 4)) { fprintf(stderr, "bad magic\n"); exit(1); }
  p += 4; int hdr[5]; memcpy(hdr, p, 20); p += 20;
  M.nbits = hdr[0]; M.d = hdr[1]; M.layers = hdr[2]; M.heads = hdr[3]; int nt = hdr[4];
  ggml_init_params ip = {b.size() * (f16 ? 2 : 1) + (size_t)nt * ggml_tensor_overhead() + (1 << 20), nullptr, false};
  M.ctx = ggml_init(ip);
  for (int i = 0; i < nt; i++) {
    int nl; memcpy(&nl, p, 4); p += 4; std::string name(p, nl); p += nl;
    int nd; memcpy(&nd, p, 4); p += 4; std::vector<int> dims(nd); memcpy(dims.data(), p, 4 * nd); p += 4 * nd;
    size_t n = 1; for (int x : dims) n *= x;
    ggml_tensor* T;
    if (nd == 1) T = ggml_new_tensor_1d(M.ctx, GGML_TYPE_F32, dims[0]);
    else T = ggml_new_tensor_2d(M.ctx, GGML_TYPE_F32, dims[1], dims[0]);  // torch [out,in] -> ne0=in, ne1=out
    const bool h16 = f16 && nd == 2;
    if (h16) ggml_fp16_to_fp32_row((const ggml_fp16_t*)p, (float*)T->data, (int64_t)n);
    else memcpy(T->data, p, n * 4);
    const float* td = (const float*)T->data;
    if (n <= 60 * 512) M.host[name] = std::vector<float>(td, td + n);
    p += n * (h16 ? 2 : 4); M.t[name] = T;
  }
}

// casmi/fpnet.py prep_peaks (MAX_PEAKS=160, floor 1e-3, mz <= prec+2, sqrt intensities)
static void prep_peaks(const float* mz0, const float* it0, int n, double prec, std::vector<float>& omz, std::vector<float>& ov) {
  omz.clear(); ov.clear();
  std::vector<double> mz, it;
  for (int i = 0; i < n; i++) if (mz0[i] <= prec + 2.0 && mz0[i] > 0) { mz.push_back(mz0[i]); it.push_back(it0[i]); }
  if (mz.empty()) return;
  double mx = *std::max_element(it.begin(), it.end()); if (mx <= 0) return;
  std::vector<double> m2, i2;
  for (size_t i = 0; i < mz.size(); i++) { double v = it[i] / mx; if (v >= 1e-3) { m2.push_back(mz[i]); i2.push_back(v); } }
  std::vector<int> idx(m2.size()); std::iota(idx.begin(), idx.end(), 0);
  if (idx.size() > 160) {
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return i2[a] > i2[b]; });
    idx.resize(160);
  }
  std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return m2[a] < m2[b]; });
  for (int i : idx) { omz.push_back((float)m2[i]); ov.push_back((float)std::sqrt(i2[i])); }
}

// casmi/spectra.py merge_spectra(mz_tol=0.005)
static void merge_spectra(const std::vector<std::pair<std::vector<float>, std::vector<float>>>& S, std::vector<float>& omz, std::vector<float>& oit) {
  std::vector<double> mz, it;
  for (auto& s : S) {
    if (s.first.empty()) continue;
    double mx = *std::max_element(s.second.begin(), s.second.end()); if (mx <= 0) continue;
    for (size_t i = 0; i < s.first.size(); i++) { mz.push_back(s.first[i]); it.push_back(s.second[i] / mx); }
  }
  omz.clear(); oit.clear(); if (mz.empty()) return;
  std::vector<int> o(mz.size()); std::iota(o.begin(), o.end(), 0);
  std::stable_sort(o.begin(), o.end(), [&](int a, int b) { return mz[a] < mz[b]; });
  std::vector<double> m(o.size()), v(o.size());
  for (size_t i = 0; i < o.size(); i++) { m[i] = mz[o[i]]; v[i] = it[o[i]]; }
  std::vector<char> keep(m.size(), 1); size_t last = 0;
  for (size_t j = 1; j < m.size(); j++) {
    if (m[j] - m[last] < 0.005) { if (v[j] >= v[last]) { keep[last] = 0; last = j; } else keep[j] = 0; }
    else last = j;
  }
  for (size_t j = 0; j < m.size(); j++) if (keep[j]) { omz.push_back((float)m[j]); oit.push_back((float)v[j]); }
}

struct Item { std::vector<float> mz, it; float prec; int ad; float ce, ce_known, merged, mode; };

static void sinemb(const std::vector<float>& inv, float x, float* out) {
  size_t n = inv.size();
  for (size_t i = 0; i < n; i++) { float a = x * inv[i]; out[i] = sinf(a); out[n + i] = cosf(a); }
}
static ggml_tensor* layernorm(ggml_context* c, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
  return ggml_add(c, ggml_mul(c, ggml_norm(c, x, 1e-5f), w), b);
}
static ggml_tensor* linear(ggml_context* c, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
  return ggml_add(c, ggml_mul_mat(c, w, x), b);
}

static std::vector<char> g_buf;

static void forward(Model& M, const Item& it, int nth, float* out) {
  const int d = M.d, H = M.heads, hd = d / H;
  std::vector<float> mz = it.mz, v = it.it;
  if (mz.empty()) { mz.push_back(0.f); v.push_back(0.f); }   // collate(): empty spectrum -> one unpadded zero peak
  const int Np = (int)mz.size(), N = Np + 1;
  if (g_buf.empty()) g_buf.resize((size_t)768 << 20);
  ggml_init_params ip = {g_buf.size(), g_buf.data(), false};
  ggml_context* c = ggml_init(ip);
  ggml_tensor* F = ggml_new_tensor_2d(c, GGML_TYPE_F32, 3 * d, Np);   // [mz_emb | nl_emb | int_emb(it, it^2)]
  const auto& imz = M.host["mz_emb.inv"]; const auto& inl = M.host["nl_emb.inv"];
  const auto& iw = M.host["int_emb.weight"]; const auto& ib = M.host["int_emb.bias"];
  for (int j = 0; j < Np; j++) {
    float* f = (float*)F->data + (size_t)j * 3 * d;
    sinemb(imz, mz[j], f);
    sinemb(inl, std::max(it.prec - mz[j], 0.0f), f + d);
    float a = v[j], a2 = v[j] * v[j];
    for (int k = 0; k < d; k++) f[2 * d + k] = iw[2 * k] * a + iw[2 * k + 1] * a2 + ib[k];
  }
  ggml_tensor* G = ggml_new_tensor_1d(c, GGML_TYPE_F32, d + d / 4 + 4);
  {
    float* g = (float*)G->data;
    sinemb(M.host["prec_emb.inv"], it.prec, g);
    sinemb(M.host["ce_emb.inv"], it.ce / 10.0f + 1.0f, g + d);
    g[d + d / 4] = it.ce_known; g[d + d / 4 + 1] = it.merged; g[d + d / 4 + 2] = it.mode;
    g[d + d / 4 + 3] = log1pf(it.prec) / 10.0f;
  }
  ggml_tensor* AD = ggml_new_tensor_1d(c, GGML_TYPE_F32, d);
  memcpy(AD->data, M.host["ad.weight"].data() + (size_t)it.ad * d, d * 4);

  ggml_tensor* p = linear(c, F, M.get("pk.weight"), M.get("pk.bias"));                     // [d, Np]
  ggml_tensor* g = ggml_add(c, linear(c, G, M.get("gl.weight"), M.get("gl.bias")), AD);   // [d]
  ggml_tensor* x = ggml_concat(c, ggml_reshape_2d(c, g, d, 1), p, 1);                      // [d, N]
  const float scale = 1.0f / sqrtf((float)hd);
  for (int l = 0; l < M.layers; l++) {
    std::string pre = "blocks." + std::to_string(l) + ".";
    ggml_tensor* y = layernorm(c, x, M.get(pre + "n1.weight"), M.get(pre + "n1.bias"));
    ggml_tensor* qkv = linear(c, y, M.get(pre + "qkv.weight"), M.get(pre + "qkv.bias"));  // [3d, N] = [3][H][hd]
    ggml_tensor* q = ggml_view_3d(c, qkv, hd, H, N, hd * 4, qkv->nb[1], 0);
    ggml_tensor* k = ggml_view_3d(c, qkv, hd, H, N, hd * 4, qkv->nb[1], (size_t)d * 4);
    ggml_tensor* vv = ggml_view_3d(c, qkv, hd, H, N, hd * 4, qkv->nb[1], (size_t)2 * d * 4);
    q = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));          // [hd, N, H]
    k = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));          // [hd, N, H]
    ggml_tensor* kq = ggml_mul_mat(c, k, q);                    // [Nk, Nq, H]
    kq = ggml_soft_max_ext(c, kq, nullptr, scale, 0.0f);
    ggml_tensor* vt = ggml_cont(c, ggml_permute(c, vv, 1, 2, 0, 3));  // [N, hd, H]
    ggml_tensor* kqv = ggml_mul_mat(c, vt, kq);                 // [hd, Nq, H]
    ggml_tensor* a = ggml_cont(c, ggml_permute(c, kqv, 0, 2, 1, 3));  // [hd, H, N]
    a = ggml_reshape_2d(c, a, d, N);
    x = ggml_add(c, x, linear(c, a, M.get(pre + "o.weight"), M.get(pre + "o.bias")));
    ggml_tensor* h = layernorm(c, x, M.get(pre + "n2.weight"), M.get(pre + "n2.bias"));
    h = ggml_gelu_erf(c, linear(c, h, M.get(pre + "ff.0.weight"), M.get(pre + "ff.0.bias")));
    x = ggml_add(c, x, linear(c, h, M.get(pre + "ff.3.weight"), M.get(pre + "ff.3.bias")));
  }
  x = layernorm(c, x, M.get("norm.weight"), M.get("norm.bias"));
  ggml_tensor* cls = ggml_cont(c, ggml_view_2d(c, x, d, 1, x->nb[1], 0));
  ggml_tensor* pk = ggml_view_2d(c, x, d, Np, x->nb[1], x->nb[1]);
  ggml_tensor* mean = ggml_mean(c, ggml_cont(c, ggml_transpose(c, pk)));  // [1, d]
  mean = ggml_reshape_2d(c, mean, d, 1);
  ggml_tensor* emb = ggml_concat(c, cls, mean, 0);                        // [2d, 1]
  ggml_tensor* h = ggml_gelu_erf(c, linear(c, emb, M.get("head.0.weight"), M.get("head.0.bias")));
  ggml_tensor* z = linear(c, h, M.get("head.3.weight"), M.get("head.3.bias"));  // [nbits, 1]
  ggml_cgraph* gf = ggml_new_graph_custom(c, 4096, false);
  ggml_build_forward_expand(gf, z);
  ggml_graph_compute_with_ctx(c, gf, nth);
  memcpy(out, z->data, (size_t)M.nbits * 4);
  ggml_free(c);
}

int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: fpnet_ggml weights.bin data_dir stem out_prefix [threads] [max_mols] [write_spec]\n"); return 1; }
  std::string wpath = argv[1], dir = argv[2], stem = argv[3], outp = argv[4];
  int nth = argc > 5 ? atoi(argv[5]) : 4, maxm = argc > 6 ? atoi(argv[6]) : 0, wspec = argc > 7 ? atoi(argv[7]) : 0;
  Model M; load_model(M, wpath);
  fprintf(stderr, "model nbits %d d %d layers %d heads %d\n", M.nbits, M.d, M.layers, M.heads);
  const int UNK = (int)(M.host["ad.weight"].size() / M.d) - 1;
  struct Head { float prec; int32_t mol, off, n, tag; };
  std::vector<char> hb = slurp(dir + "/" + stem + "_head.bin"), pb = slurp(dir + "/" + stem + "_peaks.bin");
  size_t ns = hb.size() / sizeof(Head); const Head* hd = (const Head*)hb.data();
  const float* P = (const float*)pb.data();
  struct Spec { int mol; std::vector<float> mz, it; float prec; int ad; float mode; };
  std::vector<Spec> S(ns);
  for (size_t i = 0; i < ns; i++) {
    const Head& h = hd[i]; Spec& s = S[i];
    s.mol = h.mol; s.prec = h.prec; s.mz.assign(P, P + h.n); P += h.n; s.it.assign(P, P + h.n); P += h.n;
    int a = h.tag < 0 ? -1 : (h.tag % 16 == 15 ? h.tag / 16 + 1 : h.tag / 16);
    s.ad = (a >= 0 && a < 10) ? OUR2CHEM[a] : UNK;
    s.mode = (a >= 0 && a < 10) ? OUR_MODE[a] : 1.f;
  }
  std::map<int, std::vector<int>> bymol;
  for (size_t i = 0; i < ns; i++) bymol[S[i].mol].push_back((int)i);
  std::vector<float> specz;
  if (wspec) specz.assign(ns * (size_t)M.nbits, 0.f);
  FILE* fo = fopen((outp + "_mol.bin").c_str(), "wb");
  int nmol = maxm > 0 ? std::min<int>(maxm, (int)bymol.size()) : (int)bymol.size();
  fwrite(&nmol, 4, 1, fo); fwrite(&M.nbits, 4, 1, fo);
  std::vector<float> z(M.nbits), zs(M.nbits), zm(M.nbits), mol(M.nbits);
  auto t0 = std::chrono::steady_clock::now(); int nfw = 0, done = 0;
  for (auto& kv : bymol) {
    if (done >= nmol) break;
    std::fill(zs.begin(), zs.end(), 0.f); std::fill(zm.begin(), zm.end(), 0.f);
    int cs = 0, cm = 0;
    for (int i : kv.second) {
      Item it; prep_peaks(S[i].mz.data(), S[i].it.data(), (int)S[i].mz.size(), S[i].prec, it.mz, it.it);
      it.prec = S[i].prec; it.ad = S[i].ad; it.ce = 0; it.ce_known = 0; it.merged = 1 / 4.0f; it.mode = S[i].mode;
      forward(M, it, nth, z.data()); nfw++;
      if (wspec) std::copy(z.begin(), z.end(), specz.begin() + (size_t)i * M.nbits);
      for (int b = 0; b < M.nbits; b++) zs[b] += z[b];
      cs++;
    }
    for (float md : {1.f, -1.f}) {
      std::vector<std::pair<std::vector<float>, std::vector<float>>> grp; std::vector<float> precs; std::vector<int> ads;
      for (int i : kv.second) if (S[i].mode == md) { grp.push_back({S[i].mz, S[i].it}); precs.push_back(S[i].prec); ads.push_back(S[i].ad); }
      if (grp.empty()) continue;
      std::vector<float> mm, ii; merge_spectra(grp, mm, ii);
      std::vector<float> ps = precs; std::sort(ps.begin(), ps.end());
      double pm = ps.size() % 2 ? ps[ps.size() / 2] : 0.5 * ((double)ps[ps.size() / 2 - 1] + ps[ps.size() / 2]);
      int best = ads[0], bc = 0;   // most common, first occurrence on ties
      for (int a : ads) { int cnt = (int)std::count(ads.begin(), ads.end(), a); if (cnt > bc) { bc = cnt; best = a; } }
      Item it; prep_peaks(mm.data(), ii.data(), (int)mm.size(), pm, it.mz, it.it);
      it.prec = (float)pm; it.ad = best; it.ce = 0; it.ce_known = 0; it.merged = std::min((int)grp.size(), 8) / 4.0f; it.mode = md;
      forward(M, it, nth, z.data()); nfw++;
      for (int b = 0; b < M.nbits; b++) zm[b] += z[b];
      cm++;
    }
    for (int b = 0; b < M.nbits; b++) mol[b] = 0.5f * (zs[b] / cs + zm[b] / cm);
    fwrite(&kv.first, 4, 1, fo); fwrite(mol.data(), 4, M.nbits, fo);
    done++;
    if (done % 50 == 0) fprintf(stderr, "  %d/%d mols\n", done, nmol);
  }
  fclose(fo);
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  fprintf(stderr, "%d molecules, %d forward passes, %.1f s, %.1f ms/pass (threads %d)\n", done, nfw, sec, 1000 * sec / nfw, nth);
  if (wspec) { FILE* f = fopen((outp + "_spec.bin").c_str(), "wb"); int n = (int)ns; fwrite(&n, 4, 1, f); fwrite(&M.nbits, 4, 1, f); fwrite(specz.data(), 4, specz.size(), f); fclose(f); }
  return 0;
}
