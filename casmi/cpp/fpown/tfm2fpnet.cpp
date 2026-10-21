// tfm2fpnet: export a model trained by OUR trainer (cpp/rev/tfm_train.cpp, TFM1 file, fpnet=1 mode) into the
// FPN1 (f32) / FPN2 (2-D f16) file read by cpp/fpnet/fpnet_ggml.cpp, so the existing ggml inference works unchanged.
// The tensor list (117 tensors, names, torch dims, order) is hard-coded from the FPNet module definition; no third-party
// weight file is read. See fpnet_own_STATUS.md for the mapping.
//
//   tfm2fpnet <model.bin> <out_f32.bin> <out_f16.bin> [instr=5]
//
// Mapping (ours row-major [in,out] -> torch [out,in]):
//   pk.weight[:, 0:2d]   = wtok rows 0..2d-1 (SinEmb mz | SinEmb nl),  pk.weight[:, 2d] = wtok row 2d (it),
//   pk.weight[:, 2d+1]   = wtok row 2d+1 (it^2), rest 0;  int_emb = identity on its first 2 outputs (weight e0,e1; bias 0)
//   gl.weight[:, 0:d]    = wg rows 0..d-1 (SinEmb prec), [d+d/4+2] (mode) = wg row d+1, [d+d/4+3] (log1p/10) = wg row d,
//                          ce_emb / ce_known / merged columns 0 (our data has no CE; merged spectra are treated as spectra)
//   ad.weight[row]       = ead[our adduct] + ein[instr]  (rows OUR2CHEM[a]; every other row incl. <unk>=59 gets ead[10])
//   SinEmb inv           = 2 pi / wav (our iwav = 1/wav); ce_emb.inv = FPNet's lo 0 hi 2.2 (unused: zero columns)
//   blocks / norm / head : transposes; elem_head.* = 0 (not used by fpnet_ggml)
#include "parallel.hpp"
#include "tfm_model.hpp"

#include <cstdio>
#include <iterator>
#include <string>
#include <vector>

static const int OUR2CHEM[10] = {0, 6, 4, 1, 5, 9, 8, 2, 46, 47};

static uint16_t f2h(float f) {  // IEEE half, round to nearest even
  uint32_t x; std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t e = int32_t((x >> 23) & 0xff) - 127 + 15;
  uint32_t m = x & 0x7fffffu;
  if (((x >> 23) & 0xff) == 0xff) return uint16_t(sign | 0x7c00u | (m ? 0x200u : 0));
  if (e >= 31) return uint16_t(sign | 0x7c00u);
  if (e <= 0) {
    if (e < -10) return uint16_t(sign);
    m |= 0x800000u;
    int sh = 14 - e;
    uint32_t hm = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
    if (rem > half || (rem == half && (hm & 1))) ++hm;
    return uint16_t(sign | hm);
  }
  uint32_t hm = m >> 13, rem = m & 0x1fffu;
  uint32_t h = sign | (uint32_t(e) << 10) | hm;
  if (rem > 0x1000u || (rem == 0x1000u && (hm & 1))) ++h;
  return uint16_t(h);
}
static float h2f(uint16_t h) {
  uint32_t sign = uint32_t(h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ffu, x;
  if (e == 0) {
    if (m == 0) x = sign;
    else { int k = -1; do { ++k; m <<= 1; } while (!(m & 0x400u)); x = sign | (uint32_t(127 - 15 - k) << 23) | ((m & 0x3ffu) << 13); }
  } else if (e == 31) x = sign | 0x7f800000u | (m << 13);
  else x = sign | ((e - 15 + 127) << 23) | (m << 13);
  float f; std::memcpy(&f, &x, 4); return f;
}

struct T { std::string name; std::vector<int> dims; std::vector<float> v; };

int main(int argc, char** argv) {
  if (argc < 4) { std::fprintf(stderr, "usage: tfm2fpnet model.bin out_f32.bin out_f16.bin [instr=5]\n"); return 1; }
  const int instr = argc > 4 ? std::atoi(argv[4]) : 5;
  tfm::Model m;
  if (!m.load(argv[1], false)) { std::fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
  const tfm::Cfg& c = m.c;
  const int d = c.d;
  if (c.demb != d || c.ff != 4 || c.HH != 2048 || c.nAd != 11 || c.nIn != 6 || d % c.H) {
    std::fprintf(stderr, "config not FPNet-shaped: d %d demb %d ff %d HH %d nAd %d nIn %d\n", d, c.demb, c.ff, c.HH, c.nAd, c.nIn);
    return 1;
  }
  std::vector<float> W(m.nP);
  { std::vector<tfm::real> h(m.nP); tfm::download(m.W, h.data(), m.nP); for (int64_t i = 0; i < m.nP; ++i) W[i] = float(h[i]); }
  auto P = [&](const tfm::PRef& p) { return W.data() + p.off; };
  std::vector<T> out;
  auto add = [&](const std::string& n, std::vector<int> dims) -> std::vector<float>& {
    size_t k = 1; for (int x : dims) k *= size_t(x);
    out.push_back({n, dims, std::vector<float>(k, 0.f)});
    return out.back().v;
  };
  auto vec = [&](const std::string& n, const tfm::PRef& p) {  // 1-D copy
    auto& v = add(n, {int(p.n)}); std::copy(P(p), P(p) + p.n, v.begin());
  };
  auto tr = [&](const std::string& n, const tfm::PRef& p) {  // ours [in=rows, out=cols] -> torch [out, in]
    auto& v = add(n, {p.cols, p.rows});
    const float* s = P(p);
    for (int i = 0; i < p.rows; ++i) for (int o = 0; o < p.cols; ++o) v[size_t(o) * p.rows + i] = s[size_t(i) * p.cols + o];
  };
  std::vector<double> iw(c.demb / 2);  // the trainer's own 1/wav table -> inv = float(2 pi * iwav), bit-identical
  tfm::download(m.iwav, iw.data(), int64_t(iw.size()));  // to the angle the trainer forms with fpnetPhase()
  auto invOwn = [&](const std::string& n) {
    auto& v = add(n, {int(iw.size())});
    for (size_t i = 0; i < iw.size(); ++i) v[i] = float(6.283185307179586 * iw[i]);
  };
  auto inv = [&](const std::string& n, int nn, double lo, double hi) {
    auto& v = add(n, {nn});
    for (int i = 0; i < nn; ++i) {
      double t = nn > 1 ? double(i) / double(nn - 1) : 0.0;
      v[i] = float(2.0 * 3.14159265358979323846 / std::pow(10.0, (hi - lo) * t + lo));
    }
  };
  invOwn("mz_emb.inv");
  invOwn("nl_emb.inv");
  { auto& v = add("int_emb.weight", {d, 2}); v[0 * 2 + 0] = 1.f; v[1 * 2 + 1] = 1.f; }
  add("int_emb.bias", {d});
  {
    auto& v = add("pk.weight", {d, 3 * d});
    const float* s = P(m.wtok);  // [2d+2, d]
    for (int j = 0; j < d; ++j) {
      for (int k = 0; k < 2 * d; ++k) v[size_t(j) * 3 * d + k] = s[size_t(k) * d + j];
      v[size_t(j) * 3 * d + 2 * d] = s[size_t(2 * d) * d + j];
      v[size_t(j) * 3 * d + 2 * d + 1] = s[size_t(2 * d + 1) * d + j];
    }
  }
  vec("pk.bias", m.btok);
  invOwn("prec_emb.inv");
  {
    auto& v = add("ad.weight", {60, d});
    const float* ea = P(m.ead); const float* ei = P(m.ein) + size_t(instr) * d;
    for (int r = 0; r < 60; ++r) for (int j = 0; j < d; ++j) v[size_t(r) * d + j] = ea[size_t(10) * d + j] + ei[j];
    for (int a = 0; a < 10; ++a) for (int j = 0; j < d; ++j) v[size_t(OUR2CHEM[a]) * d + j] = ea[size_t(a) * d + j] + ei[j];
  }
  inv("ce_emb.inv", d / 8, 0.0, 2.2);
  {
    const int G = d + d / 4 + 4;
    auto& v = add("gl.weight", {d, G});
    const float* s = P(m.wg);  // [d+2, d]: SinEmb(prec) d | log1p/10 | mode
    for (int j = 0; j < d; ++j) {
      for (int k = 0; k < d; ++k) v[size_t(j) * G + k] = s[size_t(k) * d + j];
      v[size_t(j) * G + d + d / 4 + 2] = s[size_t(d + 1) * d + j];
      v[size_t(j) * G + d + d / 4 + 3] = s[size_t(d) * d + j];
    }
  }
  vec("gl.bias", m.bg);
  for (int l = 0; l < c.L; ++l) {
    const tfm::BlockP& b = m.blk[l];
    std::string p = "blocks." + std::to_string(l) + ".";
    vec(p + "n1.weight", b.ln1g); vec(p + "n1.bias", b.ln1b);
    tr(p + "qkv.weight", b.wqkv); vec(p + "qkv.bias", b.bqkv);
    tr(p + "o.weight", b.wo); vec(p + "o.bias", b.bo);
    vec(p + "n2.weight", b.ln2g); vec(p + "n2.bias", b.ln2b);
    tr(p + "ff.0.weight", b.w1); vec(p + "ff.0.bias", b.b1);
    tr(p + "ff.3.weight", b.w2); vec(p + "ff.3.bias", b.b2);
  }
  vec("norm.weight", m.lnfg); vec("norm.bias", m.lnfb);
  tr("head.0.weight", m.wh1); vec("head.0.bias", m.bh1);
  tr("head.3.weight", m.wh2); vec("head.3.bias", m.bh2);
  add("elem_head.0.weight", {256, 2 * d}); add("elem_head.0.bias", {256});
  add("elem_head.2.weight", {10, 256}); add("elem_head.2.bias", {10});

  for (int f16 = 0; f16 < 2; ++f16) {
    const char* path = argv[2 + f16];
    FILE* f = std::fopen(path, "wb");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", path); return 1; }
    std::fwrite(f16 ? "FPN2" : "FPN1", 1, 4, f);
    int32_t hdr[5] = {c.NB, d, c.L, c.H, int32_t(out.size())};
    std::fwrite(hdr, 4, 5, f);
    for (auto& t : out) {
      int32_t nl = int32_t(t.name.size()), nd = int32_t(t.dims.size());
      std::fwrite(&nl, 4, 1, f); std::fwrite(t.name.data(), 1, nl, f);
      std::fwrite(&nd, 4, 1, f); std::fwrite(t.dims.data(), 4, nd, f);
      if (f16 && nd == 2) {
        std::vector<uint16_t> h(t.v.size());
        for (size_t i = 0; i < h.size(); ++i) h[i] = f2h(t.v[i]);
        std::fwrite(h.data(), 2, h.size(), f);
      } else std::fwrite(t.v.data(), 4, t.v.size(), f);
    }
    std::fclose(f);
  }
  // round trip: re-read both files and compare with the in-memory tensors
  for (int f16 = 0; f16 < 2; ++f16) {
    std::ifstream f(argv[2 + f16], std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const char* p = b.data() + 24;
    double maxAbs = 0, maxRel = 0; size_t nbad = 0;
    for (auto& t : out) {
      int32_t nl; std::memcpy(&nl, p, 4); p += 4;
      if (std::string(p, nl) != t.name) { std::fprintf(stderr, "name mismatch %s\n", t.name.c_str()); return 1; }
      p += nl;
      int32_t nd; std::memcpy(&nd, p, 4); p += 4 + 4 * nd;
      for (size_t i = 0; i < t.v.size(); ++i) {
        float x;
        if (f16 && nd == 2) { uint16_t h; std::memcpy(&h, p + 2 * i, 2); x = h2f(h); }
        else std::memcpy(&x, p + 4 * i, 4);
        double e = std::fabs(double(x) - t.v[i]);
        if (!std::isfinite(x)) ++nbad;
        maxAbs = std::max(maxAbs, e);
        if (std::fabs(t.v[i]) > 1e-4) maxRel = std::max(maxRel, e / std::fabs(t.v[i]));
      }
      p += t.v.size() * ((f16 && nd == 2) ? 2 : 4);
    }
    std::printf("round trip %s: %zu tensors, %s, max|diff| %.3e, max rel (|w|>1e-4) %.3e, non-finite %zu\n", argv[2 + f16],
                out.size(), size_t(p - b.data()) == b.size() ? "size ok" : "SIZE MISMATCH", maxAbs, maxRel, nbad);
  }
  std::printf("exported: nbits %d d %d layers %d heads %d, instrument row %d folded into the adduct table\n", c.NB, d, c.L, c.H, instr);
  return 0;
}
