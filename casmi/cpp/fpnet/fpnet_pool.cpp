// FPNet pool scoring: for every query molecule, score each pool candidate within +-ppm of the
// adduct-corrected neutral mass (same window / adduct rule as phys_ana.cpp) as cand_fp . logits,
// where logits = fpnet_ggml's molecule logits (<out>_mol.bin) and cand_fp = the candidate's packed
// 10226-bit fingerprint row (np.packbits order) from pool730_fp10226.bin (row = line of <pool>_mols.txt).
// Only the needed fingerprint rows are read (seek), not the whole 933 MB file.
// Build: g++ -O2 -std=c++17 -o fpnet_pool fpnet_pool.cpp
// usage: fpnet_pool <dir> <qry_stem> <mol_logits.bin> <pool_stem> <pool_fp.bin> <ppm> <out.txt> [dot|tan]
//   dot (default): cand_fp . logits.  tan: continuous Tanimoto sum_{b in cand} p_b / (|cand| + sum p - sum_{b in cand} p_b),
//   p = sigmoid(logits), the same form as the CNN/FT pool terms (models38_STATUS: +0.005..0.01 over dot on the timsTOF holdouts).
//   writes "<molecule_id>\t<pool name>\t<score>" per (query, candidate); candidates whose fingerprint
//   could not be computed (<pool_fp.bin>.ok byte 0) are skipped (phys_fp gives them the query mean).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>
#ifdef _WIN32
#define FSEEK64 _fseeki64
#else
#define FSEEK64 fseeko
#endif

static std::vector<char> slurp(const std::string& p, bool must = true) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) { if (must) { fprintf(stderr, "cannot read %s\n", p.c_str()); exit(1); } return {}; }
  std::streamsize n = f.tellg(); f.seekg(0);
  std::vector<char> b((size_t)n); f.read(b.data(), n); return b;
}
static std::vector<std::string> names(const std::string& p) {
  std::ifstream f(p); std::vector<std::string> v; std::string l;
  if (!f) { fprintf(stderr, "cannot read %s\n", p.c_str()); exit(1); }
  while (std::getline(f, l)) { if (!l.empty() && l.back() == '\r') l.pop_back(); v.push_back(l.substr(0, l.find('\t'))); }
  return v;
}

constexpr float ADDUCT_SHIFT[10] = {
    1.007276f, -1.007276f, 22.989218f, 18.033823f, 38.963158f,
    34.969402f, 44.998201f, -17.002740f, 1.007276f, 22.989218f,
};
static int adductIdx(int32_t tag) {
  if (tag < 0) return -1;
  return tag % 16 == 15 ? tag / 16 + 1 : tag / 16;
}

int main(int argc, char** argv) {
  if (argc < 8) { fprintf(stderr, "usage: fpnet_pool dir qry_stem mol_logits.bin pool_stem pool_fp.bin ppm out.txt\n"); return 1; }
  std::string dir = argv[1], qs = argv[2], logp = argv[3], ps = argv[4], fpp = argv[5], outp = argv[7];
  float ppm = (float)atof(argv[6]);
  const bool tan = argc > 8 && strcmp(argv[8], "tan") == 0;
  auto t0 = std::chrono::steady_clock::now();
  struct Head { float prec; int32_t mol, off, n, tag; };
  std::vector<char> hb = slurp(dir + "/" + qs + "_head.bin");
  size_t ns = hb.size() / sizeof(Head); const Head* hd = (const Head*)hb.data();
  std::vector<std::string> qn = names(dir + "/" + qs + "_mols.txt");
  std::vector<std::string> pn = names(dir + "/" + ps + "_mols.txt");
  std::vector<char> pb = slurp(dir + "/" + ps + "_head.bin");
  size_t np = pb.size() / 12;
  std::vector<float> pmass(np); std::vector<int32_t> pid(np);
  for (size_t i = 0; i < np; i++) { memcpy(&pmass[i], pb.data() + 12 * i, 4); memcpy(&pid[i], pb.data() + 12 * i + 4, 4); }
  std::vector<char> ok = slurp(fpp + ".ok", false);
  std::vector<char> lb = slurp(logp);
  int nmol, nbits; memcpy(&nmol, lb.data(), 4); memcpy(&nbits, lb.data() + 4, 4);
  const size_t nby = ((size_t)nbits + 7) / 8;
  std::map<int, std::vector<int>> specOf;
  for (size_t s = 0; s < ns; s++) specOf[hd[s].mol].push_back((int)s);
  FILE* fp = fopen(fpp.c_str(), "rb");
  if (!fp) { fprintf(stderr, "cannot read %s\n", fpp.c_str()); return 1; }
  FILE* fo = fopen(outp.c_str(), "w");
  if (!fo) { fprintf(stderr, "cannot write %s\n", outp.c_str()); return 1; }
  std::vector<unsigned char> row(nby);
  std::vector<float> p(tan ? nbits : 0);
  size_t nout = 0, nskip = 0, nq0 = 0;
  for (int k = 0; k < nmol; k++) {
    const char* rec = lb.data() + 8 + (size_t)k * (4 + 4 * (size_t)nbits);
    int m; memcpy(&m, rec, 4); const float* z = (const float*)(rec + 4);
    auto it = specOf.find(m);
    if (it == specOf.end() || m < 0 || (size_t)m >= qn.size()) continue;
    double sp = 0;
    if (tan)
      for (int b = 0; b < nbits; b++) {
        float v = std::min(30.f, std::max(-30.f, z[b]));
        p[b] = 1.f / (1.f + std::exp(-v));
        sp += p[b];
      }
    std::set<int> cs;
    for (int s : it->second) {
      const Head& h = hd[s];
      int a = adductIdx(h.tag);
      float shift = (a >= 0 && a < 10) ? ADDUCT_SHIFT[a] : 1.007276f;
      float neutral = h.prec - shift;
      if (a == 8 || a == 9) neutral *= 0.5f;
      float w = neutral * ppm * 1e-6f;
      size_t lo = std::lower_bound(pmass.begin(), pmass.end(), neutral - w) - pmass.begin();
      size_t hi = std::upper_bound(pmass.begin(), pmass.end(), neutral + w) - pmass.begin();
      for (size_t j = lo; j < hi; j++) cs.insert(pid[j]);
    }
    if (cs.empty()) nq0++;
    for (int c : cs) {
      if (c < 0 || (size_t)c >= pn.size()) continue;
      if (!ok.empty() && ((size_t)c >= ok.size() || ok[c] == 0)) { nskip++; continue; }
      if (FSEEK64(fp, (int64_t)c * (int64_t)nby, SEEK_SET) != 0 || fread(row.data(), 1, nby, fp) != nby) { nskip++; continue; }
      double sc = 0;
      int nb = 0;
      for (size_t by = 0; by < nby; by++) {
        unsigned v = row[by];
        if (!v) continue;
        for (int bit = 0; bit < 8; bit++)  // packbits: MSB first
          if (v & (0x80u >> bit)) {
            size_t b = by * 8 + (size_t)bit;
            if (b < (size_t)nbits) { sc += tan ? p[b] : z[b]; nb++; }
          }
      }
      if (tan) sc = sc / (nb + sp - sc + 1e-9);
      fprintf(fo, tan ? "%s\t%s\t%.6f\n" : "%s\t%s\t%.4f\n", qn[m].c_str(), pn[c].c_str(), sc);
      nout++;
    }
  }
  fclose(fo); fclose(fp);
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  fprintf(stderr, "fpnet_pool: %d query molecules, %zu scored candidates, %zu skipped (no fp), %zu queries with no candidate, %.1f s\n",
          nmol, nout, nskip, nq0, sec);
  return 0;
}
