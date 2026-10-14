// ana_dump: standalone analog search (same as phys_ana stage 1 + 2, variant 4:
// modified cosine with >= 6 matched peaks), dumping the top analog molecules
// per query molecule for the generation recall analysis.
//   ana_dump <dir> <qry> <lib> <K> <top> <out.txt> [maxDelta]
// out: "Q <name> <nspec> <neutral>" then "<libmol> <mcos> <nmatch> <libprec> <libadduct> <qprec>"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

struct Head { float precursor; int32_t mol, off, n, tag; };
static std::vector<char> slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(1); }
  std::streamsize n = f.tellg(); f.seekg(0);
  std::vector<char> b((size_t)n); f.read(b.data(), n); return b;
}
constexpr float ADDUCT_SHIFT[10] = {1.007276f, -1.007276f, 22.989218f, 18.033823f, 38.963158f,
                                    34.969402f, 44.998201f, -17.002740f, 1.007276f, 22.989218f};
static int adductIdx(int32_t t) { if (t < 0) return -1; return t % 16 == 15 ? t / 16 + 1 : t / 16; }

struct PairT { float s; int i, j; };
static void loadSpec(const std::string& dir, const std::string& st, std::vector<Head>& h, std::vector<float>& mz,
                     std::vector<float>& it, std::vector<std::string>& names) {
  auto hb = slurp(dir + "/" + st + "_head.bin");
  h.resize(hb.size() / sizeof(Head)); std::memcpy(h.data(), hb.data(), hb.size());
  auto pb = slurp(dir + "/" + st + "_peaks.bin");
  const float* p = (const float*)pb.data();
  size_t tot = 0; for (auto& x : h) tot += x.n;
  mz.reserve(tot); it.reserve(tot);
  for (auto& x : h) { x.off = (int32_t)mz.size(); mz.insert(mz.end(), p, p + x.n); p += x.n; it.insert(it.end(), p, p + x.n); p += x.n; }
  std::ifstream f(dir + "/" + st + "_mols.txt"); std::string line;
  while (std::getline(f, line)) { if (!line.empty() && line.back() == '\r') line.pop_back(); names.push_back(line.substr(0, line.find('\t'))); }
}

int main(int argc, char** argv) {
  if (argc < 7) { std::fprintf(stderr, "usage\n"); return 1; }
  std::string dir = argv[1], qs = argv[2], ls = argv[3], outp = argv[6];
  int shortK = std::atoi(argv[4]), top = std::atoi(argv[5]);
  float maxD = argc > 7 ? (float)std::atof(argv[7]) : 250.f;
  std::vector<Head> head, lhead; std::vector<float> mz, it, lmz, lit; std::vector<std::string> qn, ln;
  loadSpec(dir, qs, head, mz, it, qn);
  loadSpec(dir, ls, lhead, lmz, lit, ln);
  std::printf("lib %zu spectra\n", lhead.size()); std::fflush(stdout);
  constexpr int NBIN = 300000;
  std::vector<int64_t> bs(NBIN + 1, 0), ns(NBIN + 1, 0);
  std::vector<float> lnrm(lhead.size());
  for (auto& h : lhead) for (int j = 0; j < h.n; ++j) {
    int b = int(lmz[h.off + j] * 100.f); if (b >= 0 && b < NBIN) ++bs[b + 1];
    int c = int((h.precursor - lmz[h.off + j]) * 100.f + 0.5f); if (c >= 0 && c < NBIN) ++ns[c + 1];
  }
  for (int b = 0; b < NBIN; ++b) { bs[b + 1] += bs[b]; ns[b + 1] += ns[b]; }
  std::vector<int32_t> ps(bs[NBIN]), nsp(ns[NBIN]); std::vector<float> pw(bs[NBIN]), nw(ns[NBIN]);
  {
    std::vector<int64_t> f1(bs.begin(), bs.end() - 1), f2(ns.begin(), ns.end() - 1);
    for (size_t s = 0; s < lhead.size(); ++s) {
      auto& h = lhead[s]; double nrm = 0;
      for (int j = 0; j < h.n; ++j) nrm += std::max(0.f, lit[h.off + j]);
      nrm = std::sqrt(nrm) + 1e-12; lnrm[s] = (float)nrm;
      for (int j = 0; j < h.n; ++j) {
        float w = float(std::sqrt(std::max(0.f, lit[h.off + j])) / nrm);
        int b = int(lmz[h.off + j] * 100.f); if (b >= 0 && b < NBIN) { ps[f1[b]] = (int32_t)s; pw[f1[b]++] = w; }
        int c = int((h.precursor - lmz[h.off + j]) * 100.f + 0.5f); if (c >= 0 && c < NBIN) { nsp[f2[c]] = (int32_t)s; nw[f2[c]++] = w; }
      }
    }
  }
  std::printf("indexed\n"); std::fflush(stdout);
  std::unordered_map<int, std::vector<int>> specOf;
  for (size_t s = 0; s < head.size(); ++s) specOf[head[s].mol].push_back((int)s);
  std::vector<std::pair<int, std::vector<int>>> Q(specOf.begin(), specOf.end());
  std::sort(Q.begin(), Q.end(), [&](auto& a, auto& b) { return qn[a.first] < qn[b.first]; });
  std::vector<std::string> outS(Q.size());
  std::atomic<size_t> nx(0);
  std::vector<std::thread> th;
  unsigned NT = std::max(1u, std::thread::hardware_concurrency());
  for (unsigned t = 0; t < NT; ++t) th.emplace_back([&]() {
    std::vector<float> acc(lhead.size(), 0.f), acc2(lhead.size(), 0.f); std::vector<uint8_t> mark(lhead.size(), 0);
    std::vector<int32_t> touched; std::vector<float> qw; std::vector<std::pair<float, int32_t>> sh;
    std::vector<PairT> pairs; std::vector<uint8_t> uq, ul;
    for (;;) {
      size_t qi = nx++; if (qi >= Q.size()) break;
      struct Best { float s; int nm; float lp; int la; float qp; };
      std::unordered_map<int, Best> mb;
      float neutral = 0.f;
      for (int s : Q[qi].second) {
        const Head& h = head[s];
        int a = adductIdx(h.tag);
        if (neutral == 0.f && a >= 0 && a < 8) neutral = h.precursor - ADDUCT_SHIFT[a];
        double nrm = 0; for (int i = 0; i < h.n; ++i) nrm += std::max(0.f, it[h.off + i]);
        nrm = std::sqrt(nrm) + 1e-12; touched.clear(); qw.resize(h.n);
        for (int i = 0; i < h.n; ++i) {
          float wq = float(std::sqrt(std::max(0.f, it[h.off + i])) / nrm); qw[i] = wq;
          float x = mz[h.off + i] * 100.f; int b = int(x); int b2 = (x - b) < 0.5f ? b - 1 : b + 1;
          for (int bb : {b, b2}) { if (bb < 0 || bb >= NBIN) continue;
            for (int64_t k = bs[bb]; k < bs[bb + 1]; ++k) { int32_t l = ps[k]; if (!mark[l]) { mark[l] = 1; touched.push_back(l); } acc[l] += wq * pw[k]; } }
          float y = (h.precursor - mz[h.off + i]) * 100.f + 0.5f; if (y < 0.f) continue;
          int c = int(y); int c2 = (y - c) < 0.5f ? c - 1 : c + 1;
          for (int bb : {c, c2}) { if (bb < 0 || bb >= NBIN) continue;
            for (int64_t k = ns[bb]; k < ns[bb + 1]; ++k) { int32_t l = nsp[k]; if (!mark[l]) { mark[l] = 1; touched.push_back(l); } acc2[l] += wq * nw[k]; } }
        }
        sh.clear();
        for (int32_t l : touched) {
          float a1 = acc[l], a2 = acc2[l]; acc[l] = acc2[l] = 0.f; mark[l] = 0;
          if (std::fabs(lhead[l].precursor - h.precursor) > maxD) continue;
          if (a1 + a2 >= 0.2f) sh.emplace_back(a1 + a2, l);
        }
        if ((int)sh.size() > shortK) { std::nth_element(sh.begin(), sh.begin() + shortK, sh.end(), [](auto& a, auto& b) { return a.first > b.first; }); sh.resize(shortK); }
        for (auto& sl : sh) {
          const Head& lh = lhead[sl.second];
          const float* lm = lmz.data() + lh.off; const float* li = lit.data() + lh.off;
          float delta = h.precursor - lh.precursor; pairs.clear();
          for (int pass = 0; pass < 2; ++pass) {
            float shf = pass ? delta : 0.f; if (pass && std::fabs(delta) <= 0.01f) break;
            int j0 = 0;
            for (int i = 0; i < h.n; ++i) { float m = mz[h.off + i] - shf;
              while (j0 < lh.n && lm[j0] < m - 0.01f) ++j0;
              for (int j = j0; j < lh.n && lm[j] <= m + 0.01f; ++j) pairs.push_back({qw[i] * std::sqrt(std::max(0.f, li[j])) / lnrm[sl.second], i, j}); }
          }
          std::sort(pairs.begin(), pairs.end(), [](auto& a, auto& b) { return a.s > b.s; });
          uq.assign(h.n, 0); ul.assign(lh.n, 0); double sm = 0; int nm = 0;
          for (auto& p : pairs) if (!uq[p.i] && !ul[p.j]) { uq[p.i] = ul[p.j] = 1; sm += p.s; ++nm; }
          float v = (float)std::min(1.0, sm);
          if (nm < 6 || v < 0.3f) continue;
          auto f = mb.find(lh.mol);
          if (f == mb.end() || f->second.s < v) mb[lh.mol] = {v, nm, lh.precursor, adductIdx(lh.tag), h.precursor};
        }
      }
      std::vector<std::pair<float, int>> tp; for (auto& kv : mb) tp.emplace_back(kv.second.s, kv.first);
      std::sort(tp.begin(), tp.end(), [](auto& a, auto& b) { return a.first > b.first; });
      if ((int)tp.size() > top) tp.resize(top);
      std::string o; char buf[256];
      std::snprintf(buf, sizeof buf, "Q %s %zu %.5f %zu\n", qn[Q[qi].first].c_str(), Q[qi].second.size(), neutral, tp.size()); o += buf;
      for (auto& x : tp) { auto& b = mb[x.second];
        std::snprintf(buf, sizeof buf, "%d %.4f %d %.5f %d %.5f\n", x.second, b.s, b.nm, b.lp, b.la, b.qp); o += buf; }
      outS[qi] = o;
    }
  });
  for (auto& t : th) t.join();
  std::ofstream f(outp); for (auto& s : outS) f << s;
  std::printf("done %zu queries\n", Q.size());
}
