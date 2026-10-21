// cnn_gradcheck: finite-difference check of every parameter gradient of the CNN
// reverse model (cnn_model.hpp) on a tiny configuration, for both readouts
// (flat=1 and global-pool only). Builds on CPU (cl/g++) and CUDA (nvcc) from the
// same source; both must print PASS, and the losses at the same seed should
// agree between the two builds.
//   cnn_gradcheck [samplesPerTensor=60] [eps=3e-3] [seed=7]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "cnn_model.hpp"

using namespace cnn;

static double lossOf(Model& m, int B, const std::vector<real>& T) {
  m.forward(B, false, 0.f, 0);
  std::vector<real> Z;
  download(m.Zo, Z, int64_t(B) * m.NB);
  double L = 0;
  for (size_t i = 0; i < Z.size(); ++i) {
    double z = Z[i];
    double sp = z > 0 ? z + std::log1p(std::exp(-z)) : std::log1p(std::exp(z));
    L += sp - T[i] * z;
  }
  return L / B;
}

static int runCheck(int flat, int per, double eps, unsigned seed) {
  Model m;
  m.cfg.F = 4; m.cfg.off = 0.3f;
  m.cfg.Lf = 16; m.cfg.Ln = 8; m.cfg.Lp = 8;
  m.cfg.C0 = 3; m.cfg.k = 3; m.cfg.chans = {4, 5};
  m.cfg.flat = flat; m.cfg.H = {6};
  m.cfg.maxPeaks = 20; m.cfg.pairK = 6;
  m.NB = 7;
  m.build(seed, true);
  // larger random weights than the init so that every unit sees signal
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 0.6f);
  for (Param* p : m.params()) {
    std::vector<real> h(p->n);
    for (auto& x : h) x = nd(rng);
    upload(p->w, h);
  }
  const int B = 3;
  std::vector<Sample> ss(B);
  std::uniform_real_distribution<float> um(0.2f, 15.4f), ui(0.05f, 1.f);
  for (int b = 0; b < B; ++b) {
    int n = 9 + 3 * b;
    std::vector<float> mz(n), it(n);
    for (int i = 0; i < n; ++i) { mz[i] = um(rng); it[i] = ui(rng); }
    Head h{15.9f, 0, 0, n, (b * 16) + (b == 2 ? 15 : 1)};
    featurize(m.cfg, h, mz.data(), it.data(), ss[b]);
  }
  m.setBatch(ss, true);
  std::vector<real> T(size_t(B) * m.NB);
  for (auto& t : T) t = (rng() % 3 == 0) ? 1.f : 0.f;
  upload(m.T, T);
  double L0 = lossOf(m, B, T);
  m.backward(B, 1.f, 0.f);
  std::printf("[flat=%d] params %lld, head input %lld, loss %.8f  (items frag %zu nl %zu pair %zu in sample 0)\n", flat,
              (long long)m.nParams(), (long long)m.D(), L0, ss[0].frag.size(), ss[0].nl.size(), ss[0].pair.size());
  auto ps = m.params();
  const char* names[] = {"frag.stem.W", "frag.stem.b", "frag.conv0.W", "frag.conv0.b", "frag.conv1.W", "frag.conv1.b",
                         "loss.stem.W", "loss.stem.b", "loss.conv0.W", "loss.conv0.b", "loss.conv1.W", "loss.conv1.b",
                         "head0.W", "head0.b", "head1.W", "head1.b"};
  int fails = 0, totKinks = 0;
  const double relTol = sizeof(real) == 8 ? 1e-5 : 2e-2, absTol = sizeof(real) == 8 ? 1e-8 : 2e-4;
  double worst = 0;
  for (size_t pi = 0; pi < ps.size(); ++pi) {
    Param* p = ps[pi];
    std::vector<real> w, g;
    download(p->w, w, p->n);
    download(p->g, g, p->n);
    std::vector<int64_t> idx;
    if (p->n <= per) for (int64_t i = 0; i < p->n; ++i) idx.push_back(i);
    else for (int t = 0; t < per; ++t) idx.push_back(int64_t(rng() % uint32_t(p->n)));
    double maxRel = 0, maxAbs = 0, sumG = 0;
    int nz = 0, bad = 0, kinks = 0;
    for (int64_t i : idx) {
      real o = w[i];
      w[i] = o + eps; upload(p->w, w); double lp = lossOf(m, B, T);
      w[i] = o - eps; upload(p->w, w); double lm = lossOf(m, B, T);
      w[i] = o; upload(p->w, w);
      double num = (lp - lm) / (2.0 * eps), an = g[i];
      // a ReLU / max-pool kink inside [w-eps, w+eps] makes the one-sided slopes disagree:
      // such points are not differentiable there, count them separately and skip
      double fwd = (lp - L0) / eps, bwd = (L0 - lm) / eps;
      if (std::fabs(fwd - bwd) > 0.05 * (std::fabs(fwd) + std::fabs(bwd)) + (sizeof(real) == 8 ? 1e-6 : 2e-3)) { ++kinks; continue; }
      double ad = std::fabs(num - an), rel = ad / std::max(1e-3, std::fabs(num) + std::fabs(an));
      maxAbs = std::max(maxAbs, ad);
      if (std::fabs(an) > 1e-6) ++nz;
      sumG += std::fabs(an);
      if (rel > relTol && ad > absTol) {
        ++bad;
        if (bad <= 3) std::printf("    MISMATCH %s[%lld]: analytic %.6g numeric %.6g\n", names[pi], (long long)i, an, num);
      }
      if (ad > absTol) maxRel = std::max(maxRel, rel);
    }
    worst = std::max(worst, maxRel);
    fails += bad;
    totKinks += kinks;
    std::printf("  %-13s n %7lld checked %3zu (kinks %2d) nonzero %3d  mean|g| %.3e  max|diff| %.2e  max rel %.2e %s\n",
                names[pi], (long long)p->n, idx.size(), kinks, nz, sumG / idx.size(), maxAbs, maxRel, bad ? "FAIL" : "ok");
  }
  std::printf("[flat=%d] %s (%d mismatches, worst rel %.2e, %d kink points skipped)\n", flat, fails ? "FAIL" : "PASS",
              fails, worst, totKinks);
  return fails;
}

int main(int argc, char** argv) {
  int per = argc > 1 ? std::atoi(argv[1]) : 60;
  double eps = argc > 2 ? std::atof(argv[2]) : (sizeof(real) == 8 ? 1e-6 : 3e-3);
  unsigned seed = argc > 3 ? unsigned(std::atoi(argv[3])) : 7u;
  std::printf("backend: %s\n", bk::backend_name());
  int f = runCheck(1, per, eps, seed) + runCheck(0, per, eps, seed + 1);
  std::printf("%s\n", f ? "GRADCHECK FAIL" : "GRADCHECK PASS");
  return f ? 1 : 0;
}
