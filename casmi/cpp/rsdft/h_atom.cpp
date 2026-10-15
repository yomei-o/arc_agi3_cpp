// Hydrogen atom on a real-space grid: the Schrodinger equation as a difference equation.
//
//   (-1/2 laplacian - 1/r) psi = E psi        (atomic units: Hartree, bohr)
//
// Space is cut into a cubic lattice of spacing h. The Laplacian becomes the
// 7-point stencil (psi[x+h] + psi[x-h] + ... - 6 psi[x]) / h^2 - a spatial FIR
// filter. H is then a huge sparse matrix and the ground state its lowest
// eigenvector.
//
// Imaginary-time iteration finds it:  psi <- (1 - tau H) psi, then normalise.
// An eigencomponent with energy E is multiplied by (1 - tau E) every step, so
// relative to the ground state the excited ones decay geometrically - the same
// thing a recursive (IIR) low-pass filter does to high frequencies. Stability
// needs |1 - tau E_max| < 1 with E_max ~ 6/h^2 (the stencil's highest
// "frequency"), so tau = 0.9 * h^2 / 3; the number of steps grows like 1/h^2.
//
// The nucleus sits at the centre of a lattice cell, so no grid point has r = 0.
// Expected: E -> -0.5 as h -> 0 (error ~ h^2 for the stencil, plus the cusp at
// the nucleus); Richardson extrapolation of two spacings estimates the limit.
//
// Build:  g++ -O3 -fopenmp -o h_atom h_atom.cpp
// Usage:  h_atom [box_bohr=16] [h list...]

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct Result { double h, E, r_mean; int steps; };

Result solve(double L, double h) {
  const int n = int(std::lround(L / h));
  const size_t N = size_t(n) * n * n;
  auto id = [n](int i, int j, int k) { return (size_t(i) * n + j) * n + k; };
  const double c = 0.5 * (n - 1) * h;  // box centre = centre of a cell when n is even
  std::vector<double> V(N), psi(N), Hpsi(N);
  #pragma omp parallel for
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k) {
        double x = i * h - c, y = j * h - c, z = k * h - c;
        double r = std::sqrt(x * x + y * y + z * z);
        V[id(i, j, k)] = -1.0 / r;
        psi[id(i, j, k)] = std::exp(-0.5 * r);  // any positive start works; this one is close
      }
  const double tau = 0.9 * h * h / 3.0, inv_h2 = 1.0 / (h * h);
  // psi = 0 outside the box (Dirichlet): the stencil just skips missing neighbours
  auto applyH = [&](const std::vector<double>& p, std::vector<double>& out) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        for (int k = 0; k < n; ++k) {
          size_t q = id(i, j, k);
          double s = -6.0 * p[q];
          if (i > 0) s += p[q - size_t(n) * n];
          if (i < n - 1) s += p[q + size_t(n) * n];
          if (j > 0) s += p[q - n];
          if (j < n - 1) s += p[q + n];
          if (k > 0) s += p[q - 1];
          if (k < n - 1) s += p[q + 1];
          out[q] = -0.5 * s * inv_h2 + V[q] * p[q];
        }
  };
  auto dot = [&](const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0;
    #pragma omp parallel for reduction(+ : s)
    for (long q = 0; q < long(N); ++q) s += a[q] * b[q];
    return s;
  };
  double E = 0, Eprev = 1e9;
  int step = 0;
  for (; step < 200000; ++step) {
    applyH(psi, Hpsi);
    if (step % 50 == 0) {
      E = dot(psi, Hpsi) / dot(psi, psi);  // Rayleigh quotient
      if (std::fabs(E - Eprev) < 1e-9) break;
      Eprev = E;
    }
    #pragma omp parallel for
    for (long q = 0; q < long(N); ++q) psi[q] -= tau * Hpsi[q];
    double nrm = std::sqrt(dot(psi, psi) * h * h * h);
    #pragma omp parallel for
    for (long q = 0; q < long(N); ++q) psi[q] /= nrm;
  }
  double rm = 0, w = 0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k) {
        double x = i * h - c, y = j * h - c, z = k * h - c, p2 = psi[id(i, j, k)] * psi[id(i, j, k)];
        rm += std::sqrt(x * x + y * y + z * z) * p2; w += p2;
      }
  return {h, E, rm / w, step};
}

int main(int argc, char** argv) {
  double L = argc > 1 ? std::atof(argv[1]) : 16.0;
  std::vector<double> hs;
  for (int a = 2; a < argc; ++a) hs.push_back(std::atof(argv[a]));
  if (hs.empty()) hs = {0.4, 0.3, 0.2, 0.15, 0.1};
  std::printf("hydrogen atom, box %.1f bohr; exact E = -0.5 Hartree, <r> = 1.5 bohr\n", L);
  std::printf("%6s %8s %12s %10s %8s %8s\n", "h", "points", "E", "error", "<r>", "steps");
  std::vector<Result> res;
  for (double h : hs) {
    Result r = solve(L, h);
    int n = int(std::lround(L / h));
    std::printf("%6.3f %8d %12.6f %10.6f %8.4f %8d\n", h, n * n * n, r.E, r.E + 0.5, r.r_mean, r.steps);
    std::fflush(stdout);
    if (!res.empty()) {
      const Result& a = res.back();  // E(h) = E0 + C h^2  ->  E0 from two spacings
      double E0 = (a.E * h * h - r.E * a.h * a.h) / (h * h - a.h * a.h);
      std::printf("        Richardson (h=%.3f,%.3f): E0 = %.6f  error %.6f\n", a.h, h, E0, E0 + 0.5);
    }
    res.push_back(r);
  }
}
