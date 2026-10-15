// H2+ on a real-space grid: the smallest chemical bond, as a difference equation.
//
// One electron, two protons at x = -R/2, +R/2:
//   (-1/2 laplacian - 1/|r - A| - 1/|r - B|) psi = E_el psi,   E(R) = E_el + 1/R
// Same 7-point stencil and imaginary-time iteration as h_atom.cpp. Scanning R
// gives the potential energy curve: the minimum is the bond length, the depth
// below H + H+ (-0.5) the binding energy.
// Exact (Born-Oppenheimer): R_e = 1.997 bohr, E(R_e) = -0.60263 Hartree, so the
// bond energy D_e = 0.10263 Hartree = 2.79 eV = 64.4 kcal/mol.
//
// Grid coordinates are odd multiples of h/2 (n even, box centred on 0), so the
// protons at y = z = 0 never sit on a grid point.
//
// Build:  g++ -O3 -fopenmp -o h2plus h2plus.cpp
// Usage:  h2plus [h=0.2] [box=20] [R list...]

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int SUB = 1;  // sub-points per cell edge for the cell-averaged potential (1 = plain)

double solve(double L, double h, double R, int* steps_out) {
  const int n = 2 * int(std::lround(L / h / 2));
  const size_t N = size_t(n) * n * n;
  auto id = [n](int i, int j, int k) { return (size_t(i) * n + j) * n + k; };
  const double c = 0.5 * (n - 1) * h;
  std::vector<double> V(N), psi(N), Hpsi(N);
  #pragma omp parallel for
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k) {
        double x = i * h - c, y = j * h - c, z = k * h - c;
        double ra = std::sqrt((x + R / 2) * (x + R / 2) + y * y + z * z);
        double rb = std::sqrt((x - R / 2) * (x - R / 2) + y * y + z * z);
        if (SUB <= 1) {
          V[id(i, j, k)] = -1.0 / ra - 1.0 / rb;
        } else {
          // egg-box fix: the potential of a grid point = the mean of -1/r over its cell
          // (SUB^3 sub-points), so it changes smoothly as a nucleus moves through the grid
          double v = 0;
          for (int a = 0; a < SUB; ++a)
            for (int b = 0; b < SUB; ++b)
              for (int d = 0; d < SUB; ++d) {
                double xs = x + h * ((a + 0.5) / SUB - 0.5), ys = y + h * ((b + 0.5) / SUB - 0.5),
                       zs = z + h * ((d + 0.5) / SUB - 0.5);
                v -= 1.0 / std::sqrt((xs + R / 2) * (xs + R / 2) + ys * ys + zs * zs) +
                     1.0 / std::sqrt((xs - R / 2) * (xs - R / 2) + ys * ys + zs * zs);
              }
          V[id(i, j, k)] = v / (SUB * SUB * SUB);
        }
        psi[id(i, j, k)] = std::exp(-ra) + std::exp(-rb);  // LCAO-like start
      }
  const double tau = 0.9 * h * h / 3.0, inv_h2 = 1.0 / (h * h);
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
  for (; step < 100000; ++step) {
    applyH(psi, Hpsi);
    if (step % 50 == 0) {
      E = dot(psi, Hpsi) / dot(psi, psi);
      if (!std::isfinite(E) || std::fabs(E - Eprev) < 1e-9) break;
      Eprev = E;
    }
    #pragma omp parallel for
    for (long q = 0; q < long(N); ++q) psi[q] -= tau * Hpsi[q];
    double nrm = std::sqrt(dot(psi, psi));
    #pragma omp parallel for
    for (long q = 0; q < long(N); ++q) psi[q] /= nrm;
  }
  *steps_out = step;
  return E + 1.0 / R;
}

int main(int argc, char** argv) {
  double h = argc > 1 ? std::atof(argv[1]) : 0.2;
  double L = argc > 2 ? std::atof(argv[2]) : 20.0;
  if (const char* e = std::getenv("SUB")) SUB = std::atoi(e);
  std::vector<double> Rs;
  for (int a = 3; a < argc; ++a) Rs.push_back(std::atof(argv[a]));
  if (Rs.empty()) Rs = {1.0, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 4.0, 6.0};
  std::printf("H2+  h %.3f bohr, box %.1f bohr, cell-average sub %d; exact R_e 1.997, E(R_e) -0.60263, D_e 0.10263 Hartree\n", h, L, SUB);
  std::printf("%6s %12s %14s %8s\n", "R", "E(R)", "E - E(H+H+)", "steps");
  double bestR = 0, bestE = 1e9;
  for (double R : Rs) {
    int st;
    double E = solve(L, h, R, &st);
    std::printf("%6.2f %12.6f %10.4f eV %8d\n", R, E, (E + 0.5) * 27.2114, st);
    std::fflush(stdout);
    if (E < bestE) { bestE = E; bestR = R; }
  }
  std::printf("minimum on the scanned R: R %.2f  E %.6f  D_e %.5f Hartree = %.1f kcal/mol (exact 64.4)\n",
              bestR, bestE, -0.5 - bestE, (-0.5 - bestE) * 627.509);
}
