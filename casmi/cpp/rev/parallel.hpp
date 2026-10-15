// Portable parallel-for. Uses OpenMP when compiled with it (g++ -fopenmp), and a
// std::thread fan-out otherwise. This keeps a single source that parallelises under
// g++/OpenMP *and* MSVC (whose OpenMP cannot parse pragmas inside the lambdas the
// autograd tape is built from) — CPU vs parallel stays a compiler-flag choice.
#pragma once
#include <cstdint>
#ifdef _OPENMP
  #include <omp.h>
#else
  #include <thread>
  #include <vector>
  #include <algorithm>
#endif

// Optional cap on the std::thread fan-out (0 = all hardware threads), e.g. to
// share a machine or to time a 4-core target.
inline unsigned& parallel_cap() { static unsigned c = 0; return c; }

// Run body(i) for i in [0, n). body must be safe to call concurrently for distinct i.
// Each thread takes one contiguous block of i (no false sharing on element-wise writes).
template <class F>
inline void parallel_for(int64_t n, F body) {
  if (n <= 0) return;
#ifdef _OPENMP
  #pragma omp parallel for
  for (long long i = 0; i < (long long)n; ++i) body((int64_t)i);
#else
  unsigned hw = std::thread::hardware_concurrency(); if (!hw) hw = 1;
  if (parallel_cap() && hw > parallel_cap()) hw = parallel_cap();
  int64_t T = std::min<int64_t>((int64_t)hw, n);
  if (T <= 1) { for (int64_t i = 0; i < n; ++i) body(i); return; }
  std::vector<std::thread> ths;
  auto worker = [&](int64_t t) {
    int64_t i0 = n * t / T, i1 = n * (t + 1) / T;
    for (int64_t i = i0; i < i1; ++i) body(i);
  };
  for (int64_t t = 1; t < T; ++t) ths.emplace_back(worker, t);
  worker(0);
  for (auto& th : ths) th.join();
#endif
}
