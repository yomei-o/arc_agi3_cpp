// gpu_linear.cu: optional cuBLAS path for gl::linear (y[n, out] = x[n, in] W^T + b) used by the forward models
// (ICEBERG's GGNN / GRU GEMMs dominate its CPU time). Weights are uploaded once per pointer and cached; every calling
// thread has its own cuBLAS handle, stream and scratch buffers, so the CPU worker threads can all issue GEMMs.
//   nvcc -O3 -c gpu_linear.cu   ->   link with -lcublas -lcudart; enable with gl::set_gpu_linear(gl_gpu_linear)
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <map>
#include <mutex>

namespace {
std::mutex g_wmu;
std::map<const float*, float*> g_wcache;  // host weight pointer -> device copy

struct Ctx {
  cublasHandle_t h = nullptr;
  cudaStream_t s = nullptr;
  float *dx = nullptr, *dy = nullptr;
  size_t cx = 0, cy = 0;
  ~Ctx() {
    if (dx) cudaFree(dx);
    if (dy) cudaFree(dy);
    if (h) cublasDestroy(h);
    if (s) cudaStreamDestroy(s);
  }
};
thread_local Ctx t_ctx;

float* dev_weight(const float* w, size_t n) {
  std::lock_guard<std::mutex> g(g_wmu);
  auto it = g_wcache.find(w);
  if (it != g_wcache.end()) return it->second;
  float* d = nullptr;
  if (cudaMalloc(&d, n * sizeof(float)) != cudaSuccess) return nullptr;
  cudaMemcpy(d, w, n * sizeof(float), cudaMemcpyHostToDevice);
  g_wcache[w] = d;
  return d;
}
bool grow(float*& p, size_t& cap, size_t need) {
  if (need <= cap) return true;
  if (p) cudaFree(p);
  p = nullptr;
  cap = 0;
  if (cudaMalloc(&p, need * sizeof(float)) != cudaSuccess) return false;
  cap = need;
  return true;
}
}  // namespace

// returns false (caller falls back to the CPU kernel) on any CUDA problem
extern "C" bool gl_gpu_linear(const float* x, int n, const float* w, const float* b, int in, int out, float* y) {
  Ctx& c = t_ctx;
  if (!c.h) {
    if (cudaStreamCreate(&c.s) != cudaSuccess) return false;
    if (cublasCreate(&c.h) != CUBLAS_STATUS_SUCCESS) return false;
    cublasSetStream(c.h, c.s);
    cublasSetMathMode(c.h, CUBLAS_DEFAULT_MATH);  // plain FP32 (no TF32) to stay close to the CPU results
  }
  float* dw = dev_weight(w, (size_t)out * in);
  if (!dw) return false;
  if (!grow(c.dx, c.cx, (size_t)n * in) || !grow(c.dy, c.cy, (size_t)n * out)) return false;
  cudaMemcpyAsync(c.dx, x, (size_t)n * in * sizeof(float), cudaMemcpyHostToDevice, c.s);
  // row-major Y[n,out] = X[n,in] W[out,in]^T  <=>  column-major Y^T[out,n] = W^T... : C(out x n) = A(out x in, from W
  // row-major = W^T col-major -> op T) * B(in x n, X row-major = X^T col-major)
  const float one = 1.f, zero = 0.f;
  if (cublasSgemm(c.h, CUBLAS_OP_T, CUBLAS_OP_N, out, n, in, &one, dw, in, c.dx, in, &zero, c.dy, out) !=
      CUBLAS_STATUS_SUCCESS)
    return false;
  cudaMemcpyAsync(y, c.dy, (size_t)n * out * sizeof(float), cudaMemcpyDeviceToHost, c.s);
  if (cudaStreamSynchronize(c.s) != cudaSuccess) return false;
  if (b)
    for (int r = 0; r < n; r++) {
      float* yr = y + (size_t)r * out;
      for (int o = 0; o < out; o++) yr[o] += b[o];
    }
  return true;
}

extern "C" bool gl_gpu_available() {
  int nd = 0;
  return cudaGetDeviceCount(&nd) == cudaSuccess && nd > 0;
}
