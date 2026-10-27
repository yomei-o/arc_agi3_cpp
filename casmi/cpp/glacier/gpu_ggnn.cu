// gpu_ggnn.cu: device-resident ICEBERG GGNN (input projection + n x [per-edge-type messages, scatter-add, GRU cell])
// and the optional per-atom MLP head, for ice_model.hpp's set_gpu_ggnn hook (interface: ice_gpu.h).
// One upload of node features + edge structure per call, all intermediate states stay on the device, one download of
// the final node states (or of the per-node logits when the head runs on the device). Plain FP32 cuBLAS (no TF32).
// The scatter-add is a deterministic CSR gather by destination in exactly the CPU order (edge type, then edge index),
// so only the GEMM summation order and expf/tanhf differ from the CPU path.
// Every calling thread has its own cuBLAS handle, stream, device scratch and pinned staging; weights are uploaded
// once per host pointer and shared.
//   nvcc -O3 -c gpu_ggnn.cu   ->   link with -lcublas -lcudart; enable with ice::set_gpu_ggnn(gl_gpu_ggnn)
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

#include "ice_gpu.h"

namespace {
std::mutex g_wmu;
std::map<const float*, float*> g_wcache;  // host weight pointer -> device copy

float* dev_weight(const float* w, size_t n) {
  if (!w) return nullptr;
  std::lock_guard<std::mutex> g(g_wmu);
  auto it = g_wcache.find(w);
  if (it != g_wcache.end()) return it->second;
  float* d = nullptr;
  if (cudaMalloc(&d, n * sizeof(float)) != cudaSuccess) return nullptr;
  if (cudaMemcpy(d, w, n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(d);
    return nullptr;
  }
  g_wcache[w] = d;
  return d;
}

template <class T>
struct DBuf {
  T* p = nullptr;
  size_t cap = 0;
  bool grow(size_t need) {
    if (need <= cap) return true;
    if (p) cudaFree(p);
    p = nullptr;
    cap = 0;
    need = need + need / 4 + 1024;
    if (cudaMalloc(&p, need * sizeof(T)) != cudaSuccess) return false;
    cap = need;
    return true;
  }
  ~DBuf() {
    if (p) cudaFree(p);
  }
};
template <class T>
struct HBuf {  // pinned host staging
  T* p = nullptr;
  size_t cap = 0;
  bool grow(size_t need) {
    if (need <= cap) return true;
    if (p) cudaFreeHost(p);
    p = nullptr;
    cap = 0;
    need = need + need / 4 + 1024;
    if (cudaMallocHost(&p, need * sizeof(T)) != cudaSuccess) return false;
    cap = need;
    return true;
  }
  ~HBuf() {
    if (p) cudaFreeHost(p);
  }
};

struct Ctx {
  cublasHandle_t h = nullptr;
  cudaStream_t s = nullptr;
  DBuf<float> dx, dh, da, dxs, dms, dgi, dgh, dbet, droot, dtail, davg, dX, dhid, dlog;
  DBuf<int> di;
  HBuf<float> hf, hout;
  HBuf<int> hi;
  std::vector<int> pos, cnt;
  bool ok = false, failed = false;
  ~Ctx() {
    if (h) cublasDestroy(h);
    if (s) cudaStreamDestroy(s);
  }
  bool init() {
    if (ok) return true;
    if (failed) return false;
    failed = true;
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) return false;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) return false;
    cublasSetStream(h, s);
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);  // plain FP32
    failed = false;
    ok = true;
    return true;
  }
};
thread_local Ctx t_ctx;

constexpr int TPB = 256;
inline int nblk(size_t n) { return (int)((n + TPB - 1) / TPB); }

// row-major Y[n, out] = X[n, in] W[out, in]^T (no bias)
bool gemm(cublasHandle_t h, int n, int in, int out, const float* dW, const float* dX, float* dY) {
  if (n <= 0) return true;
  const float one = 1.f, zero = 0.f;
  return cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, out, n, in, &one, dW, in, dX, in, &zero, dY, out) ==
         CUBLAS_STATUS_SUCCESS;
}

__global__ void k_bias(float* y, const float* b, size_t n, int out) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) y[i] = y[i] + b[i % out];
}
__global__ void k_gather(const float* h, const int* rows, size_t n, int H, float* xs) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) xs[i] = h[(size_t)rows[i / H] * H + i % H];
}
// a[i] = sum over incoming edges (type order, then edge order) of (m_row + b_type)
__global__ void k_aggr(const float* ms, const int* rowptr, const int* eref, const float* bet, int N, int H, float* a) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= (size_t)N * H) return;
  int v = (int)(i / H), k = (int)(i % H);
  float s = 0.f;
  for (int j = rowptr[v]; j < rowptr[v + 1]; j++) {
    int e = eref[j];
    s += ms[(size_t)(e >> 2) * H + k] + bet[(e & 3) * H + k];
  }
  a[i] = s;
}
__global__ void k_gru(const float* gi, const float* gh, const float* bih, const float* bhh, int N, int H, float* h) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= (size_t)N * H) return;
  int v = (int)(i / H), k = (int)(i % H);
  const float* xi = gi + (size_t)v * 3 * H;
  const float* hi = gh + (size_t)v * 3 * H;
  float xr = xi[k] + (bih ? bih[k] : 0.f), xz = xi[H + k] + (bih ? bih[H + k] : 0.f),
        xn = xi[2 * H + k] + (bih ? bih[2 * H + k] : 0.f);
  float hr = hi[k] + (bhh ? bhh[k] : 0.f), hz = hi[H + k] + (bhh ? bhh[H + k] : 0.f),
        hn = hi[2 * H + k] + (bhh ? bhh[2 * H + k] : 0.f);
  float r = 1.f / (1.f + expf(-(xr + hr)));
  float z = 1.f / (1.f + expf(-(xz + hz)));
  float n = tanhf(xn + r * hn);
  h[i] = (1.f - z) * n + z * h[i];
}
__global__ void k_avg(const float* h, const int* gstart, const int* gsize, int G, int H, float* avg) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= (size_t)G * H) return;
  int g = (int)(i / H), k = (int)(i % H);
  float s = 0.f;
  for (int v = gstart[g]; v < gstart[g] + gsize[g]; v++) s += h[(size_t)v * H + k];
  avg[i] = s * (1.f / (float)max(1, gsize[g]));
}
__global__ void k_buildx(const float* root, const float* avg, const float* h, const float* tail, const int* gid, int N,
                         int H, int D, float* X) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= (size_t)N * D) return;
  int v = (int)(i / D), c = (int)(i % D), g = gid[v], tw = D - 3 * H;
  float x;
  if (c < H)
    x = root[c];
  else if (c < 2 * H)
    x = root[c - H] - avg[(size_t)g * H + c - H];
  else if (c < 3 * H)
    x = h[(size_t)v * H + c - 2 * H];
  else
    x = tail[(size_t)g * tw + c - 3 * H];
  X[i] = x;
}
__global__ void k_bias_relu(float* y, const float* b, size_t n, int out) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  float t = y[i] + (b ? b[i % out] : 0.f);
  y[i] = t > 0.f ? t : 0.f;
}
}  // namespace

extern "C" bool gl_gpu_ggnn(const ice::GpuGgnnW& W, const ice::GpuGraph& B, float* h_out, const ice::GpuHead* head,
                            float* logits) {
  Ctx& c = t_ctx;
  if (!c.init()) return false;
  const int N = B.N, H = W.H, E = B.E, G = B.G, in = W.in;
  if (N <= 0) return false;
  // device weights (cached)
  float* dwinp = dev_weight(W.winp, (size_t)H * in);
  float* dbinp = dev_weight(W.binp, H);
  float* dwih = dev_weight(W.wih, (size_t)3 * H * H);
  float* dwhh = dev_weight(W.whh, (size_t)3 * H * H);
  float* dbih = dev_weight(W.bih, (size_t)3 * H);
  float* dbhh = dev_weight(W.bhh, (size_t)3 * H);
  float* dwet[4];
  for (int t = 0; t < 4; t++) dwet[t] = dev_weight(W.wet[t], (size_t)H * H);
  if (!dwinp || !dwih || !dwhh) return false;
  // edge structure: per type the distinct source nodes (first-seen order), concatenated -> message rows;
  // CSR by destination with the incoming (message row, type) in (type, edge index) order
  c.pos.assign((size_t)4 * N, -1);
  int toff[5];
  std::vector<int> srcall;
  srcall.reserve(E);
  for (int t = 0; t < 4; t++) {
    toff[t] = (int)srcall.size();
    for (int e = 0; e < E; e++)
      if (B.typ[e] == t && c.pos[(size_t)t * N + B.src[e]] < 0) {
        c.pos[(size_t)t * N + B.src[e]] = (int)srcall.size();
        srcall.push_back(B.src[e]);
      }
  }
  toff[4] = (int)srcall.size();
  const int M = toff[4];
  for (int t = 0; t < 4; t++)
    if (toff[t + 1] > toff[t] && !dwet[t]) return false;
  const int tw = head ? head->D - 3 * H : 0;
  // pinned int staging: [srcall M | rowptr N+1 | eref E | gstart G | gsize G | gid N]
  const size_t o_rp = M, o_er = o_rp + N + 1, o_gs = o_er + E, o_gz = o_gs + G, o_gid = o_gz + G,
               nint = o_gid + N;
  if (!c.hi.grow(nint)) return false;
  int* hi = c.hi.p;
  std::memcpy(hi, srcall.data(), (size_t)M * sizeof(int));
  int* rp = hi + o_rp;
  std::memset(rp, 0, (size_t)(N + 1) * sizeof(int));
  for (int e = 0; e < E; e++) rp[B.dst[e] + 1]++;
  for (int v = 0; v < N; v++) rp[v + 1] += rp[v];
  c.cnt.assign(rp, rp + N);
  int* er = hi + o_er;
  for (int t = 0; t < 4; t++)
    for (int e = 0; e < E; e++)
      if (B.typ[e] == t) er[c.cnt[B.dst[e]]++] = (c.pos[(size_t)t * N + B.src[e]] << 2) | t;
  if (G > 0) {
    std::memcpy(hi + o_gs, B.gstart, (size_t)G * sizeof(int));
    std::memcpy(hi + o_gz, B.gsize, (size_t)G * sizeof(int));
  }
  std::memcpy(hi + o_gid, B.gid, (size_t)N * sizeof(int));
  // pinned float staging: [x N*in | bet 4H | root H | tail G*tw]
  const size_t f_bet = (size_t)N * in, f_root = f_bet + 4 * (size_t)H, f_tail = f_root + (head ? H : 0),
               nflt = f_tail + (size_t)G * tw;
  if (!c.hf.grow(nflt)) return false;
  float* hf = c.hf.p;
  std::memcpy(hf, B.x, (size_t)N * in * sizeof(float));
  for (int t = 0; t < 4; t++)
    if (W.bet[t])
      std::memcpy(hf + f_bet + (size_t)t * H, W.bet[t], H * sizeof(float));
    else
      std::memset(hf + f_bet + (size_t)t * H, 0, H * sizeof(float));
  if (head) {
    std::memcpy(hf + f_root, head->root, H * sizeof(float));
    if (G * tw) std::memcpy(hf + f_tail, head->tail, (size_t)G * tw * sizeof(float));
  }
  // device scratch
  if (!c.di.grow(nint) || !c.dx.grow(nflt) || !c.dh.grow((size_t)N * H) || !c.da.grow((size_t)N * H) ||
      !c.dxs.grow((size_t)std::max(M, 1) * H) || !c.dms.grow((size_t)std::max(M, 1) * H) ||
      !c.dgi.grow((size_t)N * 3 * H) || !c.dgh.grow((size_t)N * 3 * H))
    return false;
  cudaStream_t s = c.s;
  cudaMemcpyAsync(c.di.p, hi, nint * sizeof(int), cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(c.dx.p, hf, nflt * sizeof(float), cudaMemcpyHostToDevice, s);
  const int* d_src = c.di.p;
  const int* d_rp = c.di.p + o_rp;
  const int* d_er = c.di.p + o_er;
  const float* d_bet = c.dx.p + f_bet;
  float *h = c.dh.p, *a = c.da.p;
  const size_t NH = (size_t)N * H;
  // input projection
  if (!gemm(c.h, N, in, H, dwinp, c.dx.p, h)) return false;
  if (dbinp) k_bias<<<nblk(NH), TPB, 0, s>>>(h, dbinp, NH, H);
  for (int step = 0; step < W.steps; step++) {
    if (M > 0) {
      k_gather<<<nblk((size_t)M * H), TPB, 0, s>>>(h, d_src, (size_t)M * H, H, c.dxs.p);
      for (int t = 0; t < 4; t++) {
        int ns = toff[t + 1] - toff[t];
        if (ns && !gemm(c.h, ns, H, H, dwet[t], c.dxs.p + (size_t)toff[t] * H, c.dms.p + (size_t)toff[t] * H))
          return false;
      }
    }
    k_aggr<<<nblk(NH), TPB, 0, s>>>(c.dms.p, d_rp, d_er, d_bet, N, H, a);
    if (!gemm(c.h, N, H, 3 * H, dwih, a, c.dgi.p) || !gemm(c.h, N, H, 3 * H, dwhh, h, c.dgh.p)) return false;
    k_gru<<<nblk(NH), TPB, 0, s>>>(c.dgi.p, c.dgh.p, dbih, dbhh, N, H, h);
  }
  if (head) {
    const int D = head->D, HID = head->hid;
    float* dw1 = dev_weight(head->w1, (size_t)HID * D);
    float* db1 = dev_weight(head->b1, HID);
    float* dw2 = dev_weight(head->w2, HID);
    float* db2 = dev_weight(head->b2, 1);
    if (!dw1 || !dw2) return false;
    if (!c.davg.grow((size_t)std::max(G, 1) * H) || !c.dX.grow((size_t)N * D) || !c.dhid.grow((size_t)N * HID) ||
        !c.dlog.grow(N) || !c.hout.grow(N))
      return false;
    if (G > 0)
      k_avg<<<nblk((size_t)G * H), TPB, 0, s>>>(h, c.di.p + o_gs, c.di.p + o_gz, G, H, c.davg.p);
    k_buildx<<<nblk((size_t)N * D), TPB, 0, s>>>(c.dx.p + f_root, c.davg.p, h, c.dx.p + f_tail, c.di.p + o_gid, N, H,
                                                  D, c.dX.p);
    if (!gemm(c.h, N, D, HID, dw1, c.dX.p, c.dhid.p)) return false;
    k_bias_relu<<<nblk((size_t)N * HID), TPB, 0, s>>>(c.dhid.p, db1, (size_t)N * HID, HID);
    if (!gemm(c.h, N, HID, 1, dw2, c.dhid.p, c.dlog.p)) return false;
    if (db2) k_bias<<<nblk(N), TPB, 0, s>>>(c.dlog.p, db2, N, 1);
    cudaMemcpyAsync(c.hout.p, c.dlog.p, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost, s);
    if (cudaStreamSynchronize(s) != cudaSuccess || cudaGetLastError() != cudaSuccess) return false;
    std::memcpy(logits, c.hout.p, (size_t)N * sizeof(float));
    if (!h_out) return true;
  }
  if (h_out) {
    if (!c.hout.grow(NH)) return false;
    cudaMemcpyAsync(c.hout.p, h, NH * sizeof(float), cudaMemcpyDeviceToHost, s);
    if (cudaStreamSynchronize(s) != cudaSuccess) return false;
    std::memcpy(h_out, c.hout.p, NH * sizeof(float));
  } else if (cudaStreamSynchronize(s) != cudaSuccess) {
    return false;
  }
  return cudaGetLastError() == cudaSuccess;
}
