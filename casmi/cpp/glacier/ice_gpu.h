// ice_gpu.h: plain-C interface of the optional device-resident ICEBERG GGNN (gpu_ggnn.cu). Only PODs and a function
// pointer type, so CPU-only builds (g++, no CUDA) include it unchanged; ice_model.hpp calls the hook when it is set.
#pragma once

namespace ice {

// GGNN weights (host pointers; the GPU side caches device copies keyed by these pointers)
struct GpuGgnnW {
  const float *winp, *binp;  // input projection [H, in]
  const float* wet[4];       // per edge type [H, H]
  const float* bet[4];
  const float *wih, *whh, *bih, *bhh;  // GRU [3H, H]
  int in, H, steps;
};
// batch of disjoint graphs (node features [N, in], directed edges, node ranges per graph)
struct GpuGraph {
  int N, E, G;
  const float* x;
  const int *src, *dst, *typ;
  const int *gstart, *gsize, *gid;
};
// optional per-node head on the final node states: X_i = [root | root - avgpool(h)_g | h_i | tail_g] (D columns,
// tail width D - 3H), logit_i = W2 relu(W1 X_i + b1) + b2 (W2 has one output row)
struct GpuHead {
  const float* root;  // [H]
  const float* tail;  // [G, D - 3H]
  const float *w1, *b1, *w2, *b2;
  int D, hid;
};
// h_out: [N, H] final node states (may be null when head is set); logits: [N] (head only). false -> use the CPU path
using GpuGgnnFn = bool (*)(const GpuGgnnW&, const GpuGraph&, float* h_out, const GpuHead* head, float* logits);

inline GpuGgnnFn& gpu_ggnn_hook() {
  static GpuGgnnFn f = nullptr;
  return f;
}
inline void set_gpu_ggnn(GpuGgnnFn f) { gpu_ggnn_hook() = f; }

}  // namespace ice
