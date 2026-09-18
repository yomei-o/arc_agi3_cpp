// arc3_net.h - the official sample's idea, in C++, at a size a CPU can afford.
//
// StochasticGoose (the ARC-AGI-3 sample submission, and the thing every team on
// the leaderboard is standing on) trains a CNN to predict, for each action and
// each of the 4096 click coordinates, whether taking it will produce a NEW
// frame, and samples from that. It is a novelty policy with a learned, state-
// conditioned prior, and it is worth about 3.5 points.
//
// Porting it verbatim does not work here. Its backbone runs 256 channels at the
// full 64x64, which is ~3.15 G multiply-accumulates per forward pass, and it
// trains on a batch of 64 every 5 actions. That is fine on the T4 it was written
// for and hopeless on a CPU.
//
// It is also more network than the problem needs. The board is drawn several
// pixels per logical cell (5 on LS20, 3 on TU93, 4 on WA30), so a 64x64
// coordinate head is predicting at roughly sixteen times the resolution the game
// actually has. This version works on a 32x32 subsample with narrow channels and
// puts the coordinate head at 16x16 - one output per logical cell - which comes
// to about 7 M MAC per forward, some 450x cheaper, and predicts at the
// resolution the game is played at.
//
// The autograd engine underneath is yomei-o's, from othello_alphazero_cpp
// (originally mini-yolov5-cpp): ag/autograd.{h,cpp}, no external dependencies.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "ag/autograd.h"

namespace arc3net {

constexpr int IN_C = 16;    // one-hot over the 16 ARC colours
constexpr int IN_HW = 32;   // 64x64 subsampled by 2
constexpr int COORD_HW = 16;      // one coordinate output per logical cell
constexpr int N_SIMPLE = 5;       // ACTION1..ACTION5
constexpr int N_COORD = COORD_HW * COORD_HW;
constexpr int N_OUT = N_SIMPLE + N_COORD;

// ---------------------------------------------------------------------------
// Binary cross-entropy straight on the logits.
//
// The engine has sigmoid but no log, and composing BCE out of what is there
// would be numerically poor near saturation. Written directly it is also the
// simplest possible backward: d/dz = (sigmoid(z) - y) / N.
// ---------------------------------------------------------------------------
inline ag::Tensor bce_with_logits(const ag::Tensor& z, const std::vector<float>& y) {
  const int n = z.numel();
  ag::Tensor out = ag::make_op({1}, {z.n}, "bce_with_logits");
  const std::vector<float>& zd = z.data();

  double total = 0.0;
  for (int i = 0; i < n; ++i) {
    float v = zd[i];
    // log(1+exp(-|v|)) + max(v,0) - v*y, the stable form.
    total += std::max(v, 0.0f) - v * y[i] + std::log1p(std::exp(-std::fabs(v)));
  }
  out.data()[0] = float(total / std::max(1, n));

  ag::Node* Z = z.n.get();
  ag::Node* O = out.n.get();
  std::vector<float> yy = y;
  out.n->backward_fn = [Z, O, yy, n]() {
    float g = O->grad[0] / std::max(1, n);
    for (int i = 0; i < n; ++i) {
      float s = 1.0f / (1.0f + std::exp(-Z->data[i]));
      Z->grad[i] += g * (s - yy[i]);
    }
  };
  return out;
}

// ---------------------------------------------------------------------------
// Adam. Same defaults as the sample's optimiser, a larger step because this
// network is much smaller.
// ---------------------------------------------------------------------------
struct Adam {
  std::vector<ag::Tensor> params;
  std::vector<std::vector<float> > m, v;
  float lr, b1, b2, eps;
  long t;

  Adam() : lr(1e-3f), b1(0.9f), b2(0.999f), eps(1e-8f), t(0) {}

  void attach(const std::vector<ag::Tensor>& ps) {
    params = ps;
    m.assign(ps.size(), {});
    v.assign(ps.size(), {});
    for (size_t i = 0; i < ps.size(); ++i) {
      m[i].assign(ps[i].numel(), 0.0f);
      v[i].assign(ps[i].numel(), 0.0f);
    }
    t = 0;
  }

  void zero_grad() {
    for (size_t i = 0; i < params.size(); ++i) params[i].zero_grad();
  }

  void step() {
    ++t;
    float c1 = 1.0f - std::pow(b1, float(t));
    float c2 = 1.0f - std::pow(b2, float(t));
    for (size_t i = 0; i < params.size(); ++i) {
      std::vector<float>& p = params[i].data();
      std::vector<float>& g = params[i].grad();
      for (size_t j = 0; j < p.size(); ++j) {
        m[i][j] = b1 * m[i][j] + (1 - b1) * g[j];
        v[i][j] = b2 * v[i][j] + (1 - b2) * g[j] * g[j];
        float mh = m[i][j] / c1, vh = v[i][j] / c2;
        p[j] -= lr * mh / (std::sqrt(vh) + eps);
      }
    }
  }
};

// ---------------------------------------------------------------------------
// The network.
//
//   conv 16->16  3x3            32x32
//   conv 16->32  3x3 stride 2   16x16
//   conv 32->32  3x3            16x16
//   action head  maxpool 4 -> 32*4*4=512 -> 64 -> 5
//   coord  head  conv 32->16 3x3 -> conv 16->1 1x1 -> 16x16 flattened
// ---------------------------------------------------------------------------
struct ActionNet {
  // Used both as the sample's change-predictor and, in DQN mode, as Q(s, .):
  // the outputs are the same 5 + 16x16 action space either way, only what they
  // are trained to mean differs.
  ag::Tensor w1, b1_, w2, b2_, w3, b3_;
  ag::Tensor fcw, fcb, outw, outb;
  ag::Tensor cw1, cb1, cw2, cb2;
  Adam opt;

  ActionNet() { reset(); }

  static ag::Tensor kaiming(int o, int c, int k) {
    float std = std::sqrt(2.0f / float(c * k * k));
    return ag::Tensor::randn({o, c, k, k}, std, true);
  }

  // A level change means a different game state distribution; the sample
  // rebuilds its network at that point and so does this.
  void reset() {
    w1 = kaiming(16, IN_C, 3);   b1_ = ag::Tensor::zeros({16}, true);
    w2 = kaiming(32, 16, 3);     b2_ = ag::Tensor::zeros({32}, true);
    w3 = kaiming(32, 32, 3);     b3_ = ag::Tensor::zeros({32}, true);

    fcw = ag::Tensor::randn({512, 64}, std::sqrt(2.0f / 512.0f), true);
    fcb = ag::Tensor::zeros({64}, true);
    outw = ag::Tensor::randn({64, N_SIMPLE}, std::sqrt(2.0f / 64.0f), true);
    outb = ag::Tensor::zeros({N_SIMPLE}, true);

    cw1 = kaiming(16, 32, 3);    cb1 = ag::Tensor::zeros({16}, true);
    cw2 = kaiming(1, 16, 1);     cb2 = ag::Tensor::zeros({1}, true);

    opt.attach(params());
  }

  // Everything the optimiser touches, in a fixed order, so a target network can
  // be copied across element by element (the trick from mario_dqn_cpp's QNet).
  std::vector<ag::Tensor> params() {
    std::vector<ag::Tensor> ps;
    ps.push_back(w1); ps.push_back(b1_);
    ps.push_back(w2); ps.push_back(b2_);
    ps.push_back(w3); ps.push_back(b3_);
    ps.push_back(fcw); ps.push_back(fcb);
    ps.push_back(outw); ps.push_back(outb);
    ps.push_back(cw1); ps.push_back(cb1);
    ps.push_back(cw2); ps.push_back(cb2);
    return ps;
  }

  void copy_from(ActionNet& o) {
    std::vector<ag::Tensor> a = params(), b = o.params();
    for (size_t i = 0; i < a.size(); ++i) a[i].data() = b[i].data();
  }

  // x: (N, 16, 32, 32) -> (N, N_OUT)
  ag::Tensor forward(const ag::Tensor& x) {
    ag::Tensor h = ag::relu(ag::conv2d(x, w1, b1_, 1, 1));
    h = ag::relu(ag::conv2d(h, w2, b2_, 2, 1));
    h = ag::relu(ag::conv2d(h, w3, b3_, 1, 1));   // (N,32,16,16)
    int n = h.shape()[0];

    ag::Tensor a = ag::maxpool2d(h, 4, 4, 0);      // (N,32,4,4)
    a = ag::reshape(a, {n, 512});
    a = ag::relu(ag::add_bias_2d(ag::matmul(a, fcw), fcb));
    a = ag::add_bias_2d(ag::matmul(a, outw), outb);   // (N,5)

    ag::Tensor c = ag::relu(ag::conv2d(h, cw1, cb1, 1, 1));
    c = ag::conv2d(c, cw2, cb2, 1, 0);             // (N,1,16,16)
    c = ag::reshape(c, {n, N_COORD});

    // cat_channels works on (N,C,H,W); these are 2-D, so join them by hand.
    ag::Tensor out = ag::make_op({n, N_OUT}, {a.n, c.n}, "cat_logits");
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < N_SIMPLE; ++j) out.data()[i * N_OUT + j] = a.data()[i * N_SIMPLE + j];
      for (int j = 0; j < N_COORD; ++j) out.data()[i * N_OUT + N_SIMPLE + j] = c.data()[i * N_COORD + j];
    }
    ag::Node* A = a.n.get();
    ag::Node* C = c.n.get();
    ag::Node* O = out.n.get();
    out.n->backward_fn = [A, C, O, n]() {
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < N_SIMPLE; ++j) A->grad[i * N_SIMPLE + j] += O->grad[i * N_OUT + j];
        for (int j = 0; j < N_COORD; ++j) C->grad[i * N_COORD + j] += O->grad[i * N_OUT + N_SIMPLE + j];
      }
    };
    return out;
  }
};

// One-hot a 64x64 frame into (16, 32, 32), sampling every other pixel.
inline void encode(const int8_t* frame64, float* out) {
  std::fill(out, out + IN_C * IN_HW * IN_HW, 0.0f);
  for (int y = 0; y < IN_HW; ++y)
    for (int x = 0; x < IN_HW; ++x) {
      int v = frame64[(y * 2) * 64 + (x * 2)];
      if (v < 0 || v >= IN_C) v = 0;
      out[(v * IN_HW + y) * IN_HW + x] = 1.0f;
    }
}

}  // namespace arc3net
