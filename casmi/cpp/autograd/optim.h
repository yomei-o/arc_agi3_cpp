// Adam/AdamW, ported from yomei-o/yolov8_cpp's pure/optim.hpp onto this
// repo's autograd.h Tensor API (a wrapper class with .data()/.grad(), not
// yolov8_cpp's raw std::shared_ptr<Node> alias - the logic is identical,
// only the accessors differ). Kept separate from autograd.h/.cpp (copied
// verbatim from yomei-o/mini-yolov5-cpp's scratch/, already gradient-
// checked) since this part is a port, not a copy.
//
// Adam over plain momentum-SGD because a small MLP with few, cheap-to-
// compute features (see forward_model_nn.h) benefits from per-parameter
// adaptive step sizes without the warmup/lr0 tuning a deep conv net needs -
// one less thing to sweep against a holdout.
#pragma once
#include "autograd.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace ag {

inline float cosine_lr(int step, int total, float base, int warmup = 0, float min_lr = 0.f) {
  if (step < warmup) return base * float(step + 1) / float(std::max(1, warmup));
  float t = float(step - warmup) / float(std::max(1, total - warmup));
  return min_lr + 0.5f * (base - min_lr) * (1.f + std::cos(3.14159265358979f * std::min(1.f, t)));
}

struct Adam {
  std::vector<Tensor> params;
  float lr, b1, b2, eps, wd;
  bool decoupled;  // true = AdamW (decoupled weight decay), false = Adam (L2 into grad)
  std::vector<std::vector<float>> m, v;
  int t = 0;

  Adam(std::vector<Tensor> p, float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f,
       float eps = 1e-8f, float weight_decay = 0.f, bool decoupled = false)
      : params(std::move(p)), lr(lr), b1(beta1), b2(beta2), eps(eps), wd(weight_decay),
        decoupled(decoupled) {
    for (auto& q : params) {
      m.emplace_back(q.numel(), 0.f);
      v.emplace_back(q.numel(), 0.f);
    }
  }

  void zero_grad() {
    for (auto& p : params) p.zero_grad();
  }

  void step() {
    ++t;
    float bc1 = 1.f - std::pow(b1, t), bc2 = 1.f - std::pow(b2, t);
    for (size_t k = 0; k < params.size(); ++k) {
      auto& pd = params[k].data();
      auto& pg = params[k].grad();
      for (size_t i = 0; i < pd.size(); ++i) {
        if (decoupled) pd[i] -= lr * wd * pd[i];
        float g = pg[i] + (decoupled ? 0.f : wd * pd[i]);
        m[k][i] = b1 * m[k][i] + (1 - b1) * g;
        v[k][i] = b2 * v[k][i] + (1 - b2) * g * g;
        float mhat = m[k][i] / bc1, vhat = v[k][i] / bc2;
        pd[i] -= lr * mhat / (std::sqrt(vhat) + eps);
      }
    }
  }
};

}  // namespace ag
