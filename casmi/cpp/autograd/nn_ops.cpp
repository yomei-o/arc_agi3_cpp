#include "nn_ops.h"

namespace ag {

Tensor add_row_bias(const Tensor& x, const Tensor& b) {
  int N = x.shape()[0], H = x.shape()[1];
  auto out = make_op({N, H}, {x.n, b.n}, "add_row_bias");
  auto& xd = x.data();
  auto& bd = b.data();
  auto& od = out.data();
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < H; ++j) od[i * H + j] = xd[i * H + j] + bd[j];
  Node *X = x.n.get(), *B = b.n.get(), *O = out.n.get();
  out.n->backward_fn = [X, B, O, N, H]() {
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < H; ++j) {
        float g = O->grad[i * H + j];
        X->grad[i * H + j] += g;
        B->grad[j] += g;
      }
  };
  return out;
}

}  // namespace ag
