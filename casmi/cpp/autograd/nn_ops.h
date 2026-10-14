// One small addition to the vendored autograd engine: a row-broadcast bias
// add for plain 2D (N, H) tensors, the same shape of thing add_bias_nchw
// already does for (N, C, H, W) - that one broadcasts a per-channel bias
// over the spatial dims, this one broadcasts a per-feature bias over the
// batch dim. Needed because autograd.h's add() only takes same-shape
// tensors; a Linear(x, W, b) layer for an MLP needs x@W + b with b
// (out_features) broadcast over x's batch rows (N, out_features).
#pragma once
#include "autograd.h"

namespace ag {

Tensor add_row_bias(const Tensor& x, const Tensor& b);  // x:(N,H) b:(H) -> (N,H)

// Linear(x, W, b) = x @ W + b, x:(N,in) W:(in,out) b:(out) -> (N,out)
inline Tensor linear(const Tensor& x, const Tensor& W, const Tensor& b) {
  return add_row_bias(matmul(x, W), b);
}

}  // namespace ag
