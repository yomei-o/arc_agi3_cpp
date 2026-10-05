// net_test.cpp - does the network learn, and how fast is it?
//
// Two things need checking before this goes anywhere near the agent. First that
// the gradients are right, which is easiest to see by making it learn something
// with a known answer. Second the cost of a training step, because the whole
// reason for shrinking the sample's architecture was that its version cannot be
// afforded on a CPU - so the replacement has to be measured, not assumed.
//
//   g++ -O2 -std=c++17 net_test.cpp ag/autograd.cpp -o net_test -pthread

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "arc3_net.h"

using namespace arc3net;

static double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main() {
  ag::seed(1234);
  ActionNet net;

  // A task with a knowable answer: the frame carries a marker colour, and the
  // action that "works" is determined by where the marker is. If the network
  // cannot fit this, the wiring is wrong.
  const int BATCH = 16;
  const int STEPS = 60;

  std::vector<int8_t> frame(64 * 64);
  std::vector<float> xs(BATCH * IN_C * IN_HW * IN_HW);
  std::vector<float> ys(BATCH);
  std::vector<float> mask(BATCH * N_OUT);

  auto t_start = std::chrono::steady_clock::now();
  double last_loss = 0;

  for (int step = 0; step < STEPS; ++step) {
    std::fill(mask.begin(), mask.end(), 0.0f);
    for (int b = 0; b < BATCH; ++b) {
      // Marker on the left half -> ACTION1 works; right half -> ACTION2 works.
      int left = (b + step) % 2;
      std::fill(frame.begin(), frame.end(), int8_t(0));
      int mx = left ? 10 : 50;
      for (int dy = 0; dy < 6; ++dy)
        for (int dx = 0; dx < 6; ++dx) frame[(20 + dy) * 64 + (mx + dx)] = 7;
      encode(frame.data(), xs.data() + size_t(b) * IN_C * IN_HW * IN_HW);

      int act = (b % 2);                      // ask about ACTION1 or ACTION2
      mask[b * N_OUT + act] = 1.0f;
      ys[b] = (act == (left ? 0 : 1)) ? 1.0f : 0.0f;
    }

    ag::Tensor x = ag::Tensor::from(xs, {BATCH, IN_C, IN_HW, IN_HW}, false);
    ag::Tensor logits = net.forward(x);
    ag::Tensor m = ag::Tensor::from(mask, {BATCH, N_OUT}, false);
    ag::Tensor ones = ag::Tensor::from(std::vector<float>(N_OUT, 1.0f), {N_OUT, 1}, false);
    ag::Tensor picked = ag::matmul(ag::mul(logits, m), ones);   // (BATCH,1)

    ag::Tensor loss = bce_with_logits(picked, ys);
    net.opt.zero_grad();
    loss.backward();
    net.opt.step();
    last_loss = loss.item();
    if (step % 10 == 0) std::printf("step %3d  loss %.4f\n", step, last_loss);
  }

  std::printf("\nfinal loss %.4f after %d steps in %.0f ms (%.1f ms/step, batch %d)\n",
              last_loss, STEPS, ms_since(t_start), ms_since(t_start) / STEPS, BATCH);

  // Inference cost for a single frame, which is what the agent pays per action.
  std::fill(frame.begin(), frame.end(), int8_t(0));
  std::vector<float> one(IN_C * IN_HW * IN_HW);
  encode(frame.data(), one.data());
  auto t1 = std::chrono::steady_clock::now();
  const int REPS = 50;
  for (int i = 0; i < REPS; ++i) {
    ag::Tensor x1 = ag::Tensor::from(one, {1, IN_C, IN_HW, IN_HW}, false);
    ag::Tensor l1 = net.forward(x1);
    (void)l1;
  }
  std::printf("forward (batch 1): %.2f ms\n", ms_since(t1) / REPS);

  std::printf("\n%s\n", last_loss < 0.25 ? "LEARNED: loss fell, gradients are wired correctly"
                                         : "DID NOT LEARN - check the wiring");
  return last_loss < 0.25 ? 0 : 1;
}
