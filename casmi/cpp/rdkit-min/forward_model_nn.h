// Learned replacement for forward_model.h's fixed exp(-decay*k^2) H-shift
// rule: a tiny MLP (vendored autograd engine, casmi/cpp/autograd/) predicts
// each enumerated fragment's relative intensity from structural features,
// instead of a hand-picked weighting scheme. Fragment enumeration itself
// (which bonds can break) stays exactly as forward_model.h does it - that
// part is deterministic chemistry, not something to learn - only "how much
// of this fragment survives" is now a learned function, matching the split
// Sho Saga's writeup describes (rule-based enumeration + a trained
// intensity model).
//
// Feature set, v2: the first version (mass, size ratio, #bonds broken, H-
// shift) carries almost no information about *what* was cut - two
// completely different functional groups breaking off at the same size
// ratio and mass look identical to it. Holdout-measured (RESUME.md): a
// plain-MSE-trained MLP on those features scored MRR 0.151, far below the
// untrained fixed rule's 0.538 on the same pool/ppm. Elemental composition,
// aromaticity and the cut bond's own type/order are the cheapest, still
// non-GNN way to let the model tell fragments apart by *what they are*
// rather than only by size - the deliberate compromise short of actually
// reading the molecular graph through message passing.
#pragma once

#include <array>
#include <cmath>
#include <fstream>
#include <map>
#include <vector>

#include "../autograd/autograd.h"
#include "../autograd/nn_ops.h"
#include "forward_model.h"

namespace forward_model_nn {

// Base-fragment features, before H-shift expansion. k (the H-shift amount)
// and k*k are appended per shift at the point each is turned into a peak;
// adduct/instrument context is appended per spectrum (Context, below) -
// everything here is shift- and spectrum-independent, purely about the
// fragment's own chemistry.
struct FragFeatures {
  double mass = 0.0;
  double sizeRatio = 0.0;       // atoms+H in fragment / atoms+H in parent
  double bondsBroken = 0.0;     // 0 (intact molecule), 1, or 2
  double fracC = 0.0, fracH = 0.0, fracN = 0.0, fracO = 0.0, fracHet = 0.0;
  double aromaticRatio = 0.0;   // aromatic heavy atoms / heavy atoms, fragment-only
  double cutBondOrder = 0.0;    // mean bond order of the cut bond(s); 0 if intact
  double cutBondAromatic = 0.0; // 1 if any cut bond is aromatic
};

constexpr int kNumBaseFeatures = 11;
constexpr int kNumContextFeatures = 4;  // adductNorm, adductKnown, instrumentNorm, instrumentKnown
constexpr int kNumFeatures = kNumBaseFeatures + 2 + kNumContextFeatures;  // + k, k*k

// Per-spectrum conditioning - the same physical molecule fragments
// differently depending on how hard it was hit (collision energy isn't in
// the exported binary format yet, so this is adduct/instrument only for
// now; see RESUME.md's planned-next-step note on adding collision energy).
// idx<0 means unknown (encoded as 0.5, "known" flag 0, rather than
// colliding with a real category's own index).
struct Context {
  int adductIdx = -1;
  int instrumentIdx = -1;
  static constexpr int kNumAdducts = 10;
  static constexpr int kNumInstruments = 16;

  std::array<float, kNumContextFeatures> features() const {
    float an = adductIdx >= 0 ? float(adductIdx) / float(kNumAdducts) : 0.5f;
    float ak = adductIdx >= 0 ? 1.0f : 0.0f;
    float in_ = instrumentIdx >= 0 ? float(instrumentIdx) / float(kNumInstruments) : 0.5f;
    float ik = instrumentIdx >= 0 ? 1.0f : 0.0f;
    return {an, ak, in_, ik};
  }
};

// Element/aromaticity composition of one fragment (by atom index, in the
// ORIGINAL molecule's indexing - fragmentsAfterCutting returns index lists
// into `mol`, not a standalone sub-molecule).
inline void fragmentComposition(const RDKit::ROMol& mol, const std::vector<int>& atomIdx,
                                 double& fracC, double& fracH, double& fracN, double& fracO,
                                 double& fracHet, double& aromaticRatio) {
  int nC = 0, nN = 0, nO = 0, nHet = 0, nH = 0, nHeavy = 0, nAromatic = 0;
  for (int idx : atomIdx) {
    const RDKit::Atom* a = mol.getAtomWithIdx(idx);
    int z = a->getAtomicNum();
    ++nHeavy;
    if (a->getIsAromatic()) ++nAromatic;
    if (z == 6) ++nC;
    else if (z == 7) ++nN;
    else if (z == 8) ++nO;
    else ++nHet;
    nH += a->getTotalNumHs(false);
  }
  double budget = double(nHeavy + nH);
  if (budget <= 0.0) budget = 1.0;
  fracC = double(nC) / budget;
  fracH = double(nH) / budget;
  fracN = double(nN) / budget;
  fracO = double(nO) / budget;
  fracHet = double(nHet) / budget;
  aromaticRatio = nHeavy > 0 ? double(nAromatic) / double(nHeavy) : 0.0;
}

inline std::vector<FragFeatures> enumerateFragments(const RDKit::ROMol& mol) {
  std::vector<FragFeatures> out;
  int nAtomsParent = int(mol.getNumAtoms());
  int nHParent = 0;
  for (const auto& a : mol.atoms()) nHParent += a->getTotalNumHs(false);
  double parentBudget = double(nAtomsParent + nHParent);
  if (parentBudget <= 0.0) parentBudget = 1.0;

  auto makeFeat = [&](const std::vector<int>& atomIdx, double bondsBroken,
                       double cutBondOrder, double cutBondAromatic) {
    FragFeatures ff;
    ff.mass = forward_model::fragmentMass(mol, atomIdx);
    int nH = 0;
    for (int idx : atomIdx) nH += mol.getAtomWithIdx(idx)->getTotalNumHs(false);
    ff.sizeRatio = double(atomIdx.size() + size_t(nH)) / parentBudget;
    ff.bondsBroken = bondsBroken;
    fragmentComposition(mol, atomIdx, ff.fracC, ff.fracH, ff.fracN, ff.fracO, ff.fracHet,
                        ff.aromaticRatio);
    ff.cutBondOrder = cutBondOrder;
    ff.cutBondAromatic = cutBondAromatic;
    return ff;
  };

  std::vector<int> allAtoms(nAtomsParent);
  for (int i = 0; i < nAtomsParent; ++i) allAtoms[i] = i;
  out.push_back(makeFeat(allAtoms, 0.0, 0.0, 0.0));

  const RDKit::RingInfo* ri = mol.getRingInfo();
  for (const auto& bond : mol.bonds()) {
    if (ri->numBondRings(bond->getIdx()) > 0) continue;
    auto frags = forward_model::fragmentsAfterCutting(mol, {unsigned(bond->getIdx())});
    double order = bond->getBondTypeAsDouble();
    double aromatic = bond->getIsAromatic() ? 1.0 : 0.0;
    for (const auto& f : frags) out.push_back(makeFeat(f, 1.0, order, aromatic));
  }
  for (unsigned int r = 0; r < ri->numRings(); ++r) {
    const auto& ringBonds = ri->bondRings()[r];
    for (size_t i = 0; i < ringBonds.size(); ++i) {
      for (size_t j = i + 1; j < ringBonds.size(); ++j) {
        const RDKit::Bond* b1 = mol.getBondWithIdx(ringBonds[i]);
        const RDKit::Bond* b2 = mol.getBondWithIdx(ringBonds[j]);
        double order = 0.5 * (b1->getBondTypeAsDouble() + b2->getBondTypeAsDouble());
        double aromatic = (b1->getIsAromatic() || b2->getIsAromatic()) ? 1.0 : 0.0;
        auto frags = forward_model::fragmentsAfterCutting(
            mol, {unsigned(ringBonds[i]), unsigned(ringBonds[j])});
        for (const auto& f : frags) out.push_back(makeFeat(f, 2.0, order, aromatic));
      }
    }
  }
  return out;
}

struct MLP {
  int inDim, hidden;
  ag::Tensor W1, b1, W2, b2;

  static MLP init(int inDim, int hidden, uint64_t seed = 42) {
    ag::seed(seed);
    MLP m;
    m.inDim = inDim;
    m.hidden = hidden;
    float s1 = 1.0f / std::sqrt(float(inDim));
    float s2 = 1.0f / std::sqrt(float(hidden));
    m.W1 = ag::Tensor::randn({inDim, hidden}, s1, true);
    m.b1 = ag::Tensor::zeros({hidden}, true);
    m.W2 = ag::Tensor::randn({hidden, 1}, s2, true);
    m.b2 = ag::Tensor::zeros({1}, true);
    return m;
  }

  std::vector<ag::Tensor> params() const { return {W1, b1, W2, b2}; }

  // x: (N, inDim) -> (N, 1), non-negative (relative intensity, 0..1 - the
  // training targets already are). sigmoid, not relu, on the OUTPUT layer:
  // relu's gradient is exactly zero on its negative side, and a weighted-
  // MSE loss dominated by the majority zero-target rows has every
  // incentive to push the output layer's pre-activation negative almost
  // everywhere to minimize that loss - once it does, relu clips the output
  // to 0 AND kills the gradient needed to ever recover, a dead layer the
  // optimizer can't escape. Measured: three independently phase-1-trained
  // models (2000/20000/20000-with-5x-steps molecules) converged to bit-
  // identical holdout rankings (MRR 0.149) despite genuinely different
  // weights - the predicted "spectrum" was ~0 for every candidate, so every
  // cosine score was ~0 too and the ranking degenerated to whatever order
  // ties happened to sort in. A phase-2 ranking fine-tune on top of that
  // state then showed exactly this: loss frozen at precisely `margin` for
  // 850 steps straight, i.e. true and decoy scores both ~0, zero gradient.
  // sigmoid never fully dies this way (bounded in (0,1), always some
  // gradient short of extreme saturation) - this is what Sho Saga's own
  // writeup's intensity model is actually trained to predict (a relative
  // 0..1 intensity), not an unbounded non-negative value relu suits better.
  ag::Tensor forward(const ag::Tensor& x) const {
    ag::Tensor h = ag::relu(ag::linear(x, W1, b1));
    return ag::sigmoid(ag::linear(h, W2, b2));
  }

  void save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    int32_t magic = 0x464d4c50;  // "PLMF"
    f.write(reinterpret_cast<const char*>(&magic), 4);
    f.write(reinterpret_cast<const char*>(&inDim), 4);
    f.write(reinterpret_cast<const char*>(&hidden), 4);
    auto writeT = [&](const ag::Tensor& t) {
      int32_t n = t.numel();
      f.write(reinterpret_cast<const char*>(&n), 4);
      f.write(reinterpret_cast<const char*>(t.data().data()), n * sizeof(float));
    };
    writeT(W1); writeT(b1); writeT(W2); writeT(b2);
  }

  static MLP load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    int32_t magic = 0, inDim = 0, hidden = 0;
    f.read(reinterpret_cast<char*>(&magic), 4);
    f.read(reinterpret_cast<char*>(&inDim), 4);
    f.read(reinterpret_cast<char*>(&hidden), 4);
    MLP m = init(inDim, hidden);
    auto readT = [&](ag::Tensor& t) {
      int32_t n = 0;
      f.read(reinterpret_cast<char*>(&n), 4);
      f.read(reinterpret_cast<char*>(t.data().data()), n * sizeof(float));
    };
    readT(m.W1); readT(m.b1); readT(m.W2); readT(m.b2);
    return m;
  }
};

inline std::vector<float> featureRow(const FragFeatures& ff, int k, const Context& ctx) {
  std::vector<float> row = {
      float(ff.mass / 500.0), float(ff.sizeRatio), float(ff.bondsBroken),
      float(ff.fracC), float(ff.fracH), float(ff.fracN), float(ff.fracO), float(ff.fracHet),
      float(ff.aromaticRatio), float(ff.cutBondOrder), float(ff.cutBondAromatic),
      float(k), float(k * k),
  };
  auto c = ctx.features();
  row.insert(row.end(), c.begin(), c.end());
  return row;
}

struct Params {
  int maxHShift = 2;
  double massH = 1.007825;
};

// Split from predictSpectrumNN so a caller scoring one candidate against
// several of a molecule's own spectra (different adduct/instrument each)
// can enumerate the expensive, context-independent part - the fragment
// graph work - exactly once and re-run only the cheap MLP forward pass per
// spectrum's own context.
inline std::vector<std::pair<double, double>> scoreFragmentsWithContext(
    const std::vector<FragFeatures>& frags, const MLP& mlp, const Context& ctx,
    const Params& p = Params()) {
  int nShift = 2 * p.maxHShift + 1;
  int nRows = int(frags.size()) * nShift;
  std::vector<float> xs(size_t(nRows) * kNumFeatures);
  int row = 0;
  for (const auto& ff : frags)
    for (int k = -p.maxHShift; k <= p.maxHShift; ++k) {
      auto r = featureRow(ff, k, ctx);
      for (int c = 0; c < kNumFeatures; ++c) xs[size_t(row) * kNumFeatures + c] = r[c];
      ++row;
    }
  ag::Tensor x = ag::Tensor::from(xs, {nRows, kNumFeatures}, false);
  ag::Tensor y = mlp.forward(x);

  std::map<long long, double> peaks;
  row = 0;
  for (const auto& ff : frags)
    for (int k = -p.maxHShift; k <= p.maxHShift; ++k) {
      double shifted = ff.mass + double(k) * p.massH;
      peaks[llround(shifted * 1000.0)] += double(y.data()[row]);
      ++row;
    }
  std::vector<std::pair<double, double>> out;
  out.reserve(peaks.size());
  for (auto& kv : peaks) out.emplace_back(kv.first / 1000.0, kv.second);
  return out;
}

// Same output shape as forward_model::predictSpectrum: (m/z, intensity)
// pairs, mass-rounded-and-summed exactly the same way (two H-shifted peaks
// from different base fragments landing on the same milli-Da bin merge).
// Single-context convenience wrapper around scoreFragmentsWithContext -
// callers juggling several spectra per candidate should enumerate once and
// call scoreFragmentsWithContext directly instead.
inline std::vector<std::pair<double, double>> predictSpectrumNN(
    const RDKit::ROMol& mol, const MLP& mlp, const Context& ctx = Context(),
    const Params& p = Params()) {
  return scoreFragmentsWithContext(enumerateFragments(mol), mlp, ctx, p);
}

}  // namespace forward_model_nn
