// fw_score.cpp: own C++ forward-model re-scorer = the public casmi26-glacier gl_runner.py and casmi26-iceberg
// ice_runner.py (job construction, prediction, post-processing, gl/ice_ex_mean entropy scoring) on the own C++
// inference of GLACIER (gl_model.hpp) and ICEBERG (ice_model.hpp).
//
//   fw_score <in.txt> <out.tsv> [gl=glacier_w.bin] [ice=gen_w.bin,inten_w.bin] [threads=4] [budget=0]
// in.txt (written by the notebook; one block per molecule):
//   M <mid>
//   C <smiles>[	<iceberg root smiles>]       (candidates, ranked order; root = rm_stereo + InChI round trip, else
//                                              the RDKit canonical SMILES without stereo is used)
//   S <adduct>	<mode>	<instrument>	<ce_units>	<ce_orig>	<ce_ev comma list>	<prec>	<mz:it,mz:it,...>
// out.tsv: <mid>	<smiles>	<gl_ex_mean or nan>	<ice_ex_mean or nan>	<formula key>   (groups=N: only same-formula groups in the top N are predicted)
// GLACIER covers [M+H]+ spectra only (the public lineage's setting), ICEBERG [M+H]+ and [M+Na]+ (positive mode).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <regex>
#include <sstream>
#include <thread>
#include <tuple>
#include <unordered_map>

#include <functional>

#include "ice_model.hpp"

using namespace gl;

static const double FLOOR = 0.002, TOL = 0.01, PPM = 20.0, LOG4 = std::log(4.0), SIM_EPS = 0.0;
static const int TOPK = 256;

struct Spec {
  std::string adduct, mode, instrument, ce_units, ce_orig;
  std::vector<double> ce_ev, mz, it;
  double prec = 0;
};
struct Mol {
  std::string mid;
  std::vector<std::string> cands;
  std::vector<Spec> spectra;
};

// ---- query preparation (= gl_runner.clean_peaks / prep_query / _weights)
static void clean_peaks(const std::vector<double>& mz, const std::vector<double>& it, double prec, double floor, int topk,
                        double over, std::vector<double>& om, std::vector<double>& oi) {
  om.clear();
  oi.clear();
  const size_t n = mz.size();
  double mx = 0;
  for (size_t i = 0; i < n; i++)
    if (mz[i] <= prec + over && it[i] > mx) mx = it[i];
  if (mx <= 0) return;
  double thr = floor * mx;
  std::vector<size_t> idx;
  for (size_t i = 0; i < n; i++)
    if (mz[i] <= prec + over && it[i] >= thr && mz[i] > 0) idx.push_back(i);
  if ((int)idx.size() > topk) {
    std::vector<size_t> o(idx.size());
    for (size_t i = 0; i < o.size(); i++) o[i] = i;
    std::stable_sort(o.begin(), o.end(), [&](size_t a, size_t b) { return it[idx[a]] < it[idx[b]]; });
    std::vector<size_t> k2;
    for (size_t i = o.size() - topk; i < o.size(); i++) k2.push_back(idx[o[i]]);
    idx = k2;
  }
  std::vector<std::pair<double, float>> v;
  for (size_t i : idx) v.push_back({mz[i], (float)(it[i] / mx)});
  std::stable_sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.first < b.first; });
  for (auto& p : v) {
    om.push_back((double)(float)p.first);
    oi.push_back((double)p.second);
  }
}
static std::vector<double> weights(const std::vector<double>& it) {
  const size_t n = it.size();
  std::vector<double> p(n);
  double s = 0;
  for (size_t i = 0; i < n; i++) {
    p[i] = it[i];
    s += p[i];
  }
  if (s <= 0) return p;
  for (auto& x : p) x /= s;
  double S = 0;
  for (double x : p)
    if (x > 0) S -= x * std::log(x);
  if (S < 3.0) {
    double w = 0.25 + 0.25 * S, s2 = 0;
    for (auto& x : p) {
      x = std::pow(x, w);
      s2 += x;
    }
    if (s2 > 0)
      for (auto& x : p) x /= s2;
  }
  return p;
}
static void prep_query(const std::vector<double>& mz, const std::vector<double>& it, std::vector<double>& om,
                       std::vector<double>& op) {
  om.clear();
  op.clear();
  double mx = 0;
  for (double v : it) mx = std::max(mx, v);
  if (mx <= 0) return;
  double thr = FLOOR * mx;
  std::vector<size_t> idx;
  for (size_t i = 0; i < it.size(); i++)
    if (it[i] >= thr) idx.push_back(i);
  if ((int)idx.size() > TOPK) {
    std::vector<size_t> o(idx.size());
    for (size_t i = 0; i < o.size(); i++) o[i] = i;
    std::stable_sort(o.begin(), o.end(), [&](size_t a, size_t b) { return it[idx[a]] < it[idx[b]]; });
    std::vector<size_t> k2;
    for (size_t i = o.size() - TOPK; i < o.size(); i++) k2.push_back(idx[o[i]]);
    std::sort(k2.begin(), k2.end());
    idx = k2;
  }
  std::vector<double> oi;
  for (size_t i : idx) {
    om.push_back(mz[i]);
    oi.push_back(it[i]);
  }
  op = weights(oi);
}
static double entropy_sim(const std::vector<double>& qm, const std::vector<double>& qp, const std::vector<double>& cm,
                          const std::vector<double>& cp) {
  const size_t n = qm.size(), m = cm.size();
  if (!n || !m) return 0.0;
  double SA = 0, SB = 0, SAB = 0;
  for (double x : qp)
    if (x > 0) SA -= x * std::log(x);
  for (double x : cp)
    if (x > 0) SB -= x * std::log(x);
  size_t i = 0, j = 0;
  auto add = [&](double v) {
    if (v > 0) SAB -= v * std::log(v);
  };
  while (i < n && j < m) {
    double t = std::max(TOL, qm[i] * PPM * 1e-6), d = qm[i] - cm[j];
    if (d < -t) {
      add(qp[i] * 0.5);
      i++;
    } else if (d > t) {
      add(cp[j] * 0.5);
      j++;
    } else {
      add((qp[i] + cp[j]) * 0.5);
      i++;
      j++;
    }
  }
  for (; i < n; i++) add(qp[i] * 0.5);
  for (; j < m; j++) add(cp[j] * 0.5);
  double s = 1.0 - (2.0 * SAB - SA - SB) / LOG4;
  if (s < SIM_EPS) s = 0.0;
  if (s > 1.0) s = 1.0;
  return s;
}

// ---- query conditions (= gl_runner.instr_family / spectrum_ce / ce_bucket / is_covered / instr_token)
static bool contains(const std::string& t, const char* s) { return t.find(s) != std::string::npos; }
static int instr_family(std::string t) {
  std::transform(t.begin(), t.end(), t.begin(), ::tolower);
  if (t.empty() || t == "nan" || t == "none") return 5;
  if (contains(t, "timstof")) return 0;
  for (const char* s : {"orbitrap", "qft", "ftms", "hybrid ft", "itft", "exactive", "ft-icr", "fticr"})
    if (contains(t, s)) return 1;
  if (contains(t, "tof")) return 2;
  for (const char* s : {"trap", "qit", "lit"})
    if (contains(t, s)) return 3;
  for (const char* s : {"qq", "quattro", "triple"})
    if (contains(t, s)) return 4;
  return 5;
}
static double spectrum_ce(const Spec& sp) {
  if (sp.ce_units == "NCE") {
    static const std::regex num("[0-9]+\\.?[0-9]*");
    double s = 0;
    int c = 0;
    for (auto it = std::sregex_iterator(sp.ce_orig.begin(), sp.ce_orig.end(), num); it != std::sregex_iterator(); ++it) {
      s += std::atof(it->str().c_str());
      c++;
    }
    return c ? s / c : NAN;
  }
  if (sp.ce_ev.empty()) return NAN;
  double s = 0;
  for (double v : sp.ce_ev) s += v;
  return s / sp.ce_ev.size();
}
static double ce_bucket(double ce) {
  if (!std::isfinite(ce)) return NAN;
  double b = std::nearbyint(ce / 5.0) * 5.0;
  return b < 5 ? 5.0 : b;
}
static bool covered(const Spec& sp, bool mh_only) {
  std::string md = sp.mode;
  std::transform(md.begin(), md.end(), md.begin(), ::tolower);
  bool pos = md == "positive" || md == "pos" || md == "+" || md == "1";
  if (!pos) return false;
  return sp.adduct == "[M+H]+" || (!mh_only && sp.adduct == "[M+Na]+");
}

// ---- input
static std::vector<double> parse_list(const std::string& s) {
  std::vector<double> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ','))
    if (!t.empty()) v.push_back(std::atof(t.c_str()));
  return v;
}
static std::vector<Mol> read_input(const std::string& path) {
  std::ifstream f(path);
  std::vector<Mol> mols;
  std::string l;
  while (std::getline(f, l)) {
    if (!l.empty() && l.back() == '\r') l.pop_back();
    if (l.size() < 2) continue;
    if (l[0] == 'M') {
      mols.emplace_back();
      mols.back().mid = l.substr(2);
    } else if (l[0] == 'C' && !mols.empty()) {
      mols.back().cands.push_back(l.substr(2));
    } else if (l[0] == 'S' && !mols.empty()) {
      std::vector<std::string> p;
      std::stringstream ss(l.substr(2));
      std::string t;
      while (std::getline(ss, t, '\t')) p.push_back(t);
      if (p.size() < 8) continue;
      Spec s;
      s.adduct = p[0];
      s.mode = p[1];
      s.instrument = p[2];
      s.ce_units = p[3];
      s.ce_orig = p[4];
      s.ce_ev = parse_list(p[5]);
      s.prec = std::atof(p[6].c_str());
      std::stringstream ps(p[7]);
      while (std::getline(ps, t, ',')) {
        size_t c = t.find(':');
        if (c == std::string::npos) continue;
        s.mz.push_back(std::atof(t.substr(0, c).c_str()));
        s.it.push_back(std::atof(t.substr(c + 1).c_str()));
      }
      mols.back().spectra.push_back(std::move(s));
    }
  }
  return mols;
}

struct Plan {
  std::vector<double> qm, qp;
  std::string adduct, instr;
  double ceb;
};
using JobKey = std::tuple<std::string, std::string, std::string, double>;  // model input key, adduct, instr, ce

// One forward model: which spectra it covers, how a candidate maps to its input key, and one prediction.
struct FwModel {
  std::string name;
  bool mh_only = true;  // [M+H]+ only (GLACIER) or [M+H]+ and [M+Na]+ (ICEBERG)
  std::function<std::string(const std::string& cand, const std::string& root)> key_of;
  std::function<bool(const std::string& key, const std::string& adduct, const std::string& instr, double ce,
                     std::vector<double>& mz, std::vector<double>& it)>
      run;
};

// scores[k][c] for every molecule k / candidate c (NaN = none); predictions parallel over unique jobs
static std::vector<std::vector<double>> score_model(const FwModel& FM, const std::vector<Mol>& mols,
                                                    const std::vector<std::vector<std::string>>& roots,
                                                    const std::vector<std::vector<char>>& want, int NT,
                                                    double deadline, std::chrono::steady_clock::time_point t0) {
  auto el = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
  static const double SUM_CE[3] = {20.0, 40.0, 60.0};
  std::vector<std::vector<Plan>> plans(mols.size());
  std::vector<std::vector<std::string>> keys(mols.size());
  std::map<JobKey, int> job_id;
  std::vector<JobKey> jobs;
  for (size_t k = 0; k < mols.size(); k++) {
    for (const Spec& sp : mols[k].spectra) {
      if (!covered(sp, FM.mh_only)) continue;
      Plan p;
      std::vector<double> m1, i1, m2, i2;
      clean_peaks(sp.mz, sp.it, sp.prec, 0.001, 512, 2.0, m1, i1);
      clean_peaks(m1, i1, sp.prec, 0.001, 512, 2.0, m2, i2);
      prep_query(m2, i2, p.qm, p.qp);
      p.adduct = sp.adduct;
      p.instr = instr_family(sp.instrument) == 1 ? "Orbitrap" : "QTOF";
      p.ceb = ce_bucket(spectrum_ce(sp));
      plans[k].push_back(p);
    }
    keys[k].resize(mols[k].cands.size());
    if (plans[k].empty()) continue;
    for (size_t c = 0; c < mols[k].cands.size(); c++) {
      if (!want[k][c]) continue;
      keys[k][c] = FM.key_of(mols[k].cands[c], roots[k][c]);
      if (keys[k][c].empty()) continue;
      for (const Plan& p : plans[k]) {
        std::vector<double> ces;
        if (std::isfinite(p.ceb))
          ces.push_back(p.ceb);
        else
          ces.assign(SUM_CE, SUM_CE + 3);
        for (double ce : ces) {
          JobKey key{keys[k][c], p.adduct, p.instr, ce};
          if (!job_id.count(key)) {
            job_id[key] = (int)jobs.size();
            jobs.push_back(key);
          }
        }
      }
    }
  }
  std::vector<std::vector<double>> pm(jobs.size()), pi(jobs.size());
  std::vector<uint8_t> ok(jobs.size(), 0);
  std::atomic<size_t> next{0};
  std::atomic<int> nfail{0}, nskip{0};
  auto worker = [&] {
    for (;;) {
      size_t j = next.fetch_add(1);
      if (j >= jobs.size()) break;
      if (deadline > 0 && el() > deadline) {
        nskip++;
        continue;
      }
      const auto& [kk, ad, ins, ce] = jobs[j];
      try {
        ok[j] = FM.run(kk, ad, ins, ce, pm[j], pi[j]) && !pm[j].empty();
        if (!ok[j]) nfail++;
      } catch (...) {
        nfail++;
      }
    }
  };
  std::vector<std::thread> th;
  for (int t = 0; t < NT; t++) th.emplace_back(worker);
  for (auto& t : th) t.join();
  std::fprintf(stderr, "fw_score[%s]: %zu covered molecules, %zu predictions, ok %zu, failed %d, skipped (time) %d (%.1fs)\n",
               FM.name.c_str(), (size_t)std::count_if(plans.begin(), plans.end(), [](auto& p) { return !p.empty(); }),
               jobs.size(), (size_t)std::count(ok.begin(), ok.end(), 1), nfail.load(), nskip.load(), el());
  std::vector<std::vector<double>> out(mols.size());
  for (size_t k = 0; k < mols.size(); k++) {
    out[k].assign(mols[k].cands.size(), NAN);
    for (size_t c = 0; c < mols[k].cands.size(); c++) {
      if (keys[k][c].empty()) continue;
      std::vector<double> sims;
      for (const Plan& p : plans[k]) {
        std::vector<double> m, it;
        if (std::isfinite(p.ceb)) {
          int j = job_id[{keys[k][c], p.adduct, p.instr, p.ceb}];
          if (!ok[j]) continue;
          m = pm[j];
          it = pi[j];
        } else {  // summed-CE prediction (= runner merge)
          std::map<double, double> acc;
          bool all = true;
          for (double ce : SUM_CE) {
            int j = job_id[{keys[k][c], p.adduct, p.instr, ce}];
            if (!ok[j]) {
              all = false;
              break;
            }
            double mx = *std::max_element(pi[j].begin(), pi[j].end());
            for (size_t q = 0; q < pm[j].size(); q++) acc[std::nearbyint(pm[j][q] * 1e4) / 1e4] += pi[j][q] / mx;
          }
          if (!all) continue;
          double mx = 0;
          for (auto& kv : acc) mx = std::max(mx, kv.second);
          for (auto& kv : acc) {
            m.push_back(kv.first);
            it.push_back(kv.second / mx);
          }
        }
        std::vector<double> qm, qp;
        prep_query(m, it, qm, qp);
        double s = entropy_sim(p.qm, p.qp, qm, qp);
        if (std::isfinite(s)) sims.push_back(s);
      }
      if (!sims.empty()) {
        double s = 0;
        for (double v : sims) s += v;
        out[k][c] = s / sims.size();
      }
    }
  }
  return out;
}

// SMILES without stereochemistry, RDKit canonical (fallback ICEBERG root when no InChI round trip is supplied)
// molecular formula key (element counts incl. H, net charge) = what the re-ranker groups candidates by
static std::string formula_key(const std::string& smi) {
  std::unique_ptr<RDKit::RWMol> m;
  try {
    m.reset(RDKit::SmilesToMol(smi));
  } catch (...) {
    return "";
  }
  if (!m) return "";
  std::map<std::string, int> cnt;
  int chg = 0, hs = 0;
  for (const RDKit::Atom* a : m->atoms()) {
    cnt[a->getSymbol()]++;
    hs += (int)a->getTotalNumHs();
    chg += a->getFormalCharge();
  }
  if (hs) cnt["H"] += hs;
  std::string k;
  for (auto& [s, c] : cnt) k += s + std::to_string(c);
  if (chg) k += (chg > 0 ? "+" : "") + std::to_string(chg);
  return k;
}

static std::string nostereo_canon(const std::string& smi) {
  std::unique_ptr<RDKit::RWMol> m;
  try {
    m.reset(RDKit::SmilesToMol(smi));
    if (!m) return "";
    RDKit::MolOps::removeStereochemistry(*m);
    return RDKit::MolToSmiles(*m);
  } catch (...) {
    return "";
  }
}

#ifdef FW_GPU
extern "C" bool gl_gpu_linear(const float*, int, const float*, const float*, int, int, float*);
extern "C" bool gl_gpu_available();
#endif

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: fw_score <in.txt> <out.tsv> [gl=glacier_w.bin] [ice=gen_w.bin,inten_w.bin] [threads=4] "
                 "[budget=0 (s, per model)]\n");
    return 1;
  }
  auto t0 = std::chrono::steady_clock::now();
  std::string glw, icew;
  int NT = 4, groups_top = 0;
  double budget = 0;
  for (int i = 3; i < argc; i++) {
    std::string a = argv[i];
    if (a.rfind("gl=", 0) == 0) glw = a.substr(3);
    if (a.rfind("ice=", 0) == 0) icew = a.substr(4);
    if (a.rfind("threads=", 0) == 0) NT = std::max(1, std::atoi(a.c_str() + 8));
    if (a.rfind("budget=", 0) == 0) budget = std::atof(a.c_str() + 7);
    if (a.rfind("groups=", 0) == 0) groups_top = std::atoi(a.c_str() + 7);
    if (a == "gpu=1") {
#ifdef FW_GPU
      if (gl_gpu_available()) {
        set_gpu_linear(gl_gpu_linear);
        std::fprintf(stderr, "fw_score: GPU GEMMs on\n");
      } else {
        std::fprintf(stderr, "fw_score: no CUDA device - CPU\n");
      }
#else
      std::fprintf(stderr, "fw_score: built without FW_GPU - CPU\n");
#endif
    }
  }
  std::vector<Mol> mols = read_input(argv[1]);
  // candidate lines may carry the ICEBERG root ("smiles\troot")
  std::vector<std::vector<std::string>> roots(mols.size());
  for (size_t k = 0; k < mols.size(); k++)
    for (std::string& c : mols[k].cands) {
      size_t t = c.find('\t');
      roots[k].push_back(t == std::string::npos ? "" : c.substr(t + 1));
      if (t != std::string::npos) c = c.substr(0, t);
    }
  // formula of every candidate; groups=N: only members of same-formula groups (>= 2) within the first N
  // candidates are predicted (the only ones a same-formula re-ranker can move) - others get roots "" -> no key
  std::vector<std::vector<std::string>> forms(mols.size());
  std::vector<std::vector<char>> want(mols.size());
  for (size_t k = 0; k < mols.size(); k++) {
    std::map<std::string, int> cnt;
    for (size_t c = 0; c < mols[k].cands.size(); c++) {
      forms[k].push_back(formula_key(mols[k].cands[c]));
      if ((int)c < groups_top && !forms[k][c].empty()) cnt[forms[k][c]]++;
    }
    for (size_t c = 0; c < mols[k].cands.size(); c++)
      want[k].push_back(groups_top <= 0 || ((int)c < groups_top && !forms[k][c].empty() && cnt[forms[k][c]] >= 2));
  }
  std::vector<std::vector<double>> sgl, sice;
  double used = 0;
  if (!glw.empty()) {
    Model M(glw);
    std::mutex fm;
    std::unordered_map<std::string, std::shared_ptr<MolFeat>> fcache;
    std::unordered_map<std::string, std::string> ccache;
    FwModel FM;
    FM.name = "glacier";
    FM.mh_only = true;
    FM.key_of = [&](const std::string& c, const std::string&) {
      std::lock_guard<std::mutex> g(fm);
      auto it = ccache.find(c);
      if (it != ccache.end()) return it->second;
      return ccache[c] = canon(c);
    };
    FM.run = [&](const std::string& cc, const std::string& ad, const std::string& ins, double ce, std::vector<double>& mz,
                 std::vector<double>& it) {
      std::shared_ptr<MolFeat> F;
      {
        std::lock_guard<std::mutex> g(fm);
        auto f = fcache.find(cc);
        if (f != fcache.end()) F = f->second;
      }
      if (!F) {
        auto nf = std::make_shared<MolFeat>();
        if (!featurize(cc, *nf)) return false;
        std::lock_guard<std::mutex> g(fm);
        F = fcache.emplace(cc, nf).first->second;
      }
      Cond C;
      if (!cond_for(ad, ins, (float)ce, C)) return false;
      post(predict(M, *F, C), mz, it);
      return true;
    };
    sgl = score_model(FM, mols, roots, want, NT, budget, t0);
    used = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }
  if (!icew.empty()) {
    size_t cm = icew.find(',');
    ice::Net G(icew.substr(0, cm), 6, false), I(icew.substr(cm + 1), 4, true);
    float CHEM[18];
    for (int e = 0; e < 18; e++) CHEM[e] = (float)ELEMENT_MASS[e];
    std::mutex fm;
    std::unordered_map<std::string, std::shared_ptr<ice::Root>> rcache;
    std::unordered_map<std::string, std::string> kcache;
    FwModel FM;
    FM.name = "iceberg";
    FM.mh_only = false;
    FM.key_of = [&](const std::string& c, const std::string& root) {
      if (!root.empty()) return root;
      std::lock_guard<std::mutex> g(fm);
      auto it = kcache.find(c);
      if (it != kcache.end()) return it->second;
      return kcache[c] = nostereo_canon(c);
    };
    FM.run = [&](const std::string& rs, const std::string& ad, const std::string& ins, double ce, std::vector<double>& mz,
                 std::vector<double>& it) {
      std::shared_ptr<ice::Root> R;
      {
        std::lock_guard<std::mutex> g(fm);
        auto f = rcache.find(rs);
        if (f != rcache.end()) R = f->second;
      }
      if (!R) {
        auto nr = std::make_shared<ice::Root>();
        if (!ice::root_from_smiles(rs, *nr)) return false;
        std::lock_guard<std::mutex> g(fm);
        R = rcache.emplace(rs, nr).first->second;
      }
      Cond C;
      if (!cond_for(ad, ins, (float)ce, C)) return false;
      post(ice::predict(G, I, *R, C, CHEM), mz, it);
      return true;
    };
    sice = score_model(FM, mols, roots, want, NT, budget > 0 ? used + budget : 0, t0);
  }
  FILE* fo = std::fopen(argv[2], "w");
  if (!fo) return 1;
  for (size_t k = 0; k < mols.size(); k++)
    for (size_t c = 0; c < mols[k].cands.size(); c++) {
      double g = sgl.empty() ? NAN : sgl[k][c], s = sice.empty() ? NAN : sice[k][c];
      std::fprintf(fo, "%s\t%s\t", mols[k].mid.c_str(), mols[k].cands[c].c_str());
      if (std::isfinite(g)) std::fprintf(fo, "%.6f", g); else std::fprintf(fo, "nan");
      std::fprintf(fo, "\t");
      if (std::isfinite(s)) std::fprintf(fo, "%.6f", s); else std::fprintf(fo, "nan");
      std::fprintf(fo, "\t%s\n", forms[k][c].c_str());
    }
  std::fclose(fo);
  std::fprintf(stderr, "fw_score: done %.1fs\n",
               std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  return 0;
}
