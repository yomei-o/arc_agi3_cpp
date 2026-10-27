// gl_score.cpp: own C++ GLACIER re-scorer (= gl_runner.py of the public casmi26-glacier package: job construction,
// prediction, post-processing and gl_ex_mean scoring), on the own C++ inference of gl_model.hpp / gl_feat.hpp.
//
//   gl_score <glacier_w.bin> <in.txt> <out.tsv> [threads=4] [budget_s=0] [mh_only=1]
// in.txt (written by the notebook; one block per molecule):
//   M <mid>
//   C <smiles>                                  (candidates, ranked order)
//   S <adduct>\t<mode>\t<instrument>\t<ce_units>\t<ce_orig>\t<ce_ev comma list>\t<prec>\t<mz:it,mz:it,...>
// out.tsv: <mid>\t<smiles>\t<gl_ex_mean | nan>   (nan = no covered spectrum / rejected / prediction failed)
// mh_only=1 keeps only [M+H]+ spectra (the public lineage's GLACIER setting); 0 = [M+H]+ and [M+Na]+ (runner default).
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

#include "gl_feat.hpp"

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
using JobKey = std::tuple<std::string, std::string, std::string, double>;  // canonical, adduct, instr, ce

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: gl_score glacier_w.bin in.txt out.tsv [threads=4] [budget_s=0] [mh_only=1]\n");
    return 1;
  }
  auto t0 = std::chrono::steady_clock::now();
  auto el = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
  const int NT = argc > 4 ? std::max(1, std::atoi(argv[4])) : 4;
  const double budget = argc > 5 ? std::atof(argv[5]) : 0.0;
  const bool mh_only = argc > 6 ? std::atoi(argv[6]) != 0 : true;
  Model M(argv[1]);
  std::vector<Mol> mols = read_input(argv[2]);
  // plans and jobs
  std::vector<std::vector<Plan>> plans(mols.size());
  std::unordered_map<std::string, std::string> canon_of;
  std::map<JobKey, int> job_id;
  std::vector<JobKey> jobs;
  static const double SUM_CE[3] = {20.0, 40.0, 60.0};
  for (size_t k = 0; k < mols.size(); k++) {
    for (const Spec& sp : mols[k].spectra) {
      if (!covered(sp, mh_only)) continue;
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
    if (plans[k].empty()) continue;
    for (const std::string& c : mols[k].cands) {
      if (!canon_of.count(c)) canon_of[c] = canon(c);
      const std::string& cc = canon_of[c];
      if (cc.empty()) continue;
      for (const Plan& p : plans[k]) {
        std::vector<double> ces;
        if (std::isfinite(p.ceb))
          ces.push_back(p.ceb);
        else
          ces.assign(SUM_CE, SUM_CE + 3);
        for (double ce : ces) {
          JobKey key{cc, p.adduct, p.instr, ce};
          if (!job_id.count(key)) {
            job_id[key] = (int)jobs.size();
            jobs.push_back(key);
          }
        }
      }
    }
  }
  std::fprintf(stderr, "gl_score: %zu molecules, %zu covered, %zu unique candidates, %zu predictions (%.1fs)\n",
               mols.size(), (size_t)std::count_if(plans.begin(), plans.end(), [](auto& p) { return !p.empty(); }),
               canon_of.size(), jobs.size(), el());
  // order: small molecules first is not needed; keep input order (= molecule priority)
  std::vector<std::vector<double>> pm(jobs.size()), pi(jobs.size());
  std::vector<uint8_t> ok(jobs.size(), 0);
  std::atomic<size_t> next{0};
  std::atomic<int> nfail{0};
  std::mutex fm;
  std::unordered_map<std::string, std::shared_ptr<MolFeat>> fcache;
  auto worker = [&] {
    for (;;) {
      size_t j = next.fetch_add(1);
      if (j >= jobs.size()) break;
      if (budget > 0 && el() > budget) continue;
      const auto& [cc, ad, ins, ce] = jobs[j];
      std::shared_ptr<MolFeat> F;
      {
        std::lock_guard<std::mutex> g(fm);
        auto it = fcache.find(cc);
        if (it != fcache.end()) F = it->second;
      }
      if (!F) {
        auto nf = std::make_shared<MolFeat>();
        if (!featurize(cc, *nf)) {
          nfail++;
          continue;
        }
        std::lock_guard<std::mutex> g(fm);
        F = fcache.emplace(cc, nf).first->second;
      }
      Cond C;
      if (!cond_for(ad, ins, (float)ce, C)) {
        nfail++;
        continue;
      }
      try {
        std::vector<float> raw = predict(M, *F, C);
        post(raw, pm[j], pi[j]);
        ok[j] = !pm[j].empty();
      } catch (...) {
        nfail++;
      }
    }
  };
  std::vector<std::thread> th;
  for (int t = 0; t < NT; t++) th.emplace_back(worker);
  for (auto& t : th) t.join();
  size_t nok = std::count(ok.begin(), ok.end(), 1);
  std::fprintf(stderr, "gl_score: predictions ok %zu / %zu, failed %d (%.1fs, %.1f pred/s)\n", nok, jobs.size(),
               nfail.load(), el(), nok / std::max(1e-9, el()));
  // score
  FILE* fo = std::fopen(argv[3], "w");
  if (!fo) return 1;
  size_t nscored = 0;
  for (size_t k = 0; k < mols.size(); k++) {
    for (const std::string& c : mols[k].cands) {
      const std::string& cc = canon_of.count(c) ? canon_of[c] : std::string();
      std::vector<double> sims;
      if (!cc.empty())
        for (const Plan& p : plans[k]) {
          std::vector<double> m, it;
          if (std::isfinite(p.ceb)) {
            int j = job_id[{cc, p.adduct, p.instr, p.ceb}];
            if (!ok[j]) continue;
            m = pm[j];
            it = pi[j];
          } else {  // summed-CE prediction (= gl_runner.merge)
            std::map<double, double> acc;
            bool all = true;
            for (double ce : SUM_CE) {
              int j = job_id[{cc, p.adduct, p.instr, ce}];
              if (!ok[j]) {
                all = false;
                break;
              }
              double mx = *std::max_element(pi[j].begin(), pi[j].end());
              for (size_t q = 0; q < pm[j].size(); q++)
                acc[std::nearbyint(pm[j][q] * 1e4) / 1e4] += pi[j][q] / mx;
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
      if (sims.empty()) {
        std::fprintf(fo, "%s\t%s\tnan\n", mols[k].mid.c_str(), c.c_str());
      } else {
        double s = 0;
        for (double v : sims) s += v;
        std::fprintf(fo, "%s\t%s\t%.6f\n", mols[k].mid.c_str(), c.c_str(), s / sims.size());
        nscored++;
      }
    }
  }
  std::fclose(fo);
  std::fprintf(stderr, "gl_score: %zu candidates scored, total %.1fs\n", nscored, el());
  return 0;
}
