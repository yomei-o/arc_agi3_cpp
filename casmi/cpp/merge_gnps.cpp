// Merge GNPS's public spectral library (ALL_GNPS_cleaned.csv + .mgf,
// zenodo.org/records/11193898, CC-BY-4.0) into train's own reference
// library (train_head.bin/train_peaks.bin/train_mols.txt), in the exact
// same binary layout export_bin.py already writes. Unlike the PubChem
// pool (structure-only, scored by an approximate forward model), this
// adds REAL reference spectra - match.cpp's own already-strong cosine/
// entropy search can use them directly, with no forward-model
// approximation at all.
//
// Why this matters: match.cpp can only ever answer a test molecule whose
// structure already has a reference spectrum somewhere in its library -
// an estimated ~84% of this competition's real test set does not, because
// train only covers 275,810 structures. GNPS is a large, independent
// public spectral library (542,778 cleaned entries here) that likely
// covers many structures train does not - every GNPS entry whose
// structure is genuinely new to the library directly raises the ceiling
// match.cpp's existing search can reach, without touching any of
// match.cpp's own code or tuning.
//
// Identity key: train's own "name" field is already inchikey14 (the first
// dash-delimited block of a standard InChIKey, read directly off train's
// own precomputed column - see export_bin.py). GNPS's MGF carries SMILES
// but not InChIKey; the CSV sidecar does (InChIKey_smiles), joined here by
// spectrum_id. A GNPS entry with no CSV match or no InChIKey is skipped -
// its spectrum can still only ever help if we know what structure to
// credit it to.
//
// Adduct/instrument strings don't match train's export_bin.py vocabulary
// verbatim (GNPS: "[M+H]1+", "qtof"; train: "[M+H]+", "QTOF") and are
// normalised here; anything unrecognised gets tag -1 (unknown), the same
// convention export_bin.py already uses for missing/unmapped values.
//
// Usage:
//   merge_gnps <dir> <gnps_csv> <gnps_mgf> <out_stem>
// Writes <out_stem>_head.bin/_peaks.bin/_mols.txt - train's own files are
// read but never overwritten.
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr float PEAK_FLOOR = 0.002f;

const std::vector<std::string> ADDUCTS = {
    "[M+H]+", "[M-H]-", "[M+Na]+", "[M+NH4]+", "[M+K]+", "[M+Cl]-",
    "[M+CH2O2-H]-", "[M+H-H2O]+", "[2M+H]+", "[2M+Na]+"};
const std::vector<std::string> INSTR = {
    "timstof", "orbitrap", "qtof", "lc-esi-qtof", "esi-qft"};

std::string upperStrip(std::string s) {
  std::string out;
  for (char c : s)
    if (!std::isspace(static_cast<unsigned char>(c))) out += c;
  return out;
}

// GNPS adducts carry an explicit charge count RDKit's/train's own
// vocabulary doesn't ("[M+H]1+" vs "[M+H]+"); strip a single digit right
// before the trailing +/- if present, then exact-match.
int adductIndex(std::string a) {
  a = upperStrip(a);
  for (char sign : {'+', '-'}) {
    size_t p = a.rfind(sign);
    if (p != std::string::npos && p > 0 && std::isdigit((unsigned char)a[p - 1]))
      a.erase(p - 1, 1);
  }
  for (size_t i = 0; i < ADDUCTS.size(); ++i) {
    std::string ref = upperStrip(ADDUCTS[i]);
    if (a == ref) return int(i);
  }
  return -1;
}

std::string lower(std::string s) {
  for (char& c : s) c = char(std::tolower((unsigned char)c));
  return s;
}

// Substring match (case-insensitive): GNPS's msMassAnalyzer/msIonisation
// fields are freeform ("qtof", "Maxis Impact qTOF", ...) rather than
// train's fixed vocabulary.
int instrIndex(const std::string& analyzer, const std::string& ionisation) {
  std::string hay = lower(analyzer) + " " + lower(ionisation);
  // "qtof" alone would also match inside "lc-esi-qtof"; check from the end
  // backwards so the more specific multi-word entry (lc-esi-qtof, index 3)
  // wins over the shorter one (qtof, index 2) when both would match.
  for (int i = int(INSTR.size()) - 1; i >= 0; --i) {
    std::string needle = lower(INSTR[i]);
    if (hay.find(needle) != std::string::npos) return i;
  }
  return -1;
}

struct CsvRow { std::string inchikey14; };

std::vector<std::string> splitCsvLine(const std::string& line) {
  // Minimal CSV splitter: handles quoted fields with embedded commas, no
  // escaped quotes (none observed in this file's SMILES/InChI fields).
  std::vector<std::string> out;
  std::string cur;
  bool inQuotes = false;
  for (size_t i = 0; i < line.size(); ++i) {
    char c = line[i];
    if (c == '"') { inQuotes = !inQuotes; continue; }
    if (c == ',' && !inQuotes) { out.push_back(cur); cur.clear(); continue; }
    cur += c;
  }
  out.push_back(cur);
  return out;
}

std::unordered_map<std::string, std::string> loadCsvInchiKeys(const std::string& path) {
  std::unordered_map<std::string, std::string> out;
  std::ifstream f(path);
  if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(1); }
  std::string line;
  std::getline(f, line);
  auto header = splitCsvLine(line);
  int idSpecCol = -1, ikCol = -1;
  for (size_t i = 0; i < header.size(); ++i) {
    if (header[i] == "spectrum_id") idSpecCol = int(i);
    if (header[i] == "InChIKey_smiles") ikCol = int(i);
  }
  if (idSpecCol < 0 || ikCol < 0) {
    std::fprintf(stderr, "csv missing spectrum_id/InChIKey_smiles columns\n");
    std::exit(1);
  }
  long long n = 0, withKey = 0;
  while (std::getline(f, line)) {
    auto cols = splitCsvLine(line);
    if (int(cols.size()) <= std::max(idSpecCol, ikCol)) continue;
    ++n;
    const std::string& ik = cols[ikCol];
    if (ik.empty()) continue;
    std::string ik14 = ik.substr(0, ik.find('-'));
    if (ik14.size() != 14) continue;  // not a well-formed InChIKey block
    out[cols[idSpecCol]] = ik14;
    ++withKey;
  }
  std::fprintf(stderr, "csv: %lld rows, %lld with a usable InChIKey14\n", n, withKey);
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : ".";
  std::string csvPath = argc > 2 ? argv[2] : "gnps/ALL_GNPS_cleaned.csv";
  std::string mgfPath = argc > 3 ? argv[3] : "gnps/ALL_GNPS_cleaned.mgf";
  std::string outStem = argc > 4 ? argv[4] : "train2";

  auto ik14ById = loadCsvInchiKeys(dir + "/" + csvPath);

  // Seed the molecule table from train's own existing mols.txt, same dedup
  // key (inchikey14) and same mol_id numbering, so existing head.bin rows
  // (not rewritten here, only appended to) stay valid against the new
  // combined mols.txt.
  std::vector<std::string> names, smiles;
  std::unordered_map<std::string, int> molId;
  {
    std::ifstream f(dir + "/train_mols.txt");
    if (!f) { std::fprintf(stderr, "cannot read train_mols.txt\n"); return 1; }
    std::string line;
    while (std::getline(f, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      size_t t = line.find('\t');
      std::string name = line.substr(0, t);
      std::string smi = t == std::string::npos ? "" : line.substr(t + 1);
      molId[name] = int(names.size());
      names.push_back(name);
      smiles.push_back(smi);
    }
  }
  int nTrainMols = int(names.size());
  std::fprintf(stderr, "train: %d existing molecules\n", nTrainMols);

  // Copy train's existing head.bin/peaks.bin forward verbatim, then append.
  std::ofstream headOut(dir + "/" + outStem + "_head.bin", std::ios::binary);
  std::ofstream peaksOut(dir + "/" + outStem + "_peaks.bin", std::ios::binary);
  {
    std::ifstream hf(dir + "/train_head.bin", std::ios::binary);
    headOut << hf.rdbuf();
    std::ifstream pf(dir + "/train_peaks.bin", std::ios::binary);
    peaksOut << pf.rdbuf();
  }
  // Running peak offset: head.off indexes into the loader's OWN mz/inten
  // arrays (each sized to the sum of every spectrum's own peak count n),
  // not a byte/float offset into peaks.bin itself - peaks.bin interleaves
  // [mz(n floats), inten(n floats)] per spectrum, i.e. 2n floats on disk
  // but only n counted in the logical offset. Getting this wrong (originally
  // computed as raw file size / 4, exactly double the right value) silently
  // produces out-of-bounds offsets for every appended spectrum - match.exe
  // crashed (exit code 5, no output at all) the first time this ran.
  long long off = 0;
  long long nTrainSpectra = 0;
  {
    std::ifstream hf(dir + "/train_head.bin", std::ios::binary | std::ios::ate);
    nTrainSpectra = hf.tellg() / 20;  // f,i,i,i,i = 20 bytes/record
    hf.seekg(0);
    for (long long i = 0; i < nTrainSpectra; ++i) {
      float precursor; int32_t mol, hoff, n, tag;
      hf.read(reinterpret_cast<char*>(&precursor), 4);
      hf.read(reinterpret_cast<char*>(&mol), 4);
      hf.read(reinterpret_cast<char*>(&hoff), 4);
      hf.read(reinterpret_cast<char*>(&n), 4);
      hf.read(reinterpret_cast<char*>(&tag), 4);
      off += n;
    }
  }
  std::fprintf(stderr, "train: %lld existing spectra\n", nTrainSpectra);

  std::ifstream mgf(dir + "/" + mgfPath);
  if (!mgf) { std::fprintf(stderr, "cannot read %s\n", (dir + "/" + mgfPath).c_str()); return 1; }

  std::string line;
  long long nRead = 0, nKept = 0, nNoKey = 0, nNoSmiles = 0, nNoPeaks = 0, nNewMols = 0;
  std::string specId, adduct, smilesCur, analyzer, ionisation;
  float pepmass = 0.0f;
  std::vector<std::pair<float, float>> peaks;
  bool inRecord = false;

  auto flushRecord = [&]() {
    ++nRead;
    if (specId.empty()) return;
    auto itKey = ik14ById.find(specId);
    if (itKey == ik14ById.end()) { ++nNoKey; return; }
    if (smilesCur.empty()) { ++nNoSmiles; return; }
    if (peaks.empty()) { ++nNoPeaks; return; }

    std::sort(peaks.begin(), peaks.end());
    float maxIt = 0.0f;
    for (auto& p : peaks) maxIt = std::max(maxIt, p.second);
    if (maxIt <= 0.0f) return;
    std::vector<float> mz, it;
    for (auto& p : peaks)
      if (p.second >= PEAK_FLOOR * maxIt) { mz.push_back(p.first); it.push_back(p.second); }
    if (mz.empty()) return;

    const std::string& name = itKey->second;
    auto itMol = molId.find(name);
    int mid;
    if (itMol == molId.end()) {
      mid = int(names.size());
      molId[name] = mid;
      names.push_back(name);
      smiles.push_back(smilesCur);
      ++nNewMols;
    } else {
      mid = itMol->second;
      if (smiles[mid].empty()) smiles[mid] = smilesCur;
    }

    peaksOut.write(reinterpret_cast<const char*>(mz.data()), mz.size() * 4);
    peaksOut.write(reinterpret_cast<const char*>(it.data()), it.size() * 4);
    int32_t k = int32_t(mz.size());
    int32_t off32 = int32_t(off);
    int32_t tag = adductIndex(adduct) * 16 + instrIndex(analyzer, ionisation);
    headOut.write(reinterpret_cast<const char*>(&pepmass), 4);
    headOut.write(reinterpret_cast<const char*>(&mid), 4);
    headOut.write(reinterpret_cast<const char*>(&off32), 4);
    headOut.write(reinterpret_cast<const char*>(&k), 4);
    headOut.write(reinterpret_cast<const char*>(&tag), 4);
    off += k;
    ++nKept;
  };

  while (std::getline(mgf, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line == "BEGIN IONS") {
      inRecord = true;
      specId.clear(); adduct.clear(); smilesCur.clear(); analyzer.clear(); ionisation.clear();
      pepmass = 0.0f;
      peaks.clear();
      continue;
    }
    if (line == "END IONS") {
      flushRecord();
      inRecord = false;
      if (nRead % 100000 == 0)
        std::fprintf(stderr, "  %lld read, %lld kept, %lld new molecules\n", nRead, nKept, nNewMols);
      continue;
    }
    if (!inRecord) continue;
    size_t eq = line.find('=');
    if (eq != std::string::npos) {
      std::string key = line.substr(0, eq), val = line.substr(eq + 1);
      if (key == "SPECTRUMID") specId = val;
      else if (key == "PEPMASS") pepmass = float(std::atof(val.c_str()));
      else if (key == "ADDUCT") adduct = val;
      else if (key == "SMILES") smilesCur = val;
      else if (key == "MS_MASS_ANALYZER") analyzer = val;
      else if (key == "MS_IONISATION") ionisation = val;
      continue;
    }
    // a bare "mz intensity" peak line
    std::istringstream iss(line);
    float mz, it;
    if (iss >> mz >> it) peaks.emplace_back(mz, it);
  }

  headOut.close();
  peaksOut.close();
  std::ofstream molsOut(dir + "/" + outStem + "_mols.txt");
  for (size_t i = 0; i < names.size(); ++i) {
    if (i) molsOut << '\n';
    molsOut << names[i] << '\t' << smiles[i];
  }

  std::fprintf(stderr,
      "\nGNPS: %lld records read, %lld kept (%lld no InChIKey match, %lld no SMILES, "
      "%lld no peaks survived the floor), %lld new molecules\n",
      nRead, nKept, nNoKey, nNoSmiles, nNoPeaks, nNewMols);
  std::fprintf(stderr, "wrote %s_head.bin/_peaks.bin/_mols.txt: %lld total spectra, %zu total molecules\n",
               outStem.c_str(), nTrainSpectra + nKept, names.size());
  return 0;
}
