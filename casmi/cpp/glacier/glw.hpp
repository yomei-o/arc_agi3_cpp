// glw.hpp: named float32 array files ("GLW1", u32 n; per array: u32 name len, name, u32 ndim, u32 dims[ndim],
// float32 data row-major) - the GLACIER weights (glacier_w.bin, written by gl_ref.py) and the reference dumps.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct GLArr {
  std::vector<uint32_t> dims;
  std::vector<float> v;
  size_t size() const { return v.size(); }
  uint32_t d(int i) const { return dims.at(i); }
};

inline std::map<std::string, GLArr> glw_load(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot read " + path);
  char magic[4];
  uint32_t n = 0;
  if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "GLW1", 4) != 0 || std::fread(&n, 4, 1, f) != 1) {
    std::fclose(f);
    throw std::runtime_error("bad GLW1 file " + path);
  }
  std::map<std::string, GLArr> out;
  for (uint32_t k = 0; k < n; k++) {
    uint32_t l = 0, nd = 0;
    if (std::fread(&l, 4, 1, f) != 1) break;
    std::string name(l, '\0');
    if (std::fread(name.data(), 1, l, f) != l || std::fread(&nd, 4, 1, f) != 1) break;
    GLArr a;
    a.dims.resize(nd);
    if (nd && std::fread(a.dims.data(), 4, nd, f) != nd) break;
    size_t cnt = 1;
    for (uint32_t d : a.dims) cnt *= d;
    a.v.resize(cnt);
    if (cnt && std::fread(a.v.data(), 4, cnt, f) != cnt) break;
    out[name] = std::move(a);
  }
  std::fclose(f);
  return out;
}
