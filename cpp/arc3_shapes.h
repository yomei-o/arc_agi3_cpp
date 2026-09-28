// Shapes, and which of them fit which gaps.
//
// The user's first observation about this whole family of games, made before
// any of this was built: "ほとんどは色合わせというか位置合わせ。どれとどれの位置を
// 合わせるのか" - most of them are matching, and the question is which thing goes
// with which. The 27B said the same thing unprompted about cd82: move the G
// object to align with the - object.
//
// So that question should cost one lookup and not an afternoon of the model
// counting characters in an ASCII board. Everything here is a pure function of
// the current frame: it spends no actions, and actions are the only thing the
// score charges for.
//
// Included from arc3_lua.h, after Grid / background_colors().

#ifndef ARC3_SHAPES_H
#define ARC3_SHAPES_H

#include <algorithm>
#include <map>
#include <string>
#include <vector>

namespace arc3lua {

struct Piece {
  int colour = 0, area = 0;
  int cx = 0, cy = 0, w = 0, h = 0, minx = 0, miny = 0;
  bool background = false;
  std::string key;       // the shape, normalised to its bounding box
};

// Every 4-connected region of one colour, background included.
//
// components() in the core skips background colours, which is right when you
// are looking for things to click and wrong when you are looking for the hole
// a thing fits into. A hole is made of background by definition.
inline std::vector<Piece> pieces(const Grid& g, int scale) {
  const int bw = W / scale, bh = H / scale;
  std::set<int> bg = background_colors(g);
  std::vector<uint8_t> seen(size_t(bw) * size_t(bh), 0);
  std::vector<Piece> out;

  std::vector<int> stack, cells;
  for (int y0 = 0; y0 < bh; ++y0) {
    for (int x0 = 0; x0 < bw; ++x0) {
      size_t start = size_t(y0) * size_t(bw) + size_t(x0);
      if (seen[start]) continue;
      int col = int(g.at(x0 * scale, y0 * scale));
      seen[start] = 1;
      stack.clear();
      cells.clear();
      stack.push_back(int(start));
      while (!stack.empty()) {
        int cur = stack.back();
        stack.pop_back();
        cells.push_back(cur);
        int cx = cur % bw, cy = cur / bw;
        const int dx[4] = {1, -1, 0, 0}, dy[4] = {0, 0, 1, -1};
        for (int d = 0; d < 4; ++d) {
          int nx = cx + dx[d], ny = cy + dy[d];
          if (nx < 0 || ny < 0 || nx >= bw || ny >= bh) continue;
          size_t ni = size_t(ny) * size_t(bw) + size_t(nx);
          if (seen[ni]) continue;
          if (int(g.at(nx * scale, ny * scale)) != col) continue;
          seen[ni] = 1;
          stack.push_back(int(ni));
        }
      }
      if (cells.size() > size_t(bw) * size_t(bh) / 4) continue;   // scenery

      Piece p;
      p.colour = col;
      p.background = bg.count(col) != 0;
      p.area = int(cells.size());
      int mnx = bw, mny = bh, mxx = -1, mxy = -1;
      for (size_t i = 0; i < cells.size(); ++i) {
        int cx = cells[i] % bw, cy = cells[i] / bw;
        mnx = std::min(mnx, cx); mxx = std::max(mxx, cx);
        mny = std::min(mny, cy); mxy = std::max(mxy, cy);
      }
      p.minx = mnx; p.miny = mny;
      p.w = mxx - mnx + 1; p.h = mxy - mny + 1;
      p.cx = (mnx + mxx) / 2; p.cy = (mny + mxy) / 2;

      // The shape itself, as a row-major bitmap of the bounding box. Two
      // things match when this string matches, whatever their colour or
      // where they are.
      std::string key(size_t(p.w) * size_t(p.h), '.');
      for (size_t i = 0; i < cells.size(); ++i) {
        int cx = cells[i] % bw - mnx, cy = cells[i] / bw - mny;
        key[size_t(cy) * size_t(p.w) + size_t(cx)] = '#';
      }
      p.key = key;
      out.push_back(p);
    }
  }
  return out;
}

struct Match {
  const Piece* thing;
  const Piece* gap;
};

// Pairs of the same shape where one is made of background and the other is not:
// a piece and a hole it would fill.
inline std::vector<Match> fits(const std::vector<Piece>& ps, size_t limit = 12) {
  std::vector<Match> out;
  for (size_t i = 0; i < ps.size() && out.size() < limit; ++i) {
    if (ps[i].background || ps[i].area < 2) continue;
    for (size_t j = 0; j < ps.size(); ++j) {
      if (!ps[j].background || ps[j].key != ps[i].key) continue;
      if (ps[j].cx == ps[i].cx && ps[j].cy == ps[i].cy) continue;
      Match m;
      m.thing = &ps[i];
      m.gap = &ps[j];
      out.push_back(m);
      break;
    }
  }
  return out;
}

// Groups of two or more identical shapes, and the one that breaks the pattern.
inline std::map<std::string, std::vector<const Piece*> >
group_by_shape(const std::vector<Piece>& ps) {
  std::map<std::string, std::vector<const Piece*> > by;
  for (size_t i = 0; i < ps.size(); ++i)
    if (!ps[i].background && ps[i].area >= 2)
      by[ps[i].key].push_back(&ps[i]);
  return by;
}

}  // namespace arc3lua

#endif
