// arc3_core.cpp - ARC-AGI-3 agent core. Zero dependencies, C++17, CPU only.
//
// The Python side owns only the HTTP/env loop: it hands us each 64x64 frame and
// asks for the next action. Everything that constitutes thinking - object
// extraction, working out which sprite the keys move, learning which cells
// block, planning a route, replaying the move that finished the last level -
// happens here.
//
// Why this lives in C++: the competition scores each completed level as
// (human_actions / agent_actions)^2, so a real action is expensive while
// deliberation is free. We want to search a lot and act rarely, which is
// exactly the trade C++ buys us.
//
// What the frames actually look like (measured on the public games): a 64x64
// rendered view in which the logical board is drawn at a scale of several
// pixels per cell, surrounded by HUD furniture - side borders, an inventory
// panel, a remaining-actions bar - that animates on its own every step. Two
// consequences drive the design below:
//   * "the frame did not change" is useless as a collision signal, because the
//     HUD always changes. We compare the avatar's predicted position with where
//     it actually ended up instead.
//   * the avatar's colour is reused by the HUD, so it cannot be re-located by
//     matching colour and shape. We track it by integrating the translations we
//     actually observe, and re-acquire it from motion when we lose it.
//
// Build (shared lib):
//   g++ -O2 -std=c++17 -shared -fPIC arc3_core.cpp -o arc3_core.so
//   g++ -O2 -std=c++17 -shared     arc3_core.cpp -o arc3_core.dll   (w64devkit)

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <vector>

#include "arc3_net.h"

#if defined(_WIN32)
#define ARC3_API extern "C" __declspec(dllexport)
#else
#define ARC3_API extern "C" __attribute__((visibility("default")))
#endif

namespace {

constexpr int W = 64, H = 64;
constexpr int NCELL = W * H;

// Action encoding shared with the Python side (matches GameAction values).
enum : int { A_RESET = 0, A1 = 1, A2 = 2, A3 = 3, A4 = 4, A5 = 5, A6 = 6, A7 = 7 };

enum : uint8_t { UNKNOWN = 0, FREE = 1, BLOCKED = 2 };

struct Vec {
  int dx, dy;
  Vec() : dx(0), dy(0) {}
  Vec(int x, int y) : dx(x), dy(y) {}
  bool operator<(const Vec& o) const { return dx != o.dx ? dx < o.dx : dy < o.dy; }
  bool operator==(const Vec& o) const { return dx == o.dx && dy == o.dy; }
  bool zero() const { return dx == 0 && dy == 0; }
};

struct Grid {
  std::array<int8_t, NCELL> c;
  Grid() { c.fill(0); }
  int8_t at(int x, int y) const { return c[y * W + x]; }
  bool operator==(const Grid& o) const { return c == o.c; }
};

struct Box {
  int minx, miny, maxx, maxy, area;
  Box() : minx(0), miny(0), maxx(0), maxy(0), area(0) {}
  int w() const { return maxx - minx + 1; }
  int h() const { return maxy - miny + 1; }
  int cx() const { return (minx + maxx) / 2; }
  int cy() const { return (miny + maxy) / 2; }
};

// Colours covering a large share of the board are scenery, not objects.
std::set<int> background_colors(const Grid& g) {
  std::array<int, 32> hist;
  hist.fill(0);
  for (int i = 0; i < NCELL; ++i) {
    int v = g.c[i];
    if (v >= 0 && v < 32) ++hist[v];
  }
  std::set<int> bg;
  for (int i = 0; i < 32; ++i)
    if (hist[i] > NCELL / 8) bg.insert(i);
  return bg;
}

// Flood fill the same-colour region containing `idx`.
Box component_at(const Grid& g, int idx, std::vector<int>* cells) {
  Box b;
  int col = g.c[idx];
  std::vector<uint8_t> seen(NCELL, 0);
  std::vector<int> stack;
  stack.push_back(idx);
  seen[idx] = 1;
  b.minx = b.maxx = idx % W;
  b.miny = b.maxy = idx / W;
  while (!stack.empty()) {
    int j = stack.back();
    stack.pop_back();
    int x = j % W, y = j / W;
    ++b.area;
    if (cells) cells->push_back(j);
    b.minx = std::min(b.minx, x);
    b.maxx = std::max(b.maxx, x);
    b.miny = std::min(b.miny, y);
    b.maxy = std::max(b.maxy, y);
    const int nb[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int d = 0; d < 4; ++d) {
      int nx = x + nb[d][0], ny = y + nb[d][1];
      if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
      int k = ny * W + nx;
      if (seen[k] || g.c[k] != col) continue;
      seen[k] = 1;
      stack.push_back(k);
    }
  }
  return b;
}

std::vector<Box> components(const Grid& g, const std::set<int>& bg, std::vector<int>* colors) {
  std::vector<Box> out;
  std::array<uint8_t, NCELL> seen;
  seen.fill(0);
  std::vector<int> stack;
  for (int i = 0; i < NCELL; ++i) {
    if (seen[i]) continue;
    int col = g.c[i];
    if (bg.count(col)) { seen[i] = 1; continue; }
    Box b;
    b.minx = b.maxx = i % W;
    b.miny = b.maxy = i / W;
    stack.clear();
    stack.push_back(i);
    seen[i] = 1;
    while (!stack.empty()) {
      int j = stack.back();
      stack.pop_back();
      int x = j % W, y = j / W;
      ++b.area;
      b.minx = std::min(b.minx, x);
      b.maxx = std::max(b.maxx, x);
      b.miny = std::min(b.miny, y);
      b.maxy = std::max(b.maxy, y);
      const int nb[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
      for (int d = 0; d < 4; ++d) {
        int nx = x + nb[d][0], ny = y + nb[d][1];
        if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
        int k = ny * W + nx;
        if (seen[k] || g.c[k] != col) continue;
        seen[k] = 1;
        stack.push_back(k);
      }
    }
    out.push_back(b);
    if (colors) colors->push_back(col);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Knowing when we are going in circles.
//
// The frame carries a remaining-actions bar that advances every step whatever
// we do, so raw frames are never equal and "have I been here before?" cannot be
// answered by comparing them. We find those clock cells the same way the
// offline solver does - cells that change on nearly every step - blank them,
// and hash what is left. Repeats of the same hash mean the current strategy has
// stopped producing anything and it is time to try a different kind of action.
// ---------------------------------------------------------------------------
struct NoveltyIndex {
  std::array<int, NCELL> changes;  // per-cell change count
  int observations;
  std::map<uint64_t, int> seen;

  NoveltyIndex() : observations(0) { changes.fill(0); }

  void note_change(const Grid& a, const Grid& b) {
    for (int i = 0; i < NCELL; ++i)
      if (a.c[i] != b.c[i]) ++changes[i];
    ++observations;
  }

  bool is_clock(int i) const {
    // Needs enough evidence before we start blanking anything.
    return observations >= 24 && changes[i] * 10 >= observations * 9;
  }

  uint64_t hash(const Grid& g, int levels) const {
    uint64_t h = 1469598103934665603ULL ^ uint64_t(levels) * 1099511628211ULL;
    for (int i = 0; i < NCELL; ++i) {
      uint8_t v = is_clock(i) ? 0 : uint8_t(g.c[i]);
      h ^= v;
      h *= 1099511628211ULL;
    }
    return h;
  }

  // How many times we have been in this state before.
  int visit(const Grid& g, int levels) { return seen[hash(g, levels)]++; }
};

// What happens when the avatar tries to step onto a square of a given colour.
//
// This is the part that is meant to survive into a game nobody has seen. The
// meaning of a game is not knowable in advance, but "I walked into a red square
// and the level ended" is measurable, and the levels of one game share their
// mechanics. Learn the table on level 1 - which is weighted 1 out of 28 and so
// costs almost nothing to spend - and levels 2..N can be walked straight to
// their goal, which is where the weight, and the score, actually is.
struct ColorRule {
  int blocked, passed, collected, completed;
  ColorRule() : blocked(0), passed(0), collected(0), completed(0) {}
  int trials() const { return blocked + passed + collected + completed; }
  bool is_wall() const { return blocked > 0 && passed + collected + completed == 0; }
  bool is_goal() const { return completed > 0; }
  bool is_pickup() const { return collected > passed; }
};

// ---------------------------------------------------------------------------
// A learned model of the game, and search inside it.
//
// Offline, where deepcopy gives a perfect simulator, search solves 15 of 17
// public games and finds routes shorter than the human baselines. Online there
// is no simulator, and porting the search does not work: a restore costs a
// RESET plus replaying the route, so at ten times the real action budget it
// scores worse than doing nothing clever. What has to cross over is not the
// search but a model to search inside.
//
// The games all run on the same sprite engine, so the model is object-centric:
// a scene is a set of (colour, width, height) objects at positions, and an
// action displaces each class by an amount we learn by watching. Prediction is
// then cheap enough to run a beam search every step and act on its first move,
// re-planning once reality has had its say (MPC), which is what keeps an
// imperfect model useful.
// ---------------------------------------------------------------------------
struct ObjClass {
  int color, w, h;
  ObjClass() : color(0), w(0), h(0) {}
  ObjClass(int c, int w_, int h_) : color(c), w(w_), h(h_) {}
  bool operator<(const ObjClass& o) const {
    if (color != o.color) return color < o.color;
    if (w != o.w) return w < o.w;
    return h < o.h;
  }
  bool operator==(const ObjClass& o) const {
    return color == o.color && w == o.w && h == o.h;
  }
};

struct Obj {
  ObjClass k;
  int x, y;
  Obj() : x(0), y(0) {}
  Obj(const ObjClass& k_, int x_, int y_) : k(k_), x(x_), y(y_) {}
};

struct Scene {
  std::vector<Obj> objs;

  uint64_t hash() const {
    // Order-independent, so the same arrangement hashes alike however the
    // objects came out of the extractor.
    uint64_t h = 0;
    for (size_t i = 0; i < objs.size(); ++i) {
      uint64_t v = 1469598103934665603ULL;
      const Obj& o = objs[i];
      int f[5] = {o.k.color, o.k.w, o.k.h, o.x, o.y};
      for (int j = 0; j < 5; ++j) { v ^= uint64_t(f[j] + 1); v *= 1099511628211ULL; }
      h += v;
    }
    return h;
  }
};

Scene scene_of(const Grid& g, const std::set<int>& bg, int max_objects = 96) {
  Scene s;
  std::vector<int> cols;
  std::vector<Box> comps = components(g, bg, &cols);
  for (size_t i = 0; i < comps.size() && int(s.objs.size()) < max_objects; ++i) {
    // Scenery spans the board; it is not something an action moves.
    if (comps[i].w() > 24 || comps[i].h() > 24) continue;
    s.objs.push_back(Obj(ObjClass(cols[i], comps[i].w(), comps[i].h()),
                         comps[i].minx, comps[i].miny));
  }
  return s;
}

struct WorldModel {
  // (action, class) -> how far that class moved, and how often.
  std::map<std::pair<int, ObjClass>, std::map<Vec, int> > disp;
  std::map<ObjClass, int> seen_class;
  int observations;

  WorldModel() : observations(0) {}

  Vec best(int action, const ObjClass& k, int* count) const {
    std::map<std::pair<int, ObjClass>, std::map<Vec, int> >::const_iterator it =
        disp.find(std::make_pair(action, k));
    if (it == disp.end()) { if (count) *count = 0; return Vec(); }
    Vec b; int bc = 0;
    for (std::map<Vec, int>::const_iterator j = it->second.begin(); j != it->second.end(); ++j)
      if (j->second > bc) { bc = j->second; b = j->first; }
    if (count) *count = bc;
    return b;
  }

  // Match objects between two scenes by class and proximity, and record what
  // the action did to each class.
  void learn(const Scene& a, const Scene& b, int action) {
    ++observations;
    std::vector<uint8_t> taken(b.objs.size(), 0);
    for (size_t i = 0; i < a.objs.size(); ++i) {
      const Obj& o = a.objs[i];
      ++seen_class[o.k];
      int best_j = -1, best_d = 1 << 30;
      for (size_t j = 0; j < b.objs.size(); ++j) {
        if (taken[j] || !(b.objs[j].k == o.k)) continue;
        int d = std::abs(b.objs[j].x - o.x) + std::abs(b.objs[j].y - o.y);
        if (d < best_d) { best_d = d; best_j = int(j); }
      }
      if (best_j < 0 || best_d > 24) continue;   // vanished, or not the same thing
      taken[best_j] = 1;
      Vec v(b.objs[best_j].x - o.x, b.objs[best_j].y - o.y);
      disp[std::make_pair(action, o.k)][v] += 1;
    }
  }

  // What we think the scene becomes. Classes we have never seen move are left
  // where they are, which is right far more often than not.
  //
  // `blocked` says which board squares refuse the avatar, from the colour table
  // learned by walking into things. Without it the model happily predicts
  // walking through walls, and a search rewarding novelty then chases states
  // that cannot happen - which is exactly how the first version of this scored
  // worse than no model at all.
  Scene predict(const Scene& s, int action, const std::vector<uint8_t>* blocked) const {
    Scene out;
    out.objs.reserve(s.objs.size());
    for (size_t i = 0; i < s.objs.size(); ++i) {
      int c = 0;
      Vec v = best(action, s.objs[i].k, &c);
      Obj o = s.objs[i];
      if (c > 0 && !v.zero()) {
        int nx = o.x + v.dx, ny = o.y + v.dy;
        bool ok = nx >= 0 && ny >= 0 && nx + o.k.w <= W && ny + o.k.h <= H;
        if (ok && blocked) {
          for (int dy = 0; dy < o.k.h && ok; ++dy)
            for (int dx = 0; dx < o.k.w && ok; ++dx)
              if ((*blocked)[(ny + dy) * W + (nx + dx)]) ok = false;
        }
        if (ok) { o.x = nx; o.y = ny; }
      }
      out.objs.push_back(o);
    }
    return out;
  }

  bool usable() const { return observations >= 12 && !disp.empty(); }
};

// One issued action, so a route through a level can be replayed.
struct Act {
  int a, x, y;
  Act() : a(0), x(0), y(0) {}
  Act(int a_, int x_, int y_) : a(a_), x(x_), y(y_) {}
};

// A state we reached inside the current level, and the cheapest way back to it.
struct ArcEntry {
  std::vector<Act> traj;
  int visits;
  ArcEntry() : visits(0) {}
};

// Pixels per logical cell. The 64x64 frame is a rendered view - LS20 draws its
// board at 5 px per cell, TU93 at 3, WA30 at 4 - so a sprite is never smaller
// than about one cell. Without this the "smallest moved shape" rule picks up
// single-pixel specks: on TU93 it adopted a 1x1 dot of colour 4 as the avatar
// while the actual player, a 3x3 block of colour 9, went untracked.
int detect_scale(const Grid& g) {
  std::array<int, 9> hits;
  hits.fill(0);
  int edges = 0;
  for (int y = 0; y < H; ++y)
    for (int x = 1; x < W; ++x)
      if (g.c[y * W + x] != g.c[y * W + x - 1]) {
        ++edges;
        for (int s = 2; s <= 8; ++s) if (x % s == 0) ++hits[s];
      }
  for (int x = 0; x < W; ++x)
    for (int y = 1; y < H; ++y)
      if (g.c[y * W + x] != g.c[(y - 1) * W + x]) {
        ++edges;
        for (int s = 2; s <= 8; ++s) if (y % s == 0) ++hits[s];
      }
  if (edges < 16) return 1;
  for (int s = 8; s >= 2; --s)
    if (hits[s] * 10 >= edges * 9) return s;   // nearly every edge lands on the grid
  return 1;
}

// A translation of one colour between two frames.
struct Motion {
  int color;
  Vec disp;
  int anchor;  // a cell of the moved shape in the NEW frame
  bool found;
  Motion() : color(-1), anchor(-1), found(false) {}
};

// An avatar is a small sprite. Anything bigger than this is scenery, a camera
// pan or a whole-row repaint, and adopting it as the avatar is how the agent
// previously ended up "controlling" a 46x9 object and marking more squares
// blocked than the board has.
constexpr int MAX_AVATAR_CELLS = 100;
constexpr int MAX_AVATAR_SIDE = 16;

// Explain part of a frame transition as "a shape of one colour translated".
// `only_color` < 0 means "consider every colour"; the SMALLEST plausible moved
// shape wins, because that is what an avatar looks like.
Motion detect_translation(const Grid& a, const Grid& b, const std::set<int>& bg,
                          int only_color, int scale) {
  Motion best;
  int best_area = 1 << 30;
  const int min_area = std::max(4, scale * scale / 2);

  std::map<int, std::vector<int> > vanished, appeared;
  int ndiff = 0;
  for (int i = 0; i < NCELL; ++i) {
    if (a.c[i] == b.c[i]) continue;
    if (++ndiff > 1200) return best;
    int oa = a.c[i], ob = b.c[i];
    if (!bg.count(oa) && (only_color < 0 || oa == only_color)) vanished[oa].push_back(i);
    if (!bg.count(ob) && (only_color < 0 || ob == only_color)) appeared[ob].push_back(i);
  }

  for (std::map<int, std::vector<int> >::iterator kv = vanished.begin();
       kv != vanished.end(); ++kv) {
    int col = kv->first;
    std::map<int, std::vector<int> >::iterator ai = appeared.find(col);
    if (ai == appeared.end()) continue;
    const std::vector<int>& from = kv->second;
    const std::vector<int>& to = ai->second;
    if (from.size() != to.size() || from.empty()) continue;
    if (int(from.size()) < min_area) continue;      // a speck, not a sprite
    if (int(from.size()) > MAX_AVATAR_CELLS * 2) continue;
    if (int(from.size()) >= best_area) continue;

    // The changed cells must sit in a sprite-sized box, not span the board.
    int mnx = W, mny = H, mxx = -1, mxy = -1;
    for (size_t k = 0; k < from.size(); ++k) {
      int x = from[k] % W, y = from[k] / W;
      mnx = std::min(mnx, x); mxx = std::max(mxx, x);
      mny = std::min(mny, y); mxy = std::max(mxy, y);
    }
    if (mxx - mnx + 1 > MAX_AVATAR_SIDE * 2 || mxy - mny + 1 > MAX_AVATAR_SIDE * 2) continue;

    int fx = from[0] % W, fy = from[0] / W;
    for (size_t t = 0; t < to.size(); ++t) {
      Vec v(to[t] % W - fx, to[t] / W - fy);
      if (v.zero() || std::abs(v.dx) > 16 || std::abs(v.dy) > 16) continue;
      // Every cell the colour left must be matched by the same colour, shifted.
      bool ok = true;
      for (size_t k = 0; k < from.size() && ok; ++k) {
        int nx = from[k] % W + v.dx, ny = from[k] / W + v.dy;
        if (nx < 0 || ny < 0 || nx >= W || ny >= H) { ok = false; break; }
        if (b.c[ny * W + nx] != col) ok = false;
      }
      if (ok) {
        best.color = col;
        best.disp = v;
        best.anchor = (fy + v.dy) * W + (fx + v.dx);
        best.found = true;
        best_area = int(from.size());
        break;
      }
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// The agent.
// ---------------------------------------------------------------------------
struct Agent {
  std::mt19937 rng;
  std::vector<int> avail;
  bool has6;

  Grid prev, cur;
  bool have_prev;
  int steps;
  int levels;

  std::vector<int> calib_queue;

  // Avatar model.
  int av_color;
  int av_w, av_h;
  std::map<int, std::map<Vec, int> > action_moves;  // action -> displacement -> count
  int av_support;
  bool avatar_known;
  int ax, ay;  // avatar bounding-box top-left, -1 when lost

  // Where the avatar should have ended up after the action we just issued.
  bool expect_valid;
  int expect_x, expect_y;

  std::array<uint8_t, NCELL> known;
  std::array<uint8_t, NCELL> visited;
  std::array<uint8_t, NCELL> probed5;
  std::array<uint8_t, NCELL> touched;   // objects we have already walked to
  std::set<int> clicked;                // coordinates that stopped paying off

  // A click target we are still working. The solved click-only games are full
  // of the same coordinate pressed several times in a row - S5I5's level 1 is
  // thirteen clicks alternating between two points - so a policy that visits
  // each object once and never returns cannot solve them at all.
  int click_x, click_y;
  bool click_live;


  std::vector<int> plan;
  int last_action;
  int pending_x, pending_y;

  // What immediately preceded the last level completion. Levels of one game
  // share their mechanics, so the move that finished level 1 is the strongest
  // hint available for level 2 - and level 2 is worth twice as much, because
  // the per-game score weights a level by its number.
  enum { TRIG_NONE = 0, TRIG_STAND = 1, TRIG_CLICK = 2 };
  int trigger_kind;
  int trigger_action;
  int trigger_under;     // TRIG_STAND: colour the avatar was standing on
  int trig_col, trig_area, trig_w, trig_h;   // TRIG_CLICK: what we clicked
  bool trigger_valid;
  int last_under;

  int stagnant;
  int blocked_marks;
  int reacquire;  // countdown of probe moves used to find the avatar again

  // 0 = directed (plan routes to frontiers and objects)
  // 1 = sticky-random (what the offline solver actually uses)
  // 2 = directed, falling back to sticky-random once it runs dry
  // 3 = systematic Go-Explore: the offline algorithm, run for real - every
  //     `rollout_len` actions, go back to an archived state and explore from
  //     there, instead of only doing so on death or when circling
  // 4 = plan inside the learned model, act on the first move, re-plan (MPC)
  int explore_mode;
  int alpha_objects;   // how many object centroids the search may click
  int alpha_grid;      // plus this many coarse-grid points, for background hits
  int depth_cap;       // how far from the level start the search may wander
  int forget_mode;     // see the note where it is used
  int trigger_replay;  // see the note where it is used
  int clear_stats_on_level;
  int budget;          // actions this game allows, for pacing the ensemble
  int phase_mode;      // which policy the ensemble is currently running
  int rollout_len;
  int since_restart;

  // Iterative deepening over action sequences, in the real game.
  //
  // The compressed offline solutions say these levels are not deep: VC33 and
  // R11L finish level 1 in two actions, SP80 in three, CD82 in four. They are
  // hard to FIND, not hard to execute. And a mid-game RESET restarts only the
  // current level while keeping the levels already finished, so a sequence can
  // be tried, abandoned, and the next one tried, for the cost of the sequence
  // plus one action. Depth 4 over four actions is about 1,280 actions - inside
  // the roughly 3,000 a game like LS20 allows.
  struct BfsNode {
    std::vector<Act> traj;
    int next_action;
    BfsNode() : next_action(0) {}
  };
  std::vector<Act> idd_actions;   // the alphabet we search over
  std::vector<BfsNode> bfs;       // the queue, in breadth-first order
  size_t bfs_head;
  std::set<uint64_t> bfs_seen;    // one entry per distinct state reached
  std::vector<Act> idd_played;    // the route we are currently walking
  std::vector<Act> idd_solution;  // the route that finished the last level
  int bfs_phase;                  // 0 reset, 1 replay, 2 probe
  int bfs_replay_pos;
  bool idd_active, idd_try_solution;
  int idd_pos;
  long idd_tried, idd_pruned;

  // Mode 6: expand wherever we are standing.
  //
  // A breadth-first search that resets before every probe pays one RESET plus
  // the whole route for each action it tries - about eight actions per probe on
  // LS20, which buys only 54 states out of 1500 actions. But while the search
  // is moving forward it is already standing in the state it wants to expand,
  // so a probe costs exactly one action. Only a dead end needs a restore.
  // Mode 9: the official sample's policy, learned rather than counted.
  //
  // Mode 8 counts how often each action and each coordinate changed the frame.
  // That has no idea what the board looks like - a coordinate that works only
  // when a door is open gets one number for both cases. The sample instead
  // trains a CNN conditioned on the frame, which can say "this spot works NOW".
  // Same target (did the frame change?), same sampling, a network small enough
  // for a CPU (see arc3_net.h).
  struct Exp {
    std::array<int8_t, 32 * 32> s32;
    int action_idx;
    float reward;
  };
  arc3net::ActionNet net;

  // Mode 11: Double DQN over an intrinsic reward, built the way mario_dqn_cpp
  // builds one - replay, a target network, and the online net choosing the
  // argmax while the target net prices it.
  //
  // The point is to fix what the sample's predictor cannot express. That CNN
  // answers "does THIS action change the frame", which is myopic: it cannot
  // prefer a move that does nothing now but opens something up two steps later.
  // A discounted Q can. The game's own reward is far too sparse to learn from -
  // most games never finish a level at all - so the signal is novelty: reaching
  // an arrangement of objects never seen before pays, repeating one does not.
  arc3net::ActionNet qtgt;
  struct Tr {
    std::array<int8_t, 32 * 32> s, ns;
    int a;
    float r;
    bool done;
  };
  std::vector<Tr> qbuf;
  size_t qhead;
  int q_sync_every;
  float q_gamma;

  std::vector<Exp> replay;
  std::set<uint64_t> replay_seen;
  std::array<int8_t, 32 * 32> prev_s32;
  bool have_prev_s32;
  int prev_action_idx;
  int train_every, train_batch;

  // Mode 8: a tabular version of what the official sample learns.
  //
  // StochasticGoose trains a CNN to predict which actions produce a NEW frame,
  // and samples from that. Almost all of what such a net can express here is
  // "this action, or this coordinate, tends to do something", and that is a
  // table: how often each simple action and each of the 4,096 coordinates
  // changed the frame, and how often it was tried. Sampling by the posterior
  // mean of that gives the same behaviour without a network, which matters
  // because the agent has to keep up with an HTTP round trip per action.
  std::array<int, NCELL> click_tries, click_change;
  std::array<int, 8> act_tries, act_change;
  int last_click_idx;

  std::map<uint64_t, std::vector<Act> > ge_route;   // cheapest route to a state
  std::map<uint64_t, int> ge_tried;                 // alphabet entries tried there
  uint64_t ge_here;
  bool ge_ready;   // set in the constructor body; declaration order differs

  WorldModel wm;
  Scene cur_scene;
  bool have_scene;
  bool scene_is_new;   // was the arrangement in `cur` one we had not seen?
  std::set<uint64_t> scene_seen;   // arrangements we have actually been in
  int sticky_a, sticky_x, sticky_y;

  std::map<int, ColorRule> rules;   // colour stepped onto -> what happened
  int rule_target;                  // colour of the square we are stepping onto

  NoveltyIndex novelty;
  int repeats;    // consecutive choices made from an already-seen state
  int escalation; // 0 walk, 1 interact everywhere, 2 click, 3 restart the level

  // Go-Explore inside the current level. RESET restarts the level and keeps the
  // levels already finished, so "go back to a promising state and try again" is
  // available even here, where there is no simulator to fork - we pay for it in
  // real actions by replaying the route.
  int last_state;                             // 0 not played, 1 playing, 2 win, 3 over
  std::vector<Act> level_traj;                // actions since this level began
  std::map<uint64_t, ArcEntry> level_archive;
  std::vector<Act> replay_queue;
  bool replaying;
  int restarts;

  Agent()
      : rng(12345), has6(false), have_prev(false), steps(0), levels(0),
        av_color(-1), av_w(0), av_h(0), av_support(0), avatar_known(false),
        ax(-1), ay(-1), expect_valid(false), expect_x(-1), expect_y(-1),
        last_action(-1), pending_x(0), pending_y(0),
        trigger_kind(TRIG_NONE), trigger_action(-1), trigger_under(-1),
        trig_col(-1), trig_area(0), trig_w(0), trig_h(0), trigger_valid(false),
        last_under(-1), stagnant(0), blocked_marks(0), reacquire(0),
        bfs_head(0), bfs_phase(0), bfs_replay_pos(0),
        idd_active(false), idd_try_solution(false), ge_here(0),
        explore_mode(2), alpha_objects(24), alpha_grid(8), depth_cap(12),
        budget(4000), phase_mode(0), scene_is_new(false), forget_mode(0),
        trigger_replay(0), clear_stats_on_level(1),
        rollout_len(60), since_restart(0), have_scene(false),
        sticky_a(-1), sticky_x(0), sticky_y(0),
        rule_target(-1), repeats(0), escalation(0), last_state(0),
        replaying(false), restarts(0) {
    ge_ready = false;
    have_prev_s32 = false;
    prev_action_idx = -1;
    qhead = 0;
    q_sync_every = 200;
    q_gamma = 0.9f;
    train_every = 20;
    train_batch = 16;
    prev_s32.fill(0);
    click_x = click_y = -1;
    click_live = false;
    click_tries.fill(0);
    click_change.fill(0);
    act_tries.fill(0);
    act_change.fill(0);
    last_click_idx = -1;
    idd_pos = 0;
    idd_tried = 0;
    idd_pruned = 0;
    ge_here = 0;
    ge_ready = false;
    idd_pruned = 0;
    known.fill(UNKNOWN);
    visited.fill(0);
    probed5.fill(0);
    touched.fill(0);
  }

  void init(const int* actions, int n) {
    avail.clear();
    for (int i = 0; i < n; ++i) {
      if (actions[i] == A6) has6 = true;
      else if (actions[i] >= A1 && actions[i] <= A7) avail.push_back(actions[i]);
    }
    if (avail.empty() && !has6) {
      avail.push_back(A1); avail.push_back(A2); avail.push_back(A3); avail.push_back(A4);
    }
    queue_calibration();
  }

  void queue_calibration() {
    calib_queue.clear();
    for (int pass = 0; pass < 2; ++pass)
      for (size_t i = 0; i < avail.size(); ++i)
        if (avail[i] != A5 && avail[i] != A7) calib_queue.push_back(avail[i]);
  }

  bool in_bounds(int x, int y) const { return x >= 0 && y >= 0 && x < W && y < H; }

  // How much of the board a colour covers. Measured across the solved public
  // games, a level ends with the avatar standing on a colour covering under 2%
  // of the board in 14 of 19 cases, and the objects worth clicking are rare and
  // small too - so rarity is the best game-independent prior we have for "this
  // is the thing that matters".
  std::array<int, 32> color_counts() const {
    std::array<int, 32> h;
    h.fill(0);
    for (int i = 0; i < NCELL; ++i) {
      int v = cur.c[i];
      if (v >= 0 && v < 32) ++h[v];
    }
    return h;
  }

  Vec best_move(int action, int* count) const {
    std::map<int, std::map<Vec, int> >::const_iterator it = action_moves.find(action);
    if (it == action_moves.end()) { if (count) *count = 0; return Vec(); }
    Vec best; int bc = 0;
    for (std::map<Vec, int>::const_iterator k = it->second.begin(); k != it->second.end(); ++k)
      if (k->second > bc) { bc = k->second; best = k->first; }
    if (count) *count = bc;
    return best;
  }

  bool is_move_action(int a) const {
    int c = 0;
    Vec v = best_move(a, &c);
    return c > 0 && !v.zero();
  }

  // Adopt the moved shape as the avatar, if it is sprite-sized. A translation
  // can also come from a camera pan or a repainted row; those are not something
  // we control, and believing otherwise poisons the whole map.
  bool adopt(const Motion& m) {
    Box b = component_at(cur, m.anchor, 0);
    if (b.area > MAX_AVATAR_CELLS || b.w() > MAX_AVATAR_SIDE || b.h() > MAX_AVATAR_SIDE)
      return false;
    av_color = m.color;
    ax = b.minx;
    ay = b.miny;
    av_w = b.w();
    av_h = b.h();
    return true;
  }

  // Squares the avatar can stand on, in its own movement lattice.
  std::vector<int> plan_to(const std::vector<int>& targets) {
    std::vector<int> empty;
    if (!avatar_known || ax < 0 || targets.empty()) return empty;
    std::set<int> tgt(targets.begin(), targets.end());

    std::vector<int> moveActions;
    std::vector<Vec> moveVecs;
    for (size_t i = 0; i < avail.size(); ++i) {
      int a = avail[i];
      if (a == A5 || a == A6 || a == A7) continue;
      int cnt = 0;
      Vec v = best_move(a, &cnt);
      if (cnt > 0 && !v.zero()) { moveActions.push_back(a); moveVecs.push_back(v); }
    }
    if (moveActions.empty()) return empty;

    std::vector<int> par(NCELL, -1), pact(NCELL, -1);
    std::vector<uint8_t> vis(NCELL, 0);
    std::queue<int> q;
    int start = ay * W + ax;
    q.push(start);
    vis[start] = 1;
    int found = -1;
    while (!q.empty()) {
      int cell = q.front();
      q.pop();
      if (cell != start && tgt.count(cell)) { found = cell; break; }
      int x = cell % W, y = cell / W;
      for (size_t m = 0; m < moveActions.size(); ++m) {
        int nx = x + moveVecs[m].dx, ny = y + moveVecs[m].dy;
        if (!in_bounds(nx, ny)) continue;
        int k = ny * W + nx;
        if (vis[k] || known[k] == BLOCKED) continue;
        std::map<int, ColorRule>::const_iterator r = rules.find(cur.c[k]);
        if (r != rules.end() && r->second.is_wall()) continue;
        vis[k] = 1;
        par[k] = cell;
        pact[k] = moveActions[m];
        q.push(k);
      }
    }
    if (found < 0) return empty;
    std::vector<int> seq;
    for (int c = found; c != start && par[c] >= 0; c = par[c]) seq.push_back(pact[c]);
    std::reverse(seq.begin(), seq.end());
    if (seq.size() > 30) seq.resize(30);
    return seq;
  }

  // Squares whose colour has ended a level before. Once level 1 has taught us
  // this, the rest of the game is a shortest-path problem.
  std::vector<int> goal_targets() {
    std::vector<int> t;
    for (std::map<int, ColorRule>::iterator it = rules.begin(); it != rules.end(); ++it) {
      if (!it->second.is_goal()) continue;
      for (int i = 0; i < NCELL; ++i)
        if (cur.c[i] == it->first && !touched[i]) t.push_back(i);
    }
    return t;
  }

  // Squares of rare colours, rarest first - the strongest prior we have for
  // where a level ends when we have not yet seen one end.
  // Rare colours, rarest first.
  std::vector<int> rare_colours() {
    std::array<int, 32> h = color_counts();
    std::vector<std::pair<int, int> > order;   // (count, colour)
    for (int c = 0; c < 32; ++c)
      if (h[c] > 0 && h[c] < NCELL / 50 && c != av_color) order.push_back(std::make_pair(h[c], c));
    std::sort(order.begin(), order.end());
    std::vector<int> out;
    for (size_t k = 0; k < order.size(); ++k) {
      std::map<int, ColorRule>::const_iterator r = rules.find(order[k].second);
      if (r != rules.end() && r->second.is_wall()) continue;
      out.push_back(order[k].second);
    }
    return out;
  }

  std::vector<int> cells_of(int col) {
    std::vector<int> t;
    for (int i = 0; i < NCELL; ++i)
      if (cur.c[i] == col && !touched[i]) t.push_back(i);
    return t;
  }

  // A route to the nearest rare thing we can actually get to.
  //
  // This used to take only the single rarest colour and stop. On TU93 the
  // rarest colour on the board is the remaining-actions bar along the bottom
  // edge, which no route can reach - so the search returned nothing and the
  // marker that actually ends the level, the next-rarest colour, was never
  // tried at all. Rarity is a ranking, not a single answer.
  std::vector<int> plan_to_rare() {
    std::vector<int> cols = rare_colours();
    for (size_t i = 0; i < cols.size(); ++i) {
      std::vector<int> p = plan_to(cells_of(cols[i]));
      if (!p.empty()) return p;
    }
    return std::vector<int>();
  }

  std::vector<int> rare_targets() {
    std::vector<int> cols = rare_colours();
    std::vector<int> t;
    for (size_t i = 0; i < cols.size() && t.empty(); ++i) t = cells_of(cols[i]);
    return t;
  }

  // Things that vanished when walked into - collectables, most likely.
  std::vector<int> pickup_targets() {
    std::vector<int> t;
    for (std::map<int, ColorRule>::iterator it = rules.begin(); it != rules.end(); ++it) {
      if (!it->second.is_pickup() || it->second.is_goal()) continue;
      for (int i = 0; i < NCELL; ++i)
        if (cur.c[i] == it->first && !touched[i]) t.push_back(i);
    }
    return t;
  }

  // Lattice squares we have not stood on yet, nearest first via the BFS above.
  std::vector<int> frontier_targets() {
    std::vector<int> t;
    for (int i = 0; i < NCELL; ++i)
      if (!visited[i] && known[i] != BLOCKED) t.push_back(i);
    return t;
  }

  // Objects are far more informative than empty floor, so we walk to them
  // first. Anything the avatar cannot reach - the HUD, the inventory panel -
  // drops out on its own, because the route search never gets there.
  std::vector<int> object_targets(const std::set<int>& bg) {
    std::vector<int> t;
    for (int i = 0; i < NCELL; ++i) {
      int col = cur.c[i];
      if (bg.count(col) || col == av_color) continue;
      if (touched[i] || known[i] == BLOCKED) continue;
      t.push_back(i);
    }
    return t;
  }

  // Mark the object we just reached, and its whole shape, as dealt with.
  void mark_touched(int idx) {
    std::vector<int> cells;
    component_at(cur, idx, &cells);
    for (size_t i = 0; i < cells.size(); ++i) touched[cells[i]] = 1;
  }

  // Layout-specific knowledge only; the action model and the winning trigger
  // survive, because they describe the game rather than this particular level.
  void forget_map() {
    known.fill(UNKNOWN);
    visited.fill(0);
    probed5.fill(0);
    touched.fill(0);
    clicked.clear();
    plan.clear();
    // The click statistics were surviving into the next level, where the board
    // has been replaced wholesale: squares that paid off keep pulling the
    // sampler back to places that no longer exist, and squares written off in
    // the old layout stay written off in the new one. Forgetting on every new
    // arrangement was far too aggressive (it fires on every step of a movement
    // game); forgetting at a level boundary is the case it was meant for.
    // Second levels are where this shows: R11L finishes level 1 at 1x the human
    // baseline and level 2 at 167x.
    if (clear_stats_on_level) {
      click_tries.fill(0);
      click_change.fill(0);
      act_tries.fill(0);
      act_change.fill(0);
    }
    ax = ay = -1;
    expect_valid = false;
    reacquire = int(avail.size()) * 2;
  }

  // Pick somewhere worth going back to: rarely revisited, and cheap to reach.
  const ArcEntry* restart_target() {
    const ArcEntry* best = 0;
    double best_w = -1e18;
    for (std::map<uint64_t, ArcEntry>::iterator it = level_archive.begin();
         it != level_archive.end(); ++it) {
      if (int(it->second.traj.size()) > rollout_len) continue;  // too dear to replay
      double w = -double(it->second.visits) * 10.0
                 - double(it->second.traj.size()) * 0.2
                 + double(rng() % 1000) / 1000.0;
      if (w > best_w) { best_w = w; best = &it->second; }
    }
    return best;
  }

  void restart_from(const ArcEntry* e) {
    ++restarts;
    replay_queue.clear();
    if (e) {
      replay_queue = e->traj;
      const_cast<ArcEntry*>(e)->visits += 1;
    }
    replaying = !replay_queue.empty();
    escalation = 0;
    repeats = 0;
  }

  void on_new_level() {
    // Geometry changed; keep the action model and the winning trigger, drop the
    // map and re-acquire the avatar from its next movement.
    known.fill(UNKNOWN);
    visited.fill(0);
    probed5.fill(0);
    touched.fill(0);
    clicked.clear();
    plan.clear();
    ax = ay = -1;
    expect_valid = false;
    reacquire = int(avail.size()) * 2;
    level_traj.clear();
    level_archive.clear();
    replay_queue.clear();
    replaying = false;
    ge_route.clear();
    ge_tried.clear();
    ge_ready = false;
    if (explore_mode == 11) {
      net.reset();
      qtgt.copy_from(net);
      qbuf.clear();
      qhead = 0;
      have_prev_s32 = false;
    }
    if (explore_mode == 9 || explore_mode == 10) {
      // A new level is a different distribution; the sample rebuilds its
      // network here and so do we.
      net.reset();
      replay.clear();
      replay_seen.clear();
      have_prev_s32 = false;
    }
    if (explore_mode == 5 || explore_mode == 6) idd_begin();
  }

  void observe(const int8_t* frame, int levels_completed, int state) {
    last_state = state;
    Grid g;
    std::memcpy(g.c.data(), frame, NCELL);
    prev = cur;
    cur = g;
    bool changed = have_prev ? !(prev == cur) : true;
    std::set<int> bg = background_colors(cur);

    // ACTION6 takes coordinates, so whatever it moved, it did not move it by a
    // displacement belonging to "ACTION6" - it moved it by something that
    // depends on where we clicked. Learning a movement vector for it produced
    // an avatar model for games that have no avatar at all, and with it 1500
    // imaginary walls on a board of 4096 squares.
    bool positional = (last_action >= A1 && last_action <= A5) || last_action == A7;

    if (have_prev && positional && last_action != A_RESET) {
      int scale = detect_scale(cur);
      Motion m = detect_translation(prev, cur, bg, av_color, scale);
      if (!m.found && av_color >= 0) {
        // Many of these games are modal: a click or a press of the interact key
        // changes WHICH thing the direction keys move. Pinning the avatar to
        // one colour for the whole game breaks the moment that happens, so the
        // rule is simply "the avatar is whatever the direction keys are moving
        // now" - if the one we were tracking did not move and something else
        // did, that something else is the avatar.
        m = detect_translation(prev, cur, bg, -1, scale);
        if (m.found && m.color != av_color) {
          // A different sprite answers to the keys: start its map afresh, since
          // what blocks one thing need not block another.
          known.fill(UNKNOWN);
          visited.fill(0);
          plan.clear();
          av_w = av_h = 0;
        }
      }

      if (m.found) {
        if (rule_target >= 0) {
          ColorRule& r = rules[rule_target];
          if (levels_completed > levels) ++r.completed;
          else if (cur.c[expect_y * W + expect_x] != rule_target) ++r.collected;
          else ++r.passed;
        }
        action_moves[last_action][m.disp] += 1;
        ++av_support;
        if (av_support >= 2) avatar_known = true;
        adopt(m);
        known[ay * W + ax] = FREE;
        visited[ay * W + ax] = 1;
        // Standing somewhere counts as having tried it. Without this the agent
        // walks to the same goal-coloured square forever: on LS20 it learned
        // the right colour from level 1 and then spent 3,794 of its 3,881
        // actions walking to squares of that colour on level 2, never
        // searching again - the search counter stayed at zero for the whole
        // game.
        mark_touched(ay * W + ax);
        if (reacquire > 0) reacquire = 0;
      } else if (expect_valid && ax >= 0) {
        if (rule_target >= 0) ++rules[rule_target].blocked;
        // We asked it to move and it stayed put: that square is not walkable,
        // and every step queued behind it was computed on a stale map.
        if (in_bounds(expect_x, expect_y)) {
          int b = expect_y * W + expect_x;
          if (known[b] != BLOCKED) ++blocked_marks;
          known[b] = BLOCKED;
          mark_touched(b);   // whatever is standing there, we have met it
        }
        visited[ay * W + ax] = 1;
        plan.clear();
      }
    }
    expect_valid = false;

    if (levels_completed > levels) {
      // Finishing a level repaints the whole board, so the avatar's move is not
      // detectable on this frame and the rule above never fires. Record it here
      // instead: the square we were stepping onto is what ended the level, and
      // that is the single most valuable thing to know for levels 2..N.
      if (rule_target >= 0) ++rules[rule_target].completed;
      if (idd_active || idd_try_solution || explore_mode == 6) {
        if (!idd_played.empty()) idd_solution = idd_played;
        idd_try_solution = false;
      }
      levels = levels_completed;
      trigger_action = last_action;
      trigger_valid = true;
      rule_target = -1;
      if (last_action == A6 && have_prev) {
        // Describe the thing we clicked, in the frame as it was before the
        // click, so the same kind of thing can be found in the next level.
        trigger_kind = TRIG_CLICK;
        int idx = pending_y * W + pending_x;
        Box b = component_at(prev, idx, 0);
        trig_col = prev.c[idx];
        trig_area = b.area;
        trig_w = b.w();
        trig_h = b.h();
      } else {
        trigger_kind = TRIG_STAND;
        trigger_under = last_under;
      }
      on_new_level();
    }

    if (have_prev && last_action == A6 && click_x >= 0) {
      if (changed) {
        click_live = true;          // still paying off - press it again
      } else {
        click_live = false;
        clicked.insert(click_y * W + click_x);
      }
    }

    {
      Scene ns = scene_of(cur, bg);
      if (have_prev && have_scene && last_action >= A1 && last_action <= A7)
        wm.learn(cur_scene, ns, last_action);
      cur_scene = ns;
      have_scene = true;
      scene_is_new = scene_seen.insert(scene_key(levels_completed)).second;
    }

    if (explore_mode == 6) {
      uint64_t h = cur_scene.hash() ^ (uint64_t(levels_completed) * 0x9E3779B97F4A7C15ULL);
      std::map<uint64_t, std::vector<Act> >::iterator it = ge_route.find(h);
      if (it == ge_route.end())
        ge_route[h] = level_traj;
      else if (level_traj.size() < it->second.size())
        it->second = level_traj;      // a cheaper way back, worth keeping
      ge_here = h;
      ge_ready = true;
    }

    if (explore_mode == 5 && idd_active && bfs_phase == 0 && !idd_played.empty()) {
      // We have just probed one action from the current node. Keep the state it
      // reached only if it is new: everything else is a route we have already
      // got a cheaper way to reach, which is what makes this a search over
      // states rather than over sequences.
      // Key on the objects, not the raw frame. The remaining-actions bar sits
      // in the frame and advances on its own, so a frame hash makes every state
      // look new and the search degenerates into enumerating sequences. The bar
      // is wider than any sprite and so is already excluded from the scene.
      uint64_t h = cur_scene.hash() ^ (uint64_t(levels_completed) * 0x9E3779B97F4A7C15ULL);
      if (bfs_seen.insert(h).second) {
        BfsNode child;
        child.traj = idd_played;
        bfs.push_back(child);
      } else {
        ++idd_pruned;
      }
    }

    if (explore_mode == 11 && have_prev_s32 && prev_action_idx >= 0) {
      Tr t;
      t.s = prev_s32;
      subsample(cur, t.ns);
      t.a = prev_action_idx;
      // Reaching an arrangement never seen before is what we pay for; merely
      // repainting the picture is worth a little; doing nothing is worth zero.
      t.r = scene_is_new ? 1.0f : (changed ? 0.2f : 0.0f);
      if (levels_completed > levels) t.r += 5.0f;
      t.done = (levels_completed > levels) || last_state == 3;
      if (qbuf.size() < 20000) qbuf.push_back(t);
      else { qbuf[qhead] = t; qhead = (qhead + 1) % qbuf.size(); }
      have_prev_s32 = false;
    }

    if ((explore_mode == 9 || explore_mode == 10) && have_prev_s32 && prev_action_idx >= 0) {
      uint64_t h = exp_hash(prev_s32, prev_action_idx);
      if (replay_seen.insert(h).second) {
        Exp e;
        e.s32 = prev_s32;
        e.action_idx = prev_action_idx;
        e.reward = changed ? 1.0f : 0.0f;
        replay.push_back(e);
        if (replay.size() > 20000) {
          replay.erase(replay.begin(), replay.begin() + 5000);
        }
      }
      have_prev_s32 = false;
    }

    if (have_prev && last_action >= 0 && last_action < 8) {
      ++act_tries[last_action];
      if (changed) ++act_change[last_action];
      if (last_action == A6 && last_click_idx >= 0) {
        ++click_tries[last_click_idx];
        if (changed) ++click_change[last_click_idx];
      }
    }

    // A square that has stopped responding has stopped responding IN THIS
    // SITUATION. The compressed offline solutions are full of coordinates that
    // go inert and then matter again once something else has moved - S5I5's
    // level 1 is A once, B three times, A five times, B four times - so counts
    // that are never forgotten make those games unsolvable by construction.
    //
    // But halving EVERY count on every new arrangement was worse than not
    // forgetting at all (6 levels and 0.06 against 8 and 0.07): in a movement
    // game every step of the avatar is a new arrangement, so the evidence was
    // being halved on essentially every action and never accumulated. Only the
    // squares that have actually been written off get their second chance; a
    // square that has ever paid off keeps its record intact.
    // forget_mode: 0 off, 1 only squares that never paid off, 2 everything.
    // Measured on all 25 games at 20,000 actions: off gives 8 levels and 0.07,
    // inert-only gives 4 and 0.01, halve-everything gives 6 and 0.06. Forgetting
    // is simply wrong here, and it is wrong for a reason worth keeping: in a
    // movement game every step of the avatar is a new arrangement, so anything
    // keyed on "a new arrangement appeared" fires on essentially every action.
    // The switch stays so the claim can be re-checked rather than believed.
    if (forget_mode && scene_is_new && (explore_mode == 8 || explore_mode == 10)) {
      for (int i = 0; i < NCELL; ++i) {
        if (forget_mode == 2) { click_tries[i] >>= 1; click_change[i] >>= 1; }
        else if (click_change[i] == 0 && click_tries[i] > 0) click_tries[i] >>= 1;
      }
      clicked.clear();
    }

    if (have_prev) novelty.note_change(prev, cur);
    // Archive this state under the cheapest route we know to it. Replaying a
    // route costs real actions, so cheaper is strictly better.
    if (!replaying && int(level_traj.size()) <= 4 * rollout_len) {
      uint64_t h = novelty.hash(cur, levels_completed);
      std::map<uint64_t, ArcEntry>::iterator it = level_archive.find(h);
      if (it == level_archive.end()) {
        ArcEntry e;
        e.traj = level_traj;
        level_archive[h] = e;
      } else if (level_traj.size() < it->second.traj.size()) {
        it->second.traj = level_traj;
      }
    }
    int before = novelty.visit(cur, levels_completed);
    if (before > 0) {
      ++repeats;
    } else {
      repeats = 0;
      escalation = 0;   // we are getting somewhere again
    }
    // Circling: 40 straight re-entries into states we have already been in.
    if (repeats > 40) {
      repeats = 0;
      if (escalation < 3) ++escalation;
      plan.clear();
      if (escalation >= 1) probed5.fill(0);
      if (escalation >= 2) clicked.clear();
    }

    rule_target = -1;
    if (ax >= 0 && ay >= 0 && have_prev) last_under = prev.at(ax, ay);
    stagnant = changed ? 0 : stagnant + 1;
    have_prev = true;
    ++steps;
  }

  int emit(int a) { return emit(a, pending_x, pending_y); }

  int emit(int a, int x, int y) {
    last_action = a;
    pending_x = x;
    pending_y = y;
    last_click_idx = (a == A6 && in_bounds(x, y)) ? y * W + x : -1;
    if (a == A_RESET) {
      level_traj.clear();       // the level starts over
      forget_map();
    } else {
      level_traj.push_back(Act(a, x, y));
    }
    // Remember where a move action should take us, so the next frame tells us
    // whether that square was walkable.
    int cnt = 0;
    Vec v = best_move(a, &cnt);
    if (cnt > 0 && !v.zero() && ax >= 0 && a != A6) {
      expect_x = ax + v.dx;
      expect_y = ay + v.dy;
      expect_valid = in_bounds(expect_x, expect_y);
      rule_target = expect_valid ? cur.c[expect_y * W + expect_x] : -1;
    }
    return a;
  }

  // Repeat the last action half the time. In a grid world a run of one
  // direction goes somewhere; an independent draw each step mostly jitters on
  // the spot. This is the policy the offline solver uses, and it is what finds
  // level completions there within a few thousand steps.
  int sticky_explore(const std::set<int>& bg) {
    if (click_live && click_x >= 0 && has6) return emit(A6, click_x, click_y);
    if (sticky_a >= 0 && (rng() % 100) < 50)
      return emit(sticky_a, sticky_x, sticky_y);

    std::vector<Act> cands;
    for (size_t i = 0; i < avail.size(); ++i) cands.push_back(Act(avail[i], 0, 0));
    if (has6) {
      std::vector<int> cols;
      std::vector<Box> comps = components(cur, bg, &cols);
      for (size_t i = 0; i < comps.size() && i < 16; ++i)
        cands.push_back(Act(A6, comps[i].cx(), comps[i].cy()));
      for (int i = 0; i < 16; ++i)
        cands.push_back(Act(A6, int(rng() % W), int(rng() % H)));
    }
    if (cands.empty()) return emit(A1, 0, 0);
    Act a = cands[rng() % cands.size()];
    sticky_a = a.a; sticky_x = a.x; sticky_y = a.y;
    if (a.a == A6) { click_x = a.x; click_y = a.y; click_live = false; }
    return emit(a.a, a.x, a.y);
  }

// Where the avatar is in a scene, if we have identified one.
  int avatar_in(const Scene& s) const {
    if (av_color < 0) return -1;
    for (size_t i = 0; i < s.objs.size(); ++i)
      if (s.objs[i].k.color == av_color && s.objs[i].k.w == av_w && s.objs[i].k.h == av_h)
        return int(i);
    return -1;
  }

  // Distance from the avatar to the nearest object of a colour that has ended a
  // level before. Once level 1 has taught us that colour, closing this distance
  // is the whole game.
  double goal_distance(const Scene& s) const {
    int ai = avatar_in(s);
    if (ai < 0) return -1.0;
    double best = -1.0;
    for (size_t i = 0; i < s.objs.size(); ++i) {
      std::map<int, ColorRule>::const_iterator r = rules.find(s.objs[i].k.color);
      if (r == rules.end() || !r->second.is_goal()) continue;
      double d = std::abs(s.objs[i].x - s.objs[ai].x) + std::abs(s.objs[i].y - s.objs[ai].y);
      if (best < 0 || d < best) best = d;
    }
    return best;
  }

  // Beam search inside the learned model. We only ever play its first move and
  // then look again, so the model being approximate costs us a step, not a plan.
  std::vector<uint8_t> blocked_map() const {
    std::vector<uint8_t> b(NCELL, 0);
    for (int i = 0; i < NCELL; ++i) {
      if (known[i] == BLOCKED) { b[i] = 1; continue; }
      std::map<int, ColorRule>::const_iterator r = rules.find(cur.c[i]);
      if (r != rules.end() && r->second.is_wall()) b[i] = 1;
    }
    return b;
  }

  int model_explore(const std::set<int>& bg) {
    if (!wm.usable()) return sticky_explore(bg);
    std::vector<uint8_t> blocked = blocked_map();

    std::vector<int> moves;
    for (size_t i = 0; i < avail.size(); ++i)
      if (avail[i] != A6) moves.push_back(avail[i]);
    if (moves.empty()) return sticky_explore(bg);

    const int DEPTH = 10, WIDTH = 24;
    struct Node {
      Scene s;
      std::vector<int> acts;
      double score;
    };
    std::vector<Node> beam;
    Node root;
    root.s = cur_scene;
    root.score = 0.0;
    beam.push_back(root);

    std::set<uint64_t> local;
    double d0 = goal_distance(cur_scene);

    for (int d = 0; d < DEPTH; ++d) {
      std::vector<Node> next;
      next.reserve(beam.size() * moves.size());
      double decay = 1.0;
      for (int k = 0; k < d; ++k) decay *= 0.92;
      for (size_t b = 0; b < beam.size(); ++b) {
        for (size_t m = 0; m < moves.size(); ++m) {
          Node n;
          n.s = wm.predict(beam[b].s, moves[m], &blocked);
          n.acts = beam[b].acts;
          n.acts.push_back(moves[m]);
          uint64_t h = n.s.hash();
          double bonus = 0.0;
          if (!scene_seen.count(h) && !local.count(h)) bonus += 1.0;
          local.insert(h);
          double dn = goal_distance(n.s);
          if (d0 >= 0 && dn >= 0) bonus += (d0 - dn) * 0.25;   // walking towards it
          n.score = beam[b].score + bonus * decay;
          next.push_back(n);
        }
      }
      if (next.empty()) break;
      std::sort(next.begin(), next.end(),
                [](const Node& a, const Node& b) { return a.score > b.score; });
      if (int(next.size()) > WIDTH) next.resize(WIDTH);
      beam.swap(next);
    }

    if (beam.empty() || beam[0].acts.empty() || beam[0].score <= 0.0)
      return sticky_explore(bg);   // the model sees nothing worth doing
    return emit(beam[0].acts[0], 0, 0);
  }

// The alphabet to enumerate. Movement games give four to six actions; click
  // games would give 4,096, so those are narrowed to the rare, small objects
  // that the solved games say are the ones that matter.
  void build_idd_alphabet() {
    idd_actions.clear();
    for (size_t i = 0; i < avail.size(); ++i)
      if (avail[i] != A6) idd_actions.push_back(Act(avail[i], 0, 0));
    if (has6) {
      std::set<int> bg = background_colors(cur);
      std::vector<int> cols;
      std::vector<Box> comps = components(cur, bg, &cols);
      std::array<int, 32> hist = color_counts();
      std::vector<std::pair<int, int> > order;
      for (size_t i = 0; i < comps.size(); ++i) {
        int c = cols[i] >= 0 && cols[i] < 32 ? cols[i] : 0;
        order.push_back(std::make_pair(hist[c] * 4 + comps[i].area, int(i)));
      }
      std::sort(order.begin(), order.end());
      // Measured against the compressed offline solutions: the winning clicks
      // are object centroids, but they sit at ranks 11, 14, 17 and 24 in this
      // ordering, so a top-10 alphabet simply never offers them. VC33's level 2
      // also needs a click on bare background, which no centroid covers, hence
      // the grid points.
      size_t take = idd_actions.empty() ? size_t(alpha_objects) : size_t(alpha_objects) / 3;
      for (size_t i = 0; i < order.size() && i < take; ++i) {
        const Box& b = comps[order[i].second];
        idd_actions.push_back(Act(A6, b.cx(), b.cy()));
      }
      if (alpha_grid > 0) {
        int step = std::max(1, int(std::sqrt(double(NCELL) / double(alpha_grid))));
        for (int y = step / 2; y < H; y += step)
          for (int x = step / 2; x < W; x += step)
            idd_actions.push_back(Act(A6, x, y));
      }
    }
  }

  void idd_begin() {
    build_idd_alphabet();
    // Actions observed to move something go first, so the shallow layers of the
    // search spend their actions on moves that do anything.
    std::stable_sort(idd_actions.begin(), idd_actions.end(),
                     [this](const Act& x, const Act& y) {
                       int cx = 0, cy = 0;
                       best_move(x.a, &cx);
                       best_move(y.a, &cy);
                       return cx > cy;
                     });
    bfs.clear();
    bfs.push_back(BfsNode());          // the level's starting state
    bfs_head = 0;
    bfs_seen.clear();
    bfs_phase = 0;
    bfs_replay_pos = 0;
    idd_played.clear();
    idd_pos = 0;
    idd_tried = 0;
    idd_pruned = 0;
    ge_here = 0;
    ge_ready = false;
    idd_active = !idd_actions.empty();
    idd_try_solution = !idd_solution.empty();
  }

  // Move on when the current node has had every action tried on it.
  void bfs_advance() {
    while (bfs_head < bfs.size() &&
           bfs[bfs_head].next_action >= int(idd_actions.size()))
      ++bfs_head;
    if (bfs_head >= bfs.size()) idd_active = false;
  }

// Sample an action in proportion to how often it has done something. A spot
  // never tried is optimistic (1/2), so everything gets looked at once before
  // anything is written off, and a spot that keeps working keeps being chosen.
  // Posterior mean, with dead spots pushed down hard. Plain (c+1)/(t+2) leaves
  // a coordinate that has done nothing four times still worth a third of an
  // untried one, and with a few hundred candidates most of the budget goes to
  // squares that have already proved inert.
  double click_weight(int idx, double prior = 1.0) const {
    int c = click_change[idx], t = click_tries[idx];
    if (t == 0) return 0.5 * prior;         // untried: go on what it looks like
    if (c == 0) return 0.5 / double(1 + t * t);
    return double(c + 1) / double(t + 2);
  }

  // What a square looks like before anything has been tried there. Across the
  // 29 level completions the offline solver found, the click that ended a level
  // landed on a colour covering under 2% of the board with an area of 40 cells
  // or fewer, every time. Counting alone starts every candidate at the same
  // 0.5, which throws that away; this is the only place the agent gets to use
  // it before it has evidence of its own.
  double click_prior(int colour, int area) const {
    std::array<int, 32> hist = color_counts();
    double share = (colour >= 0 && colour < 32) ? double(hist[colour]) / double(NCELL) : 1.0;
    double w = 1.0;
    if (share < 0.02) w *= 3.0;
    if (area > 0 && area <= 40) w *= 2.0;
    return w;
  }

  int novelty_sample(const std::set<int>& bg) {
    double total = 0.0;
    std::vector<std::pair<double, Act> > cands;
    for (size_t i = 0; i < avail.size(); ++i) {
      int a = avail[i];
      if (a == A6) continue;
      double w = (act_change[a] + 1.0) / (act_tries[a] + 2.0);
      cands.push_back(std::make_pair(w, Act(a, 0, 0)));
      total += w;
    }
    if (has6) {
      // Offer the objects, and a coarse sweep so bare background is reachable
      // too - VC33's second level is finished by clicking empty board.
      std::vector<int> cols;
      std::vector<Box> comps = components(cur, bg, &cols);
      // Rank before truncating. The extractor returns objects in scan order, so
      // taking the first 48 meant that on a board with more than 48 objects the
      // ones further down were never offered at all - regardless of how rare or
      // how small they were, which is the only evidence we have about which
      // ones matter.
      std::vector<std::pair<double, int> > by_prior;
      for (size_t i = 0; i < comps.size(); ++i) {
        int col = cols[i] >= 0 && cols[i] < 32 ? cols[i] : 0;
        by_prior.push_back(std::make_pair(-click_prior(col, comps[i].area), int(i)));
      }
      std::sort(by_prior.begin(), by_prior.end());
      for (size_t k = 0; k < by_prior.size() && k < 48; ++k) {
        int i = by_prior[k].second;
        int idx = comps[i].cy() * W + comps[i].cx();
        int col = cols[i] >= 0 && cols[i] < 32 ? cols[i] : 0;
        double w = click_weight(idx, click_prior(col, comps[i].area));
        cands.push_back(std::make_pair(w, Act(A6, comps[i].cx(), comps[i].cy())));
        total += w;
      }
      // The coarse sweep is 256 extra candidates against roughly 48 object
      // centroids, so it dilutes the search more than five to one - and every
      // click that ended a level in the offline solutions landed on an object,
      // not on bare board. alpha_grid controls the stride so the trade can be
      // measured; 0 turns the sweep off entirely.
      if (alpha_grid > 0) {
        int step = std::max(2, 64 / std::max(1, alpha_grid));
        for (int y = 1; y < H; y += step)
          for (int x = 1; x < W; x += step) {
            int idx = y * W + x;
            double w = click_weight(idx);
            cands.push_back(std::make_pair(w, Act(A6, x, y)));
            total += w;
          }
      }
    }
    if (cands.empty()) return emit(A1, 0, 0);

    double r = (double(rng() % 1000000) / 1000000.0) * total;
    for (size_t i = 0; i < cands.size(); ++i) {
      r -= cands[i].first;
      if (r <= 0.0) return emit(cands[i].second.a, cands[i].second.x, cands[i].second.y);
    }
    const Act& a = cands.back().second;
    return emit(a.a, a.x, a.y);
  }

uint64_t scene_key(int levels) const {
    return cur_scene.hash() ^ (uint64_t(levels) * 0x9E3779B97F4A7C15ULL);
  }

  static void subsample(const Grid& g, std::array<int8_t, 32 * 32>& out) {
    for (int y = 0; y < 32; ++y)
      for (int x = 0; x < 32; ++x) out[y * 32 + x] = g.c[(y * 2) * W + (x * 2)];
  }

  static uint64_t exp_hash(const std::array<int8_t, 32 * 32>& s, int a) {
    uint64_t h = 1469598103934665603ULL ^ uint64_t(a + 1) * 1099511628211ULL;
    for (size_t i = 0; i < s.size(); ++i) { h ^= uint8_t(s[i]); h *= 1099511628211ULL; }
    return h;
  }

  void net_train() {
    if (int(replay.size()) < train_batch) return;
    const int B = train_batch;
    std::vector<float> xs(size_t(B) * arc3net::IN_C * arc3net::IN_HW * arc3net::IN_HW, 0.0f);
    std::vector<float> mask(size_t(B) * arc3net::N_OUT, 0.0f);
    std::vector<float> ys(B, 0.0f);
    std::array<int8_t, NCELL> full;
    for (int b = 0; b < B; ++b) {
      const Exp& e = replay[rng() % replay.size()];
      // encode() wants a 64x64 frame; the replay keeps the 32x32 subsample, so
      // widen it back out rather than storing sixteen times the memory.
      for (int y = 0; y < 32; ++y)
        for (int x = 0; x < 32; ++x) {
          int8_t v = e.s32[y * 32 + x];
          full[(y * 2) * W + (x * 2)] = v;
          full[(y * 2) * W + (x * 2) + 1] = v;
          full[(y * 2 + 1) * W + (x * 2)] = v;
          full[(y * 2 + 1) * W + (x * 2) + 1] = v;
        }
      arc3net::encode(full.data(),
                      xs.data() + size_t(b) * arc3net::IN_C * arc3net::IN_HW * arc3net::IN_HW);
      mask[size_t(b) * arc3net::N_OUT + e.action_idx] = 1.0f;
      ys[b] = e.reward;
    }
    ag::Tensor x = ag::Tensor::from(xs, {B, arc3net::IN_C, arc3net::IN_HW, arc3net::IN_HW}, false);
    ag::Tensor logits = net.forward(x);
    ag::Tensor m = ag::Tensor::from(mask, {B, arc3net::N_OUT}, false);
    ag::Tensor ones = ag::Tensor::from(std::vector<float>(arc3net::N_OUT, 1.0f),
                                       {arc3net::N_OUT, 1}, false);
    ag::Tensor picked = ag::matmul(ag::mul(logits, m), ones);
    ag::Tensor loss = arc3net::bce_with_logits(picked, ys);
    net.opt.zero_grad();
    loss.backward();
    net.opt.step();
  }

  // Ask the network which action is most likely to change something, and sample
  // from it. Actions the game does not offer are removed first.
  int net_choose() {
    std::vector<float> one(arc3net::IN_C * arc3net::IN_HW * arc3net::IN_HW);
    arc3net::encode(cur.c.data(), one.data());
    ag::Tensor x = ag::Tensor::from(one, {1, arc3net::IN_C, arc3net::IN_HW, arc3net::IN_HW}, false);
    ag::Tensor logits = net.forward(x);

    std::vector<double> w(arc3net::N_OUT, 0.0);
    double total = 0.0;
    for (int i = 0; i < arc3net::N_SIMPLE; ++i) {
      int act = A1 + i;
      if (std::find(avail.begin(), avail.end(), act) == avail.end()) continue;
      w[i] = 1.0 / (1.0 + std::exp(-double(logits.data()[i])));
      total += w[i];
    }
    if (has6) {
      for (int i = 0; i < arc3net::N_COORD; ++i) {
        int k = arc3net::N_SIMPLE + i;
        w[k] = 1.0 / (1.0 + std::exp(-double(logits.data()[k])));
        total += w[k];
      }
    }
    if (total <= 0.0) return emit(avail.empty() ? A1 : avail[rng() % avail.size()], 0, 0);

    double r = (double(rng() % 1000000) / 1000000.0) * total;
    int pick = -1;
    for (int i = 0; i < arc3net::N_OUT; ++i) {
      if (w[i] <= 0.0) continue;
      r -= w[i];
      if (r <= 0.0) { pick = i; break; }
    }
    if (pick < 0) pick = arc3net::N_SIMPLE - 1;

    subsample(cur, prev_s32);
    have_prev_s32 = true;
    prev_action_idx = pick;

    if (pick < arc3net::N_SIMPLE) return emit(A1 + pick, 0, 0);
    int cell = pick - arc3net::N_SIMPLE;
    int cx = (cell % arc3net::COORD_HW) * 4 + 2;
    int cy = (cell / arc3net::COORD_HW) * 4 + 2;
    return emit(A6, cx, cy);
  }

// Mode 10: the counting policy and the network, multiplied.
  //
  // At 20,000 actions they reach the same number of levels but not the same
  // ones - counting gets FT09, VC33 and AR25, the network gets LP85 to level 5,
  // which is as deep as the offline solver ever managed there. They are good at
  // different things: counting is useful from the first action and then
  // saturates, the network knows nothing at first and keeps improving.
  //
  // Multiplying the two needs no schedule. An untrained network outputs about
  // 0.5 everywhere, which is a flat factor and leaves the counts in charge;
  // as it learns it sharpens and takes over wherever it has an opinion.
  int blended_choose(const std::set<int>& bg) {
    std::vector<float> one(arc3net::IN_C * arc3net::IN_HW * arc3net::IN_HW);
    arc3net::encode(cur.c.data(), one.data());
    ag::Tensor x = ag::Tensor::from(one, {1, arc3net::IN_C, arc3net::IN_HW, arc3net::IN_HW}, false);
    ag::Tensor logits = net.forward(x);

    std::vector<std::pair<double, Act> > cands;
    double total = 0.0;

    for (size_t i = 0; i < avail.size(); ++i) {
      int a = avail[i];
      if (a == A6 || a < A1 || a > A5) continue;
      double table = (act_change[a] + 1.0) / (act_tries[a] + 2.0);
      double netp = 1.0 / (1.0 + std::exp(-double(logits.data()[a - A1])));
      double w = table * netp;
      cands.push_back(std::make_pair(w, Act(a, 0, 0)));
      total += w;
    }

    if (has6) {
      // The network scores one cell per logical square; the table scores exact
      // pixels. Pair each candidate pixel with the cell it falls in.
      std::vector<int> cols;
      std::vector<Box> comps = components(cur, bg, &cols);
      for (size_t i = 0; i < comps.size() && i < 48; ++i) {
        int px = comps[i].cx(), py = comps[i].cy();
        int cell = (py / 4) * arc3net::COORD_HW + (px / 4);
        double netp = 1.0 / (1.0 + std::exp(
            -double(logits.data()[arc3net::N_SIMPLE + cell])));
        int col = cols[i] >= 0 && cols[i] < 32 ? cols[i] : 0;
        double w = click_weight(py * W + px, click_prior(col, comps[i].area)) * netp;
        cands.push_back(std::make_pair(w, Act(A6, px, py)));
        total += w;
      }
      for (int y = 1; y < H; y += 4)
        for (int x = 1; x < W; x += 4) {
          int cell = (y / 4) * arc3net::COORD_HW + (x / 4);
          double netp = 1.0 / (1.0 + std::exp(
              -double(logits.data()[arc3net::N_SIMPLE + cell])));
          double w = click_weight(y * W + x) * netp;
          cands.push_back(std::make_pair(w, Act(A6, x, y)));
          total += w;
        }
    }
    if (cands.empty() || total <= 0.0)
      return emit(avail.empty() ? A1 : avail[rng() % avail.size()], 0, 0);

    double r = (double(rng() % 1000000) / 1000000.0) * total;
    Act chosen = cands.back().second;
    for (size_t i = 0; i < cands.size(); ++i) {
      r -= cands[i].first;
      if (r <= 0.0) { chosen = cands[i].second; break; }
    }

    // Remember what was asked of which state, so the network can be told
    // afterwards whether it was right.
    subsample(cur, prev_s32);
    have_prev_s32 = true;
    prev_action_idx = (chosen.a == A6)
        ? arc3net::N_SIMPLE + (chosen.y / 4) * arc3net::COORD_HW + (chosen.x / 4)
        : (chosen.a - A1);
    if (chosen.a == A6) { click_x = chosen.x; click_y = chosen.y; click_live = false; }
    return emit(chosen.a, chosen.x, chosen.y);
  }

  static void widen(const std::array<int8_t, 32 * 32>& s32, std::array<int8_t, NCELL>& full) {
    for (int y = 0; y < 32; ++y)
      for (int x = 0; x < 32; ++x) {
        int8_t v = s32[y * 32 + x];
        full[(y * 2) * W + (x * 2)] = v;
        full[(y * 2) * W + (x * 2) + 1] = v;
        full[(y * 2 + 1) * W + (x * 2)] = v;
        full[(y * 2 + 1) * W + (x * 2) + 1] = v;
      }
  }

  std::vector<float> q_values(const std::array<int8_t, 32 * 32>& s32,
                              arc3net::ActionNet& which) {
    std::array<int8_t, NCELL> full;
    widen(s32, full);
    std::vector<float> one(arc3net::IN_C * arc3net::IN_HW * arc3net::IN_HW);
    arc3net::encode(full.data(), one.data());
    ag::Tensor x = ag::Tensor::from(one, {1, arc3net::IN_C, arc3net::IN_HW, arc3net::IN_HW}, false);
    ag::Tensor q = which.forward(x);
    return q.data();
  }

  bool q_legal(int idx) const {
    if (idx < arc3net::N_SIMPLE)
      return std::find(avail.begin(), avail.end(), A1 + idx) != avail.end();
    return has6;
  }

  void q_train() {
    const int B = 8;   // three forward passes per step, so a smaller batch
    if (int(qbuf.size()) < B * 2) return;

    std::vector<int> pick(B);
    for (int i = 0; i < B; ++i) pick[i] = int(rng() % qbuf.size());

    std::array<int8_t, NCELL> full;
    std::vector<float> xs_next(size_t(B) * arc3net::IN_C * arc3net::IN_HW * arc3net::IN_HW, 0.0f);
    std::vector<float> xs_cur(size_t(B) * arc3net::IN_C * arc3net::IN_HW * arc3net::IN_HW, 0.0f);
    for (int b = 0; b < B; ++b) {
      widen(qbuf[pick[b]].ns, full);
      arc3net::encode(full.data(),
                      xs_next.data() + size_t(b) * arc3net::IN_C * arc3net::IN_HW * arc3net::IN_HW);
      widen(qbuf[pick[b]].s, full);
      arc3net::encode(full.data(),
                      xs_cur.data() + size_t(b) * arc3net::IN_C * arc3net::IN_HW * arc3net::IN_HW);
    }

    // Double DQN: the online net says which action is best next, the target net
    // says what it is worth. One net doing both overestimates.
    std::vector<float> target(B, 0.0f);
    {
      ag::Tensor nx = ag::Tensor::from(xs_next,
          {B, arc3net::IN_C, arc3net::IN_HW, arc3net::IN_HW}, false);
      ag::Tensor q_online = net.forward(nx);
      ag::Tensor q_target = qtgt.forward(nx);
      for (int b = 0; b < B; ++b) {
        float best = 0.0f;
        int besti = -1;
        for (int i = 0; i < arc3net::N_OUT; ++i) {
          if (!q_legal(i)) continue;
          float v = q_online.data()[size_t(b) * arc3net::N_OUT + i];
          if (besti < 0 || v > best) { best = v; besti = i; }
        }
        float boot = (besti < 0 || qbuf[pick[b]].done)
                         ? 0.0f
                         : q_target.data()[size_t(b) * arc3net::N_OUT + besti];
        target[b] = qbuf[pick[b]].r + q_gamma * boot;
      }
    }

    ag::Tensor x = ag::Tensor::from(xs_cur,
        {B, arc3net::IN_C, arc3net::IN_HW, arc3net::IN_HW}, false);
    ag::Tensor q = net.forward(x);
    std::vector<float> mask(size_t(B) * arc3net::N_OUT, 0.0f);
    for (int b = 0; b < B; ++b) mask[size_t(b) * arc3net::N_OUT + qbuf[pick[b]].a] = 1.0f;
    ag::Tensor m = ag::Tensor::from(mask, {B, arc3net::N_OUT}, false);
    ag::Tensor ones = ag::Tensor::from(std::vector<float>(arc3net::N_OUT, 1.0f),
                                       {arc3net::N_OUT, 1}, false);
    ag::Tensor picked = ag::matmul(ag::mul(q, m), ones);        // (B,1)
    ag::Tensor tgt = ag::Tensor::from(target, {B, 1}, false);
    ag::Tensor diff = ag::sub(picked, tgt);
    ag::Tensor loss = ag::mean(ag::mul(diff, diff));

    net.opt.zero_grad();
    loss.backward();
    net.opt.step();

    if (net.opt.t % q_sync_every == 0) qtgt.copy_from(net);
  }

  // Softmax over the legal Q values. Greedy would stop exploring, and here
  // exploration IS the reward.
  int q_choose() {
    std::array<int8_t, 32 * 32> s32;
    subsample(cur, s32);
    std::vector<float> q = q_values(s32, net);

    float best = 0.0f;
    bool any = false;
    for (int i = 0; i < arc3net::N_OUT; ++i)
      if (q_legal(i) && (!any || q[i] > best)) { best = q[i]; any = true; }
    if (!any) return emit(avail.empty() ? A1 : avail[rng() % avail.size()], 0, 0);

    const float temp = 0.5f;
    double total = 0.0;
    std::vector<double> w(arc3net::N_OUT, 0.0);
    for (int i = 0; i < arc3net::N_OUT; ++i) {
      if (!q_legal(i)) continue;
      w[i] = std::exp(double(q[i] - best) / temp);
      total += w[i];
    }
    double r = (double(rng() % 1000000) / 1000000.0) * total;
    int pick = -1;
    for (int i = 0; i < arc3net::N_OUT; ++i) {
      if (w[i] <= 0.0) continue;
      r -= w[i];
      if (r <= 0.0) { pick = i; break; }
    }
    if (pick < 0) pick = 0;

    prev_s32 = s32;
    have_prev_s32 = true;
    prev_action_idx = pick;

    if (pick < arc3net::N_SIMPLE) return emit(A1 + pick, 0, 0);
    int cell = pick - arc3net::N_SIMPLE;
    int cx = (cell % arc3net::COORD_HW) * 4 + 2;
    int cy = (cell / arc3net::COORD_HW) * 4 + 2;
    return emit(A6, cx, cy);
  }

  int choose() {
    std::set<int> bg = background_colors(cur);

    if (explore_mode == 11) {
      if (last_state == 3) return emit(A_RESET, 0, 0);
      if (!calib_queue.empty()) {
        int a = calib_queue.front();
        calib_queue.erase(calib_queue.begin());
        return emit(a);
      }
      if (!plan.empty()) {
        int a = plan.front();
        plan.erase(plan.begin());
        return emit(a);
      }
      if (avatar_known && ax >= 0) {
        std::vector<int> p = plan_to(goal_targets());
        if (!p.empty()) {
          plan = p;
          int a = plan.front();
          plan.erase(plan.begin());
          return emit(a);
        }
      }
      if (steps % 30 == 0) q_train();
      return q_choose();
    }

    if (explore_mode == 10) {
      if (last_state == 3) return emit(A_RESET, 0, 0);
      if (!calib_queue.empty()) {
        int a = calib_queue.front();
        calib_queue.erase(calib_queue.begin());
        return emit(a);
      }
      if (has6 && click_live && click_x >= 0) return emit(A6, click_x, click_y);
      if (!plan.empty()) {
        int a = plan.front();
        plan.erase(plan.begin());
        return emit(a);
      }
      if (avatar_known && ax >= 0) {
        std::vector<int> p = plan_to(goal_targets());
        if (!p.empty()) {
          plan = p;
          int a = plan.front();
          plan.erase(plan.begin());
          return emit(a);
        }
      }
      if (steps % train_every == 0) net_train();
      return blended_choose(bg);
    }

    if (explore_mode == 9) {
      if (last_state == 3) return emit(A_RESET, 0, 0);
      if (!calib_queue.empty()) {
        int a = calib_queue.front();
        calib_queue.erase(calib_queue.begin());
        return emit(a);
      }
      if (!plan.empty()) {
        int a = plan.front();
        plan.erase(plan.begin());
        return emit(a);
      }
      // Whatever we learned about where a level ends still beats guessing.
      if (avatar_known && ax >= 0) {
        std::vector<int> p = plan_to(goal_targets());
        if (!p.empty()) {
          plan = p;
          int a = plan.front();
          plan.erase(plan.begin());
          return emit(a);
        }
      }
      if (steps % train_every == 0) net_train();
      return net_choose();
    }

    if (explore_mode == 8) {
      // A spot that just did something is the best candidate for doing
      // something again - S5I5's level 1 is two coordinates pressed thirteen
      // times between them. The counting policy alone will drift off it.
      if (has6 && click_live && click_x >= 0) return emit(A6, click_x, click_y);

      // If there is something we control, WALK. Sampling a direction from a
      // distribution is a terrible way to cross a room: counting alone takes
      // 12,067 actions to finish level 1 of LS20 against a human baseline of
      // 22, where routing to rare-coloured objects does it in about 87. Only
      // completions inside roughly ten times the baseline score at all, so this
      // is the difference between a level that counts and one that does not.
      if (!plan.empty()) {
        int a = plan.front();
        plan.erase(plan.begin());
        return emit(a);
      }
      if (avatar_known && ax >= 0) {
        std::vector<int> p = plan_to(goal_targets());
        if (p.empty()) p = plan_to(pickup_targets());
        if (p.empty()) p = plan_to_rare();
        if (p.empty()) p = plan_to(frontier_targets());
        if (!p.empty()) {
          plan = p;
          int a = plan.front();
          plan.erase(plan.begin());
          return emit(a);
        }
        // Standing on something new with an interact key available.
        if (std::find(avail.begin(), avail.end(), A5) != avail.end() &&
            !probed5[ay * W + ax]) {
          probed5[ay * W + ax] = 1;
          return emit(A5);
        }
      }
      if (last_state == 3) return emit(A_RESET, 0, 0);
      if (!calib_queue.empty()) {
        int a = calib_queue.front();
        calib_queue.erase(calib_queue.begin());
        return emit(a);
      }
      // Everything learned about where a level ends still applies.
      if (!plan.empty()) {
        int a = plan.front();
        plan.erase(plan.begin());
        return emit(a);
      }
      if (avatar_known && ax >= 0) {
        std::vector<int> p = plan_to(goal_targets());
        if (!p.empty()) {
          plan = p;
          int a = plan.front();
          plan.erase(plan.begin());
          return emit(a);
        }
      }
      // Levels of one game share their mechanics, so whatever click finished
      // the last level is the best guess for this one. Until now mode 8
      // returned before ever reaching this, so the agent relearned every level
      // from nothing - and the second levels show it: LP85 6x then 219x, CD82
      // 11x then 45x.
      // ...but only for the opening moves of the level. Left unbounded this
      // branch returns ahead of the counting policy for the whole level and
      // monopolises it, clicking one lookalike after another: 16 levels and
      // 0.38 became 13 and 0.35, and it specifically cost the SECOND levels of
      // R11L, LP85 and VC33. Try what worked, then get out of the way.
      // trigger_replay: off by default, because it does not work. Measured on
      // all 25 games at 20,000 actions, replaying the click that finished the
      // previous level gives 13 levels and 0.35 against 16 and 0.38 without it,
      // whether it runs for the whole level or only its opening moves. It
      // specifically costs the SECOND levels of R11L, LP85 and VC33 - the very
      // thing it was meant to buy - because matching by colour and size picks
      // one lookalike after another and crowds out the search that was finding
      // them. The switch stays so the claim can be re-checked.
      if (trigger_replay && trigger_valid && trigger_kind == TRIG_CLICK && has6 &&
          int(level_traj.size()) < 12) {
        std::vector<int> cols;
        std::vector<Box> comps = components(cur, bg, &cols);
        int best = -1;
        double best_d = 1e18;
        for (size_t i = 0; i < comps.size(); ++i) {
          int key = comps[i].cy() * W + comps[i].cx();
          if (clicked.count(key)) continue;
          double d = (cols[i] == trig_col ? 0.0 : 8.0)
                   + std::abs(comps[i].area - trig_area) * 2.0
                   + std::abs(comps[i].w() - trig_w) * 1.0
                   + std::abs(comps[i].h() - trig_h) * 1.0;
          if (d < best_d) { best_d = d; best = int(i); }
        }
        if (best >= 0) {
          const Box& b = comps[best];
          clicked.insert(b.cy() * W + b.cx());
          click_x = b.cx();
          click_y = b.cy();
          click_live = false;
          return emit(A6, click_x, click_y);
        }
      }

      return novelty_sample(bg);
    }

    // Mode 7: spend the budget on several policies in turn.
    //
    // They do not fail on the same games - the heuristic gets CN04, LS20, SP80
    // and LF52, sticky-random gets LP85, SP80 and VC33 - and a level left
    // unfinished scores zero however many actions went into it, so there is
    // nothing to protect by staying with one. The games that do fall, fall
    // early (LS20's level 1 costs 87 actions of about 3,900), so splitting the
    // budget costs little.
    if (explore_mode == 7) {
      int spent = steps;
      int m = (spent * 3 < budget) ? 0 : (spent * 3 < 2 * budget ? 1 : 6);
      if (m != phase_mode) {
        phase_mode = m;
        plan.clear();
        replaying = false;
        replay_queue.clear();
        clicked.clear();
        probed5.fill(0);
        touched.fill(0);
        if (m == 6) idd_begin();
      }
      int saved = explore_mode;
      explore_mode = m;
      int a = choose();
      explore_mode = saved;
      return a;
    }

    // Dead: RESET is the only legal action. Use it to resume from a state worth
    // revisiting rather than from the start of the level.
    if (last_state == 3) {
      restart_from(restart_target());
      return emit(A_RESET, 0, 0);
    }

    // Retracing a route we already know.
    if (replaying) {
      if (!replay_queue.empty()) {
        Act a = replay_queue.front();
        replay_queue.erase(replay_queue.begin());
        if (replay_queue.empty()) replaying = false;
        return emit(a.a, a.x, a.y);
      }
      replaying = false;
    }

    // 1. Calibration: find out what each action does.
    if (!calib_queue.empty()) {
      int a = calib_queue.front();
      calib_queue.erase(calib_queue.begin());
      return emit(a);
    }

    // 2. Lost the avatar (new level): probe until we see something move.
    if (reacquire > 0 && ax < 0) {
      --reacquire;
      int a = avail.empty() ? A1 : avail[reacquire % avail.size()];
      return emit(a);
    }

    if (explore_mode == 1 && calib_queue.empty() && plan.empty()) {
      // Pure stochastic mode still uses everything we have learned about where
      // a level ends; it only replaces the wandering.
      if (avatar_known && ax >= 0) {
        std::vector<int> p = plan_to(goal_targets());
        if (!p.empty()) {
          plan = p;
          int a = plan.front();
          plan.erase(plan.begin());
          return emit(a);
        }
      }
      return sticky_explore(bg);
    }

    // The alphabet is built when a level starts, and the first level of a game
    // starts before any level transition has happened - so without this the
    // search never ran at all, and every sweep of its parameters came back
    // identical because none of that code was being reached.
    if ((explore_mode == 5 || explore_mode == 6) && idd_actions.empty() &&
        calib_queue.empty() && have_scene)
      idd_begin();

    if (explore_mode == 6 && ge_ready && !idd_actions.empty()) {
      if (replaying) {
        if (!replay_queue.empty()) {
          Act a = replay_queue.front();
          replay_queue.erase(replay_queue.begin());
          if (replay_queue.empty()) replaying = false;
          return emit(a.a, a.x, a.y);
        }
        replaying = false;
      }

      // The levels of a game share their mechanics, so before searching this
      // level from scratch, replay whatever finished the last one.
      if (idd_try_solution && last_state != 3) {
        if (idd_pos < int(idd_solution.size())) {
          Act a = idd_solution[idd_pos++];
          idd_played.push_back(a);
          return emit(a.a, a.x, a.y);
        }
        idd_try_solution = false;
        idd_pos = 0;
      }

      // Once level 1 has told us which colour ends a level, going there beats
      // searching - and the later levels, where this pays, are the ones the
      // per-game score weights most heavily.
      if (last_state != 3 && avatar_known && ax >= 0) {
        std::vector<int> p2 = plan_to(goal_targets());
        if (!p2.empty()) {
          plan = p2;
          int a = plan.front();
          plan.erase(plan.begin());
          return emit(a);
        }
      }

      // Dead means only RESET is legal; take it and go back somewhere useful.
      // So does wandering too far: the compressed offline solutions run from
      // two to seventeen actions, so depth beyond that is almost certainly not
      // where the answer is, and a depth-first walk that keeps going never
      // returns to try the rest of the alphabet near the start.
      if (last_state != 3 && int(level_traj.size()) < depth_cap) {
        int& t = ge_tried[ge_here];
        if (t < int(idd_actions.size())) {
          Act a = idd_actions[t++];
          ++idd_tried;
          idd_played.push_back(a);
          return emit(a.a, a.x, a.y);      // one action per probe
        }
      }

      // Nothing left to try here: return to the cheapest state that still has
      // something untried.
      const std::vector<Act>* best = 0;
      size_t best_len = 0;
      for (std::map<uint64_t, std::vector<Act> >::iterator it = ge_route.begin();
           it != ge_route.end(); ++it) {
        std::map<uint64_t, int>::iterator k = ge_tried.find(it->first);
        if (k != ge_tried.end() && k->second >= int(idd_actions.size())) continue;
        if (!best || it->second.size() < best_len) { best = &it->second; best_len = it->second.size(); }
      }
      if (best) {
        replay_queue = *best;
        replaying = !replay_queue.empty();
        idd_played.clear();
        ++restarts;
        return emit(A_RESET, 0, 0);
      }
      // Everything reachable has been tried: let the heuristic policy take over.
    }

    if (explore_mode == 5) {
      if (!idd_active && !idd_try_solution && idd_actions.empty()) idd_begin();

      // Levels of one game share their mechanics, so the sequence that ended
      // the last level is the first thing worth trying on this one.
      if (idd_try_solution) {
        if (idd_pos < int(idd_solution.size())) {
          Act a = idd_solution[idd_pos++];
          idd_played.push_back(a);
          return emit(a.a, a.x, a.y);
        }
        idd_try_solution = false;
        idd_pos = 0;
        idd_played.clear();
        return emit(A_RESET, 0, 0);
      }

      if (idd_active) {
        bfs_advance();
        if (!idd_active) {
          // nothing left to expand
        } else if (last_state == 3) {
          bfs_phase = 0;                       // died; start the next probe
        }
        if (idd_active) {
          BfsNode& n = bfs[bfs_head];
          if (bfs_phase == 0) {                // go back to the level start
            bfs_phase = 1;
            bfs_replay_pos = 0;
            idd_played.clear();
            return emit(A_RESET, 0, 0);
          }
          if (bfs_phase == 1) {                // walk to this node
            if (bfs_replay_pos < int(n.traj.size())) {
              Act a = n.traj[bfs_replay_pos++];
              idd_played.push_back(a);
              return emit(a.a, a.x, a.y);
            }
            bfs_phase = 2;
          }
          // Try one untried action from here; the next frame tells us whether
          // it reached somewhere new.
          Act a = idd_actions[n.next_action++];
          idd_played.push_back(a);
          ++idd_tried;
          bfs_phase = 0;
          return emit(a.a, a.x, a.y);
        }
      }
      // Search exhausted: fall through to the heuristic policy.
    }

    if (explore_mode == 4 && calib_queue.empty() && plan.empty() && !replaying) {
      if (avatar_known && ax >= 0) {
        std::vector<int> p = plan_to(goal_targets());
        if (!p.empty()) {
          plan = p;
          int a = plan.front();
          plan.erase(plan.begin());
          return emit(a);
        }
      }
      if (has6 && click_live && click_x >= 0) return emit(A6, click_x, click_y);
      return model_explore(bg);
    }

    // Systematic Go-Explore: a rollout ends, we go back somewhere promising.
    if (explore_mode == 3 && !replaying && ++since_restart >= rollout_len) {
      since_restart = 0;
      const ArcEntry* t = restart_target();
      if (t) {
        restart_from(t);
        return emit(A_RESET, 0, 0);
      }
    }
    if (explore_mode == 3 && calib_queue.empty() && plan.empty()) {
      if (avatar_known && ax >= 0) {
        std::vector<int> p = plan_to(goal_targets());
        if (!p.empty()) {
          plan = p;
          int a = plan.front();
          plan.erase(plan.begin());
          return emit(a);
        }
      }
      return sticky_explore(bg);
    }

    // 3. Follow the current plan.
    if (!plan.empty()) {
      int a = plan.front();
      plan.erase(plan.begin());
      return emit(a);
    }

    // Whatever we last clicked is still changing the world: press it again.
    // The compressed offline solutions for the click games are largely runs of
    // one coordinate - S5I5's level 1 is two points pressed thirteen times
    // between them - and this has to outrank replaying the previous level's
    // winning click, which otherwise moves on after a single press and leaves
    // the agent clicking dead spots (LP85 sat for 27 straight actions without
    // the frame changing at all).
    if (has6 && click_live && click_x >= 0) return emit(A6, click_x, click_y);

    // 3b. Circling with nothing left to try: restart this level. A mid-game
    //     RESET restarts only the current level and keeps the levels we have
    //     already finished (arcengine base_game.handle_reset), so it is a retry
    //     rather than a surrender.
    if (escalation >= 3) {
      restart_from(restart_target());
      return emit(A_RESET, 0, 0);
    }

    // 4. We know what ends a level here: walk to it. This is the whole point
    //    of spending level 1 on exploration.
    if (avatar_known && ax >= 0) {
      std::vector<int> p = plan_to(goal_targets());
      if (p.empty()) p = plan_to(pickup_targets());
      if (p.empty()) p = plan_to_rare();
      if (!p.empty()) {
        plan = p;
        int a = plan.front();
        plan.erase(plan.begin());
        return emit(a);
      }
    }

    // 4a. The previous level ended on a click: find the thing that looks most
    //     like what we clicked then, and click that.
    if (trigger_valid && trigger_kind == TRIG_CLICK && has6) {
      std::vector<int> cols;
      std::vector<Box> comps = components(cur, bg, &cols);
      int best = -1;
      double best_d = 1e18;
      for (size_t i = 0; i < comps.size(); ++i) {
        int key = comps[i].cy() * W + comps[i].cx();
        if (clicked.count(key)) continue;
        double d = (cols[i] == trig_col ? 0.0 : 8.0)
                 + std::abs(comps[i].area - trig_area) * 2.0
                 + std::abs(comps[i].w() - trig_w) * 1.0
                 + std::abs(comps[i].h() - trig_h) * 1.0;
        if (d < best_d) { best_d = d; best = int(i); }
      }
      if (best >= 0) {
        const Box& b = comps[best];
        clicked.insert(b.cy() * W + b.cx());
        return emit(A6, b.cx(), b.cy());
      }
    }

    // 4b. The previous level ended by moving onto something: walk to the
    //     nearest square of that colour and repeat the winning action.
    if (trigger_valid && trigger_kind == TRIG_STAND && avatar_known && trigger_under >= 0) {
      std::vector<int> t;
      for (int i = 0; i < NCELL; ++i)
        if (cur.c[i] == trigger_under && !visited[i]) t.push_back(i);
      if (!t.empty()) {
        std::vector<int> p = plan_to(t);
        if (!p.empty()) {
          if (trigger_action >= 0 && !is_move_action(trigger_action))
            p.push_back(trigger_action);
          plan = p;
          int a = plan.front();
          plan.erase(plan.begin());
          return emit(a);
        }
      }
    }

    // 5. Directed exploration. Objects first - walking onto or up against a
    //    thing is what changes the world; empty floor almost never is.
    if (avatar_known && ax >= 0) {
      // Standing on something new? Interact before walking on - three of the
      // level completions in the solved games came from ACTION5 while standing
      // on a rare colour.
      if (std::find(avail.begin(), avail.end(), A5) != avail.end() &&
          !probed5[ay * W + ax]) {
        probed5[ay * W + ax] = 1;
        return emit(A5);
      }
      mark_touched(ay * W + ax);

      std::vector<int> p = plan_to(object_targets(bg));
      if (p.empty()) p = plan_to(frontier_targets());
      if (!p.empty()) {
        plan = p;
        int a = plan.front();
        plan.erase(plan.begin());
        return emit(a);
      }
      if (explore_mode == 2) return sticky_explore(bg);
      // Fully explored and nothing to interact with: forget the map so the
      // frontier refills, in case the world changed under us.
      visited.fill(0);
      known.fill(UNKNOWN);
    }

    // 6. Click-style game.
    if (has6) {
      // Still getting a response out of the last spot? Keep pressing it.
      if (click_live && click_x >= 0) return emit(A6, click_x, click_y);

      std::vector<int> cols;
      std::vector<Box> comps = components(cur, bg, &cols);
      std::array<int, 32> hist = color_counts();
      std::vector<std::pair<int, int> > order;  // (cost, index)
      for (size_t i = 0; i < comps.size(); ++i) {
        int col = cols[i] >= 0 && cols[i] < 32 ? cols[i] : 0;
        // Rare colour first, small shape next. Sorting by largest area was
        // backwards: across every solved public game the object whose click
        // ended a level covered under 2% of the board and had an area of 40
        // cells or fewer.
        order.push_back(std::make_pair(hist[col] * 4 + comps[i].area, int(i)));
      }
      std::sort(order.begin(), order.end());
      for (size_t i = 0; i < order.size(); ++i) {
        const Box& b = comps[order[i].second];
        int key = b.cy() * W + b.cx();
        if (clicked.count(key)) continue;
        click_x = b.cx();
        click_y = b.cy();
        click_live = false;
        return emit(A6, click_x, click_y);
      }
      for (int y = 2; y < H; y += 4)
        for (int x = 2; x < W; x += 4) {
          int key = y * W + x;
          if (clicked.count(key)) continue;
          click_x = x;
          click_y = y;
          click_live = false;
          return emit(A6, x, y);
        }
      clicked.clear();
    }

    // 7. Fallback.
    int a = A1;
    if (!avail.empty()) {
      std::uniform_int_distribution<int> d(0, int(avail.size()) - 1);
      a = avail[d(rng)];
    }
    return emit(a);
  }
};

std::vector<Agent*> g_agents;

}  // namespace

ARC3_API int arc3_new(const int* actions, int n) {
  Agent* a = new Agent();
  a->init(actions, n);
  g_agents.push_back(a);
  return int(g_agents.size()) - 1;
}

// Choose the exploration policy: 0 directed, 1 sticky-random, 2 directed then
// sticky-random. Exposed so the two can be measured against each other rather
// than argued about.
ARC3_API void arc3_set_explore(int h, int mode) {
  if (h < 0 || h >= int(g_agents.size()) || !g_agents[h]) return;
  g_agents[h]->explore_mode = mode;
}

// How many click targets the search may consider. Bigger alphabets cover more
// of the board and cost more probes per state.
ARC3_API void arc3_set_alphabet(int h, int objects, int grid) {
  if (h < 0 || h >= int(g_agents.size()) || !g_agents[h]) return;
  g_agents[h]->alpha_objects = objects;
  g_agents[h]->alpha_grid = grid;
}

ARC3_API void arc3_set_budget(int h, int n) {
  if (h < 0 || h >= int(g_agents.size()) || !g_agents[h]) return;
  g_agents[h]->budget = n > 0 ? n : 4000;
}

ARC3_API void arc3_set_clear_stats(int h, int on) {
  if (h < 0 || h >= int(g_agents.size()) || !g_agents[h]) return;
  g_agents[h]->clear_stats_on_level = on;
}

ARC3_API void arc3_set_trigger(int h, int on) {
  if (h < 0 || h >= int(g_agents.size()) || !g_agents[h]) return;
  g_agents[h]->trigger_replay = on;
}

ARC3_API void arc3_set_forget(int h, int mode) {
  if (h < 0 || h >= int(g_agents.size()) || !g_agents[h]) return;
  g_agents[h]->forget_mode = mode;
}

ARC3_API void arc3_set_depth(int h, int cap) {
  if (h < 0 || h >= int(g_agents.size()) || !g_agents[h]) return;
  g_agents[h]->depth_cap = cap;
}

ARC3_API void arc3_free(int h) {
  if (h >= 0 && h < int(g_agents.size()) && g_agents[h]) {
    delete g_agents[h];
    g_agents[h] = 0;
  }
}

ARC3_API void arc3_observe(int h, const int8_t* frame, int levels_completed, int state) {
  if (h < 0 || h >= int(g_agents.size()) || !g_agents[h]) return;
  g_agents[h]->observe(frame, levels_completed, state);
}

ARC3_API int arc3_choose(int h, int* out_x, int* out_y) {
  if (h < 0 || h >= int(g_agents.size()) || !g_agents[h]) return A1;
  Agent* a = g_agents[h];
  int act = a->choose();
  if (out_x) *out_x = a->pending_x;
  if (out_y) *out_y = a->pending_y;
  return act;
}

ARC3_API void arc3_stats(int h, int* out) {
  if (h < 0 || h >= int(g_agents.size()) || !g_agents[h]) return;
  Agent* a = g_agents[h];
  out[0] = a->avatar_known ? 1 : 0;
  out[1] = a->av_color;
  out[2] = a->ax;
  out[3] = a->ay;
  out[4] = a->steps;
  out[5] = a->levels;
  out[6] = a->trigger_valid ? 1 : 0;
  out[7] = a->blocked_marks;
  out[8] = a->av_w;
  out[9] = a->av_h;
  out[10] = int(a->plan.size());
  out[11] = a->stagnant;
  int nt = 0;
  for (int i = 0; i < NCELL; ++i) nt += a->touched[i] ? 1 : 0;
  out[12] = nt;
  out[13] = a->escalation;
  out[14] = int(a->novelty.seen.size());
  out[15] = a->restarts;
  int goals = 0, walls = 0;
  for (std::map<int, ColorRule>::const_iterator it = a->rules.begin();
       it != a->rules.end(); ++it) {
    if (it->second.is_goal()) ++goals;
    if (it->second.is_wall()) ++walls;
  }
  out[16] = int(a->rules.size());
  out[17] = goals;
  out[18] = walls;
  out[19] = a->wm.observations;
  out[20] = int(a->wm.disp.size());
  out[21] = int(a->scene_seen.size());
  out[22] = int(a->cur_scene.objs.size());
  out[23] = int(a->bfs.size());
  out[24] = int(a->idd_tried);
  out[25] = int(a->idd_actions.size());
  out[26] = int(a->idd_solution.size());
  out[27] = int(a->idd_pruned);
  out[28] = int(a->bfs_head);
  out[29] = int(a->ge_route.size());
  out[30] = int(a->ge_tried.size());
}
