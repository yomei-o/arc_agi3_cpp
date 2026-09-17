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
#include <cstring>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <vector>

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
                          int only_color) {
  Motion best;
  int best_area = 1 << 30;

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
  std::map<uint64_t, std::vector<Act> > ge_route;   // cheapest route to a state
  std::map<uint64_t, int> ge_tried;                 // alphabet entries tried there
  uint64_t ge_here;
  bool ge_ready;

  WorldModel wm;
  Scene cur_scene;
  bool have_scene;
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
        idd_active(false), idd_try_solution(false), ge_here(0), ge_ready(false),
        explore_mode(2), rollout_len(60), since_restart(0), have_scene(false),
        sticky_a(-1), sticky_x(0), sticky_y(0),
        rule_target(-1), repeats(0), escalation(0), last_state(0),
        replaying(false), restarts(0) {
    click_x = click_y = -1;
    click_live = false;
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
        if (cur.c[i] == it->first) t.push_back(i);
    }
    return t;
  }

  // Squares of rare colours, rarest first - the strongest prior we have for
  // where a level ends when we have not yet seen one end.
  std::vector<int> rare_targets() {
    std::array<int, 32> h = color_counts();
    std::vector<std::pair<int, int> > order;   // (count, colour)
    for (int c = 0; c < 32; ++c)
      if (h[c] > 0 && h[c] < NCELL / 50 && c != av_color) order.push_back(std::make_pair(h[c], c));
    std::sort(order.begin(), order.end());
    std::vector<int> t;
    for (size_t k = 0; k < order.size(); ++k) {
      int col = order[k].second;
      std::map<int, ColorRule>::const_iterator r = rules.find(col);
      if (r != rules.end() && r->second.is_wall()) continue;
      for (int i = 0; i < NCELL; ++i)
        if (cur.c[i] == col && !touched[i]) t.push_back(i);
      if (!t.empty()) break;    // deal with the rarest colour present first
    }
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
      Motion m = detect_translation(prev, cur, bg, av_color);
      if (!m.found && av_color >= 0) {
        // The avatar did not move; maybe something else did, and maybe we are
        // tracking the wrong colour. Only re-open the question when lost.
        if (ax < 0) m = detect_translation(prev, cur, bg, -1);
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
      scene_seen.insert(cur_scene.hash());
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
      size_t take = idd_actions.empty() ? 10 : 4;
      for (size_t i = 0; i < order.size() && i < take; ++i) {
        const Box& b = comps[order[i].second];
        idd_actions.push_back(Act(A6, b.cx(), b.cy()));
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

  int choose() {
    std::set<int> bg = background_colors(cur);

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
      if (last_state != 3) {
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
      if (p.empty()) p = plan_to(rare_targets());
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
