// A game the LLM can play, expressed as a Lua library.
//
// The problem this solves is not "call C++ from Lua" - that part is easy - but
// the shape of a turn. The agent is driven from outside: Python owns the ARC
// environment, asks us for one action, applies it, and comes back with the new
// frame. A policy written as a script wants the opposite shape: it wants to say
// press(3), look at what happened, and carry on, as a person would.
//
// Lua coroutines join the two. The script runs inside a coroutine; press() and
// click() are C functions that lua_yield the action they were asked for; the
// host returns it, steps the game, and resumes the coroutine with the new
// board. The script never learns it was suspended. This is the one place where
// the choice of Lua over Python pays a concrete debt - the same trick with
// Python generators needs the interpreter's frame machinery, where here it is
// two calls.
//
// Included from arc3_core.cpp, after Grid / Box / components() / detect_scale().

#ifndef ARC3_LUA_H
#define ARC3_LUA_H

extern "C" {
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"
}

#include <map>
#include <string>
#include <vector>

namespace arc3lua {

// The glyph table is the one view_game.py prints, so a board in a prompt and a
// board in a log are the same text. Nothing about the model depends on it, but
// everything about reading its answers does.
static const char* GLYPH = ".:-=+*#%@ABCDEFG";

inline char glyph_of(int colour) { return GLYPH[((colour % 16) + 16) % 16]; }

struct Event {
  const char* kind;     // "moved" | "appeared" | "vanished" | "recoloured"
  int colour, area;
  int fx, fy, tx, ty;
};

// How far a thing may travel in one action and still be called the same thing.
// Without a cap the matcher pairs objects across the whole board: on a level
// with a dozen identical 4-cell shapes it reports a chain of impossible
// journeys, each object "moving" to where the next one already was. Something
// that far away is not evidence of a move, it is evidence of two objects.
constexpr int MAX_MOVE = 5;

struct Thing {
  int colour, area, cx, cy, w, h, minx, miny;
};

inline std::vector<Thing> things_of(const Grid& g, int scale) {
  std::set<int> bg = background_colors(g);
  std::vector<int> cols;
  std::vector<Box> comps = components(g, bg, &cols);
  std::vector<Thing> out;
  out.reserve(comps.size());
  for (size_t i = 0; i < comps.size(); ++i) {
    const Box& b = comps[i];
    Thing t;
    t.colour = (i < cols.size()) ? cols[i] : 0;
    t.area = b.area / (scale * scale ? scale * scale : 1);
    if (t.area <= 0) t.area = 1;
    t.cx = b.cx() / scale;
    t.cy = b.cy() / scale;
    t.w = (b.w() + scale - 1) / scale;
    t.h = (b.h() + scale - 1) / scale;
    t.minx = b.minx / scale;
    t.miny = b.miny / scale;
    out.push_back(t);
  }
  return out;
}

// Describe a step as things happening to objects rather than as cells flipping.
//
// This is not a nicety. The same 27B, shown the same board and the same action,
// answers "change (6,9) to F" when handed a list of changed cells and "the
// player controls the F object at (6,10)" when handed this. A model reasons in
// the vocabulary it is given, so the matching is done here, where the board is,
// and the script is handed the result instead of the evidence for it.
inline std::vector<Event> diff_things(const std::vector<Thing>& before,
                                      const std::vector<Thing>& after) {
  std::vector<Event> out;
  std::vector<char> used(after.size(), 0);
  for (size_t i = 0; i < before.size(); ++i) {
    const Thing& b = before[i];
    int best = -1, bestd = 1 << 30;
    for (size_t j = 0; j < after.size(); ++j) {
      if (used[j] || after[j].colour != b.colour) continue;
      if (std::abs(after[j].area - b.area) * 4 > std::max(b.area, 1)) continue;
      int d = std::abs(after[j].cx - b.cx) + std::abs(after[j].cy - b.cy);
      if (d < bestd && d <= MAX_MOVE) { best = int(j); bestd = d; }
    }
    if (best < 0) {
      Event e = {"vanished", b.colour, b.area, b.cx, b.cy, b.cx, b.cy};
      out.push_back(e);
      continue;
    }
    used[best] = 1;
    const Thing& a = after[best];
    if (a.cx != b.cx || a.cy != b.cy) {
      Event e = {"moved", b.colour, b.area, b.cx, b.cy, a.cx, a.cy};
      out.push_back(e);
    }
  }
  for (size_t j = 0; j < after.size(); ++j)
    if (!used[j]) {
      const Thing& a = after[j];
      Event e = {"appeared", a.colour, a.area, a.cx, a.cy, a.cx, a.cy};
      out.push_back(e);
    }
  return out;
}

// The board as a person would read it: one character per logical cell, with
// rulers, the way a go or shogi diagram is written. The 64x64 frame is a
// rendering - several pixels per cell with furniture around it - so printing it
// raw is unreadable and, worse, puts coordinates in the answer that no action
// can use.
inline std::string board_text(const Grid& g, int scale) {
  int w = W / scale, h = H / scale;
  std::string s;
  s.reserve(size_t(h + 3) * size_t(w + 6));
  s += "    ";
  for (int x = 0; x < w; ++x) s += char('0' + (x / 10) % 10);
  s += "\n    ";
  for (int x = 0; x < w; ++x) s += char('0' + x % 10);
  s += '\n';
  char buf[8];
  for (int y = 0; y < h; ++y) {
    std::snprintf(buf, sizeof buf, "%3d ", y);
    s += buf;
    for (int x = 0; x < w; ++x) s += glyph_of(g.at(x * scale, y * scale));
    s += '\n';
  }
  return s;
}

// ---------------------------------------------------------------------------

struct Host;
Host* host_of(lua_State* L);

struct Host {
  lua_State* L = nullptr;
  lua_State* co = nullptr;      // the coroutine the policy runs in
  bool finished = false;
  std::string error;            // non-empty once the script has failed

  Grid cur, prev;
  int scale = 1;
  int level = 0, steps = 0, state = 1;
  std::vector<int> avail;
  std::vector<Thing> cur_things, prev_things;
  std::string log;

  // set by press/click just before they yield
  int want_a = 0, want_x = 0, want_y = 0;
  int last_a = 0;               // what the host actually applied

  // What the host has worked out for itself, so the script need not.
  int av_colour = -1, av_area = -1;          // the avatar, once something moved
  std::map<int, Vec> button;                 // button -> the step it takes
  std::vector<uint8_t> blocked;              // logical cells we failed to enter
  std::map<int, int> wall_hits;              // colour -> times it stopped us
  int wall_after = 1;                        // how many times before we believe it
  int bw = 0, bh = 0;

  // move_to state, carried across the yields it makes
  int mv_tx = 0, mv_ty = 0, mv_left = 0;

  // The tabular policy, available to the script as explore().
  Agent* tab = nullptr;
  int tab_left = 0;

  ~Host() { if (L) lua_close(L); delete tab; }
};

// Learn the two things a walking game never tells you: which shape is you, and
// which buttons move it where.
//
// Both are free. The host already computes the object-level diff for changes(),
// so the sprite that moved under a button IS the avatar and the displacement IS
// what that button does. And when a button with a known step leaves the avatar
// where it was, the cell it was aimed at is a wall - which is the one fact a
// script written by a language model never keeps, and the reason the first one
// bounced off the same wall for twelve hundred actions.
inline void learn(Host& hs) {
  if (hs.last_a < 1 || hs.last_a > 5) return;
  std::vector<Event> ev = diff_things(hs.prev_things, hs.cur_things);
  for (size_t i = 0; i < ev.size(); ++i) {
    if (std::strcmp(ev[i].kind, "moved") != 0) continue;
    if (hs.av_colour >= 0 && ev[i].colour != hs.av_colour) continue;
    hs.av_colour = ev[i].colour;
    hs.av_area = ev[i].area;
    hs.button[hs.last_a] = Vec(ev[i].tx - ev[i].fx, ev[i].ty - ev[i].fy);
    return;
  }
  // nothing moved: if we know what this button does, we just met a wall
  std::map<int, Vec>::const_iterator it = hs.button.find(hs.last_a);
  if (it == hs.button.end() || hs.av_colour < 0) return;
  for (size_t i = 0; i < hs.prev_things.size(); ++i) {
    const Thing& t = hs.prev_things[i];
    if (t.colour != hs.av_colour || t.area != hs.av_area) continue;
    int nx = t.cx + it->second.dx, ny = t.cy + it->second.dy;
    if (nx < 0 || ny < 0 || nx >= hs.bw || ny >= hs.bh) return;
    hs.blocked[size_t(ny) * size_t(hs.bw) + size_t(nx)] = 1;
    // Generalise from the one cell to its colour - but only after it has
    // stopped us `wall_after` times.
    //
    // Remembering only the square we bumped into means learning a wall a square
    // at a time: on ka59 that was eighty actions to discover four cells of one
    // barrier. Believing a colour on the strength of a single bump is the
    // opposite error, and it is the worse one: across all 25 games it made
    // every route look impossible and the policy gave up inside twenty actions
    // everywhere. How many bumps it should take is a number, so it is measured
    // (ARC3_WALL_AFTER) rather than argued about.
    ++hs.wall_hits[int(hs.cur.at(nx * hs.scale, ny * hs.scale))];
    return;
  }
}

inline const Thing* avatar(const Host& hs) {
  for (size_t i = 0; i < hs.cur_things.size(); ++i)
    if (hs.cur_things[i].colour == hs.av_colour && hs.cur_things[i].area == hs.av_area)
      return &hs.cur_things[i];
  return nullptr;
}

// The first button of a shortest route to (tx,ty), or 0 if there is none.
//
// This is the whole argument for putting primitives in C++ rather than letting
// the model write its own loop. The score is (human actions / ours) squared, so
// a route that wanders is not merely inelegant, it is most of the score. A
// breadth-first search over the steps we have learned, avoiding the walls we
// have hit, is optimal for the map we know - and it costs the script one line.
inline int first_step_towards(const Host& hs, int tx, int ty) {
  const Thing* me = avatar(hs);
  if (!me || hs.button.empty() || hs.bw <= 0) return 0;
  if (me->cx == tx && me->cy == ty) return 0;

  const size_t n = size_t(hs.bw) * size_t(hs.bh);
  std::vector<int> first(n, -1);
  std::vector<uint8_t> seen(n, 0);
  std::vector<int> q;
  q.reserve(n);
  size_t start = size_t(me->cy) * size_t(hs.bw) + size_t(me->cx);
  seen[start] = 1;
  q.push_back(int(start));

  for (size_t qi = 0; qi < q.size(); ++qi) {
    int cur = q[qi];
    int cx = cur % hs.bw, cy = cur / hs.bw;
    for (std::map<int, Vec>::const_iterator it = hs.button.begin();
         it != hs.button.end(); ++it) {
      int nx = cx + it->second.dx, ny = cy + it->second.dy;
      if (nx < 0 || ny < 0 || nx >= hs.bw || ny >= hs.bh) continue;
      size_t ni = size_t(ny) * size_t(hs.bw) + size_t(nx);
      if (seen[ni] || hs.blocked[ni]) continue;
      if (!(nx == tx && ny == ty)) {
        std::map<int, int>::const_iterator w =
            hs.wall_hits.find(int(hs.cur.at(nx * hs.scale, ny * hs.scale)));
        if (w != hs.wall_hits.end() && w->second >= hs.wall_after) continue;
      }
      seen[ni] = 1;
      first[ni] = (cur == int(start)) ? it->first : first[size_t(cur)];
      if (nx == tx && ny == ty) return first[ni];
      q.push_back(int(ni));
    }
  }
  return 0;
}

inline void push_thing(lua_State* L, const Thing& t) {
  lua_newtable(L);
  lua_pushinteger(L, t.cx);     lua_setfield(L, -2, "x");
  lua_pushinteger(L, t.cy);     lua_setfield(L, -2, "y");
  lua_pushinteger(L, t.colour); lua_setfield(L, -2, "colour");
  lua_pushinteger(L, t.area);   lua_setfield(L, -2, "area");
  lua_pushinteger(L, t.w);      lua_setfield(L, -2, "w");
  lua_pushinteger(L, t.h);      lua_setfield(L, -2, "h");
  lua_pushinteger(L, t.minx);   lua_setfield(L, -2, "minx");
  lua_pushinteger(L, t.miny);   lua_setfield(L, -2, "miny");
  char gl[2] = {glyph_of(t.colour), 0};
  lua_pushstring(L, gl);        lua_setfield(L, -2, "glyph");
}

// --- the library ------------------------------------------------------------

inline int l_objects(lua_State* L) {
  Host* hs = host_of(L);
  lua_newtable(L);
  for (size_t i = 0; i < hs->cur_things.size(); ++i) {
    push_thing(L, hs->cur_things[i]);
    lua_rawseti(L, -2, int(i + 1));
  }
  return 1;
}

inline int l_changes(lua_State* L) {
  Host* hs = host_of(L);
  std::vector<Event> ev = diff_things(hs->prev_things, hs->cur_things);
  lua_newtable(L);
  for (size_t i = 0; i < ev.size(); ++i) {
    lua_newtable(L);
    lua_pushstring(L, ev[i].kind);   lua_setfield(L, -2, "kind");
    lua_pushinteger(L, ev[i].colour);lua_setfield(L, -2, "colour");
    lua_pushinteger(L, ev[i].area);  lua_setfield(L, -2, "area");
    lua_pushinteger(L, ev[i].fx);    lua_setfield(L, -2, "fx");
    lua_pushinteger(L, ev[i].fy);    lua_setfield(L, -2, "fy");
    lua_pushinteger(L, ev[i].tx);    lua_setfield(L, -2, "x");
    lua_pushinteger(L, ev[i].ty);    lua_setfield(L, -2, "y");
    lua_pushinteger(L, ev[i].tx - ev[i].fx); lua_setfield(L, -2, "dx");
    lua_pushinteger(L, ev[i].ty - ev[i].fy); lua_setfield(L, -2, "dy");
    lua_rawseti(L, -2, int(i + 1));
  }
  return 1;
}

inline int l_board_text(lua_State* L) {
  Host* hs = host_of(L);
  std::string s = board_text(hs->cur, hs->scale);
  lua_pushlstring(L, s.data(), s.size());
  return 1;
}

inline int l_at(lua_State* L) {
  Host* hs = host_of(L);
  int x = int(luaL_checkinteger(L, 1)), y = int(luaL_checkinteger(L, 2));
  int w = W / hs->scale, h = H / hs->scale;
  if (x < 0 || y < 0 || x >= w || y >= h) { lua_pushnil(L); return 1; }
  lua_pushinteger(L, hs->cur.at(x * hs->scale, y * hs->scale));
  return 1;
}

inline int l_size(lua_State* L) {
  Host* hs = host_of(L);
  lua_pushinteger(L, W / hs->scale);
  lua_pushinteger(L, H / hs->scale);
  return 2;
}

inline int l_level(lua_State* L) { lua_pushinteger(L, host_of(L)->level); return 1; }
inline int l_steps(lua_State* L) { lua_pushinteger(L, host_of(L)->steps); return 1; }

inline int l_actions(lua_State* L) {
  Host* hs = host_of(L);
  lua_newtable(L);
  for (size_t i = 0; i < hs->avail.size(); ++i) {
    lua_pushinteger(L, hs->avail[i]);
    lua_rawseti(L, -2, int(i + 1));
  }
  return 1;
}

inline int l_say(lua_State* L) {
  Host* hs = host_of(L);
  const char* s = luaL_optstring(L, 1, "");
  hs->log += s;
  hs->log += '\n';
  return 0;
}

// press/click/reset: hand the action to the host and suspend.
//
// The continuation does the bookkeeping the script would otherwise have to do
// itself - it reports whether the board actually changed, which is the one
// question worth asking after every action and the thing the tabular policy
// spends its whole life measuring.
inline int act_resume(lua_State* L, int, lua_KContext) {
  Host* hs = host_of(L);
  lua_pushboolean(L, !(hs->cur == hs->prev));
  return 1;
}

inline int l_press(lua_State* L) {
  Host* hs = host_of(L);
  int n = int(luaL_checkinteger(L, 1));
  luaL_argcheck(L, n >= 1 && n <= 5, 1, "button is 1..5");
  hs->want_a = n; hs->want_x = 0; hs->want_y = 0;
  return lua_yieldk(L, 0, 0, act_resume);
}

inline int l_click(lua_State* L) {
  Host* hs = host_of(L);
  hs->want_a = A6;
  hs->want_x = int(luaL_checkinteger(L, 1)) * hs->scale;
  hs->want_y = int(luaL_checkinteger(L, 2)) * hs->scale;
  return lua_yieldk(L, 0, 0, act_resume);
}

// move_to(x, y [, limit]) - walk there, however many actions that takes.
//
// One Lua call, many yields: the continuation re-enters the same decision, so
// the script says where it wants to be and the host spends the actions. This is
// the shape every primitive here should have. A model is good at deciding where
// to go and bad at not bouncing off a wall on the way.
int move_drive(lua_State* L);

inline int move_cont(lua_State* L, int, lua_KContext) { return move_drive(L); }

inline int move_drive(lua_State* L) {
  Host* hs = host_of(L);
  const Thing* me = avatar(*hs);
  if (me && me->cx == hs->mv_tx && me->cy == hs->mv_ty) {
    lua_pushboolean(L, 1);
    return 1;
  }
  int b = (hs->mv_left-- > 0) ? first_step_towards(*hs, hs->mv_tx, hs->mv_ty) : 0;
  if (b == 0) {
    lua_pushboolean(L, 0);        // no route we know of, or out of patience
    return 1;
  }
  hs->want_a = b; hs->want_x = 0; hs->want_y = 0;
  return lua_yieldk(L, 0, 0, move_cont);
}

inline int l_move_to(lua_State* L) {
  Host* hs = host_of(L);
  hs->mv_tx = int(luaL_checkinteger(L, 1));
  hs->mv_ty = int(luaL_checkinteger(L, 2));
  hs->mv_left = int(luaL_optinteger(L, 3, 200));
  return move_drive(L);
}

// Who the host thinks you are, and what it has learned the buttons do.
inline int l_me(lua_State* L) {
  Host* hs = host_of(L);
  const Thing* me = avatar(*hs);
  if (!me) { lua_pushnil(L); return 1; }
  push_thing(L, *me);
  return 1;
}

inline int l_walls(lua_State* L) {
  Host* hs = host_of(L);
  lua_newtable(L);
  int k = 0;
  for (int y = 0; y < hs->bh; ++y)
    for (int x = 0; x < hs->bw; ++x)
      if (hs->blocked[size_t(y) * size_t(hs->bw) + size_t(x)]) {
        lua_newtable(L);
        lua_pushinteger(L, x); lua_setfield(L, -2, "x");
        lua_pushinteger(L, y); lua_setfield(L, -2, "y");
        lua_rawseti(L, -2, ++k);
      }
  return 1;
}

// explore(n) - hand the next n actions to the policy that actually wins.
//
// The scripted agent had, at this point, won nothing: neither a policy written
// by hand nor four written by the model finished a single level of the
// twenty-five, while the tabular counter policy finishes twelve. Arguing that
// the script ought to do better is not a plan. Letting it call the thing that
// works is: the model can then write "explore until something happens, then
// walk there", which is a strategy neither half can express alone.
int explore_drive(lua_State* L);

inline int explore_cont(lua_State* L, int, lua_KContext) { return explore_drive(L); }

inline int explore_drive(lua_State* L) {
  Host* hs = host_of(L);
  if (hs->tab_left-- <= 0) {
    lua_pushboolean(L, 1);
    return 1;
  }
  // The real state code, not a constant. The tabular policy is told the
  // state and returns RESET itself when that is the only legal action; handed
  // a hardcoded 1 it never learns the game ended, and every action after the
  // first GAME_OVER is thrown away. That alone took it from twelve levels to
  // one.
  hs->tab->observe(hs->cur.c.data(), hs->level, hs->state);
  int a = hs->tab->choose();
  hs->want_a = a;
  hs->want_x = hs->tab->pending_x;
  hs->want_y = hs->tab->pending_y;
  return lua_yieldk(L, 0, 0, explore_cont);
}

inline int l_explore(lua_State* L) {
  Host* hs = host_of(L);
  if (!hs->tab) {
    hs->tab = new Agent();
    std::vector<int> av = hs->avail;
    if (av.empty()) av.push_back(A1);
    hs->tab->init(av.data(), int(av.size()));
  }
  hs->tab_left = int(luaL_optinteger(L, 1, 50));
  return explore_drive(L);
}

inline int l_reset(lua_State* L) {
  Host* hs = host_of(L);
  hs->want_a = A_RESET; hs->want_x = 0; hs->want_y = 0;
  return lua_yieldk(L, 0, 0, act_resume);
}

// What to show the model between turns.
//
// Everything here is something the host worked out and the model would
// otherwise have to rediscover by spending actions: which shape it controls,
// what each button does, which colours have stopped it. Handing this over is
// the difference between asking the model to play and asking it to guess.
inline std::string state_text(Host& hs) {
  char buf[256];
  std::string s;
  std::snprintf(buf, sizeof buf, "level=%d actions_used=%d board=%dx%d\n",
                hs.level, hs.steps, hs.bw, hs.bh);
  s += buf;

  const Thing* me = avatar(hs);
  if (me) {
    std::snprintf(buf, sizeof buf,
                  "you control: the '%c' object of %d cells, now at (%d,%d)\n",
                  glyph_of(me->colour), me->area, me->cx, me->cy);
    s += buf;
  } else {
    s += "you control: not known yet - press each button once and it will be\n";
  }

  if (!hs.button.empty()) {
    s += "buttons:";
    for (std::map<int, Vec>::const_iterator it = hs.button.begin();
         it != hs.button.end(); ++it) {
      std::snprintf(buf, sizeof buf, " %d moves you (%+d,%+d)", it->first,
                    it->second.dx, it->second.dy);
      s += buf;
    }
    s += "\n";
  }

  if (!hs.wall_hits.empty()) {
    s += "colours that stopped you:";
    for (std::map<int, int>::const_iterator it = hs.wall_hits.begin();
         it != hs.wall_hits.end(); ++it) {
      std::snprintf(buf, sizeof buf, " '%c' x%d", glyph_of(it->first), it->second);
      s += buf;
    }
    s += "\n";
  }

  s += "objects now:\n";
  for (size_t k = 0; k < hs.cur_things.size() && k < 40; ++k) {
    const Thing& t = hs.cur_things[k];
    std::snprintf(buf, sizeof buf, "  '%c' area %d at (%d,%d) box %dx%d\n",
                  glyph_of(t.colour), t.area, t.cx, t.cy, t.w, t.h);
    s += buf;
  }
  if (hs.cur_things.size() > 40) s += "  ...\n";
  return s;
}

// --- the state -------------------------------------------------------------

static const char* HOST_KEY = "arc3.host";

inline Host* host_of(lua_State* L) {
  lua_getfield(L, LUA_REGISTRYINDEX, HOST_KEY);
  Host* hs = static_cast<Host*>(lua_touserdata(L, -1));
  lua_pop(L, 1);
  return hs;
}

// Take away everything that touches the machine.
//
// This is the whole of the sandbox, and it is four lines because the globals of
// a Lua state are an ordinary table that we own. The alternative - picking a
// safe subset of a language whose modules import each other - is the work the
// leading harness had to do by hand, and it is work that can be got wrong
// quietly. Here what is missing is missing because nothing put it there.
inline void seal(lua_State* L) {
  static const char* gone[] = {"io", "os", "package", "require", "dofile",
                               "loadfile", "load", "collectgarbage", "debug",
                               nullptr};
  for (int i = 0; gone[i]; ++i) {
    lua_pushnil(L);
    lua_setglobal(L, gone[i]);
  }
}

inline void open_library(Host& hs) {
  hs.L = luaL_newstate();
  luaL_openlibs(hs.L);
  seal(hs.L);

  lua_pushlightuserdata(hs.L, &hs);
  lua_setfield(hs.L, LUA_REGISTRYINDEX, HOST_KEY);

  static const luaL_Reg api[] = {
    {"objects",    l_objects},
    {"changes",    l_changes},
    {"board_text", l_board_text},
    {"at",         l_at},
    {"size",       l_size},
    {"level",      l_level},
    {"steps",      l_steps},
    {"actions",    l_actions},
    {"say",        l_say},
    {"press",      l_press},
    {"click",      l_click},
    {"move_to",    l_move_to},
    {"explore",    l_explore},
    {"me",         l_me},
    {"walls",      l_walls},
    {"restart",    l_reset},
    {nullptr, nullptr},
  };
  for (int i = 0; api[i].name; ++i) {
    lua_pushcfunction(hs.L, api[i].func);
    lua_setglobal(hs.L, api[i].name);
  }
}

inline void observe(Host& hs, const Grid& g, int level, int state,
                    const std::vector<int>& avail) {
  hs.prev = hs.cur;
  hs.prev_things = hs.cur_things;
  hs.cur = g;
  hs.level = level;
  hs.state = state;
  hs.avail = avail;
  hs.scale = detect_scale(g);
  if (hs.scale < 1) hs.scale = 1;
  hs.cur_things = things_of(g, hs.scale);

  int nw = W / hs.scale, nh = H / hs.scale;
  if (nw != hs.bw || nh != hs.bh) {
    hs.bw = nw; hs.bh = nh;
    hs.blocked.assign(size_t(nw) * size_t(nh), 0);
  }
  learn(hs);
}

// Load the policy. Returns false and fills hs.error if it does not compile.
inline bool start(Host& hs, const std::string& script) {
  // A second call replaces the script but keeps everything the host has
  // learned: the avatar, the buttons, the walls, the tabular policy's counters.
  // That is the point of loading more than once - the model gets to see a few
  // moves and write the next few, without the agent forgetting the game.
  hs.finished = false;
  hs.error.clear();
  hs.co = lua_newthread(hs.L);
  lua_setfield(hs.L, LUA_REGISTRYINDEX, "arc3.co");   // anchor against the GC
  if (luaL_loadbuffer(hs.co, script.data(), script.size(), "policy") != LUA_OK) {
    hs.error = lua_tostring(hs.co, -1) ? lua_tostring(hs.co, -1) : "load failed";
    hs.finished = true;
    return false;
  }
  return true;
}

enum { LUA_WANTS_ACTION = 1, LUA_DONE = 0, LUA_FAILED = -1 };

// Run the policy until it asks for an action, finishes, or breaks.
inline int resume(Host& hs, int* out_a, int* out_x, int* out_y) {
  if (hs.finished) return hs.error.empty() ? LUA_DONE : LUA_FAILED;
  int nres = 0;
  int rc = lua_resume(hs.co, hs.L, 0, &nres);
  if (rc == LUA_YIELD) {
    *out_a = hs.want_a; *out_x = hs.want_x; *out_y = hs.want_y;
    hs.last_a = hs.want_a;
    ++hs.steps;
    return LUA_WANTS_ACTION;
  }
  hs.finished = true;
  if (rc != LUA_OK) {
    const char* m = lua_tostring(hs.co, -1);
    hs.error = m ? m : "runtime error";
    return LUA_FAILED;
  }
  return LUA_DONE;
}

} // namespace arc3lua

#endif
