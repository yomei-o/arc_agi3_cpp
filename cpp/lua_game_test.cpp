// Prove the Lua game library on a game we control.
//
// The ARC environment lives behind Python and a network, so testing against it
// confuses two questions: whether the binding works, and whether the policy is
// any good. This file answers only the first. The mock is the smallest game
// with the properties that matter - a sprite that moves under four buttons, a
// wall it cannot pass, and a goal that advances the level - so a script that
// finishes it must have read the board, seen what its own action did, and acted
// on the answer.
#include "arc3_core.cpp"
#include "arc3_lua.h"

#include <cstdio>
#include <cstring>

namespace {

constexpr int SC = 2;        // pixels per logical cell, to exercise the scaling
constexpr int BW = 16, BH = 12;

struct MockGame {
  int ax = 1, ay = 1;        // the avatar, in logical cells
  int gx = 13, gy = 9;       // the goal
  int level = 0;
  int actions = 0;

  bool wall(int x, int y) const {
    if (x < 0 || y < 0 || x >= BW || y >= BH) return true;
    return x == 8 && y != 5;                     // one wall with one door
  }

  Grid frame() const {
    Grid g;                                      // colour 0 is the floor
    for (int y = 0; y < BH; ++y)
      for (int x = 0; x < BW; ++x) {
        int c = 0;
        if (wall(x, y)) c = 5;
        else if (x == gx && y == gy) c = 3;
        if (x == ax && y == ay) c = 9;
        for (int dy = 0; dy < SC; ++dy)
          for (int dx = 0; dx < SC; ++dx)
            g.c[(y * SC + dy) * W + (x * SC + dx)] = int8_t(c);
      }
    return g;
  }

  void step(int a, int, int) {
    ++actions;
    int nx = ax, ny = ay;
    if (a == 1) --ny;
    else if (a == 2) ++ny;
    else if (a == 3) --nx;
    else if (a == 4) ++nx;
    else return;                                 // clicks do nothing here
    if (!wall(nx, ny)) { ax = nx; ay = ny; }
    if (ax == gx && ay == gy) { ++level; ax = 1; ay = 1; }
  }
};

// The policy a model might write: find yourself, find the goal, walk there,
// and notice when a button does nothing.
const char* POLICY = R"LUA(
local function me()
  for _, o in ipairs(objects()) do
    if o.area == 1 and o.colour == 9 then return o end
  end
end

local function goal()
  for _, o in ipairs(objects()) do
    if o.colour == 3 then return o end
  end
end

say("board at the start:")
say(board_text())

local blocked = 0
for i = 1, 400 do
  local a, g = me(), goal()
  if not a or not g then break end
  if level() >= 2 then say("two levels done in " .. steps() .. " actions"); return end

  local button
  if blocked % 2 == 0 then
    if     g.x > a.x then button = 4
    elseif g.x < a.x then button = 3
    elseif g.y > a.y then button = 2
    else                  button = 1 end
  else
    -- the sideways step that gets round the wall
    if g.y > a.y then button = 2 elseif g.y < a.y then button = 1 else button = 4 end
  end

  local changed = press(button)
  if not changed then
    blocked = blocked + 1
  else
    -- confirm the library reports the move, not just the pixels
    local moved = false
    for _, e in ipairs(changes()) do
      if e.kind == "moved" and e.colour == 9 then moved = true end
    end
    if not moved then blocked = blocked + 1 else blocked = 0 end
  end
end
say("gave up at level " .. level() .. " after " .. steps() .. " actions")
)LUA";

}  // namespace

int main() {
  MockGame game;
  arc3lua::Host hs;
  arc3lua::open_library(hs);

  std::vector<int> avail = {1, 2, 3, 4, 6};
  arc3lua::observe(hs, game.frame(), game.level, avail);

  if (!arc3lua::start(hs, POLICY)) {
    std::printf("LOAD FAILED: %s\n", hs.error.c_str());
    return 1;
  }

  int a = 0, x = 0, y = 0, guard = 0;
  for (;;) {
    int rc = arc3lua::resume(hs, &a, &x, &y);
    if (rc != arc3lua::LUA_WANTS_ACTION) {
      std::printf("%s", hs.log.c_str());
      if (rc == arc3lua::LUA_FAILED) {
        std::printf("RUNTIME ERROR: %s\n", hs.error.c_str());
        return 1;
      }
      break;
    }
    if (++guard > 2000) { std::printf("the policy never stopped\n"); return 1; }
    game.step(a, x, y);
    arc3lua::observe(hs, game.frame(), game.level, avail);
  }

  std::printf("levels=%d  actions=%d\n", game.level, game.actions);
  if (game.level < 2) { std::printf("FAILED: the script did not finish two levels\n"); return 1; }
  std::printf("LUA_GAME_OK\n");
  return 0;
}
