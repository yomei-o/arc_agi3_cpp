// The shape the agent would actually use: C++ owns the game, Lua is handed a
// few high-level primitives and writes the policy with them.
#include <cstdio>
extern "C" {
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"
}
static int l_find_objects(lua_State* L) {          // stand-in for the real one
  lua_newtable(L);
  for (int i = 1; i <= 3; ++i) {
    lua_newtable(L);
    lua_pushinteger(L, i * 5); lua_setfield(L, -2, "x");
    lua_pushinteger(L, i * 7); lua_setfield(L, -2, "y");
    lua_rawseti(L, -2, i);
  }
  return 1;
}
static int l_move_to(lua_State* L) {
  std::printf("move_to(%lld,%lld)\n", (long long)luaL_checkinteger(L, 1),
                                      (long long)luaL_checkinteger(L, 2));
  lua_pushboolean(L, 1);
  return 1;
}
int main() {
  lua_State* L = luaL_newstate();
  luaL_openlibs(L);
  lua_register(L, "find_objects", l_find_objects);
  lua_register(L, "move_to", l_move_to);
  const char* policy =
      "for _, o in ipairs(find_objects()) do move_to(o.x, o.y) end\n"
      "print('lua policy ran', _VERSION)\n";
  if (luaL_dostring(L, policy)) {
    std::printf("LUA ERROR: %s\n", lua_tostring(L, -1));
    return 1;
  }
  lua_close(L);
  std::printf("EMBED_OK\n");
  return 0;
}
