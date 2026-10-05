// Lua as one C++ translation unit.
//
// Our Kaggle notebook compiles a single embedded C++ string with g++ and loads
// the result through ctypes; there is no build system and no second file. So
// the question that matters is not "does Lua build" - it does, everywhere - but
// "does the whole interpreter go into one .cpp with the C++ compiler we already
// have there". That is what this file tests. Lua 5.4 does not ship onelua.c, so
// the includes are listed by hand.
extern "C" {
#include "lprefix.h"
}
// C linkage for the whole interpreter. Compiling Lua as C++ mangles every
// symbol, so a translation unit that declares the API the normal way -
// extern "C", as lua.hpp does - fails to link against it. Wrapping the
// includes fixes that and keeps Lua linkable from C as well.
extern "C" {
#define LUA_CORE
#define LUA_LIB
#define ltable_c
#define lvm_c
#include "lapi.c"
#include "lcode.c"
#include "lctype.c"
#include "ldebug.c"
#include "ldo.c"
#include "ldump.c"
#include "lfunc.c"
#include "lgc.c"
#include "llex.c"
#include "lmem.c"
#include "lobject.c"
#include "lopcodes.c"
#include "lparser.c"
#include "lstate.c"
#include "lstring.c"
#include "ltable.c"
#include "ltm.c"
#include "lundump.c"
#include "lvm.c"
#include "lzio.c"
#include "lauxlib.c"
#include "lbaselib.c"
#include "lcorolib.c"
#include "ldblib.c"
// io/os/loadlib are compiled in because linit.c refers to their luaopen_*.
// Keeping them out of the sandbox is a runtime decision - clear the globals
// after opening - not a compile-time one.
#include "liolib.c"
#include "loslib.c"
#include "loadlib.c"
#include "lmathlib.c"
#include "lstrlib.c"
#include "ltablib.c"
#include "lutf8lib.c"
#include "linit.c"
}
