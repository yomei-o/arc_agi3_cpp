#include <RDGeneral/versions.h>

const char * RDKit::rdkitVersion = "2026.03.1";

// The Boost version as detected at build time.
// CMake's Boost_LIB_VERSION is defined by the FindBoost.cmake module
// to be the same as the value from <boost/version.hpp>
const char * RDKit::boostVersion = "";

// The system/compiler on which RDKit was built as detected at build time.
const char * RDKit::rdkitBuild = "Windows|10.0.26200||MSVC|64-bit";
