# A from-scratch build of just the RDKit modules the forward model needs:
# SMILES parsing, the molecular graph, ring perception. Not RDKit's own
# CMakeLists.txt - that one unconditionally builds ~30 modules (drawing,
# 3D conformers, reactions, Python wrappers, ...) regardless of which flags
# are turned off, because almost everything is an unguarded add_subdirectory.
# Fighting that gave a day of unrelated build failures (ChemDraw, Freetype,
# Eigen, a C++20-only constexpr pattern) for features nothing here uses.
#
# This instead vendors the ~60 .cpp files that RDKit's own GraphMol/
# SmilesParse/RDGeneral/DataStructs/Geometry CMakeLists.txt actually list as
# belonging to those five libraries (see casmi/third_party/README.md for how
# that list was read off, not guessed), and links them in their real
# dependency order: RDGeneral -> DataStructs -> Geometry -> GraphMol ->
# SmilesParse. Extending this (e.g. adding Descriptors for formula/mass, or
# FileParsers) means copying the next module's file list the same way and
# adding four lines below - not re-fighting the upstream build.
cmake_minimum_required(VERSION 3.20)
project(rdkit_min CXX)

set(CMAKE_CXX_STANDARD 20)          # RDKit's Geometry/point.h uses constexpr
set(CMAKE_CXX_STANDARD_REQUIRED ON) # virtual destructors, legal only in C++20
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Boost REQUIRED)        # header-only usage throughout - no
                                     # compiled component needed (see README)
find_package(Threads REQUIRED)

if(MSVC)
  # the flex-generated lexers include <unistd.h> unconditionally unless told
  # not to - MSVC has no such header. RDKit's own SmilesParse/CMakeLists.txt
  # carries this same define for the same reason.
  add_definitions("/D YY_NO_UNISTD_H")
endif()

function(rdkit_min_lib name)
  file(GLOB srcs "${CMAKE_CURRENT_SOURCE_DIR}/${name}/*.cpp")
  add_library(${name} STATIC ${srcs})
  target_include_directories(${name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${name} PUBLIC Boost::boost Threads::Threads ${ARGN})
endfunction()

rdkit_min_lib(RDGeneral)
rdkit_min_lib(DataStructs RDGeneral)
rdkit_min_lib(Geometry DataStructs RDGeneral)
# RDKit's own headers expect GraphMol's own include dir on the path too
# (e.g. SmilesParse includes "GraphMol/RDKitBase.h" relative to Code/).
add_library(GraphMol STATIC)
file(GLOB GraphMol_srcs "${CMAKE_CURRENT_SOURCE_DIR}/GraphMol/*.cpp")
target_sources(GraphMol PRIVATE ${GraphMol_srcs})
target_include_directories(GraphMol PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(GraphMol PUBLIC Geometry DataStructs RDGeneral Boost::boost Threads::Threads)

# CIP (R/S) stereo descriptor assignment - SmilesParse.cpp calls into this
# during sanitization. MolEnumerator and FileParsers are NOT pulled in as
# real libraries: SmilesParse only needed two of their headers
# (LinkNode.h, MolFileStereochem.h, MolSGroupParsing.h), all inline/declare-
# only, copied directly into matching GraphMol/<name>/ paths so the
# #include lines resolve without the heavier modules behind them.
add_library(CIPLabeler STATIC)
file(GLOB CIPLabeler_srcs
     "${CMAKE_CURRENT_SOURCE_DIR}/GraphMol/CIPLabeler/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/GraphMol/CIPLabeler/configs/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/GraphMol/CIPLabeler/rules/*.cpp")
target_sources(CIPLabeler PRIVATE ${CIPLabeler_srcs})
target_include_directories(CIPLabeler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(CIPLabeler PUBLIC GraphMol)

add_library(SmilesParse STATIC)
file(GLOB SmilesParse_srcs "${CMAKE_CURRENT_SOURCE_DIR}/GraphMol/SmilesParse/*.cpp")
target_sources(SmilesParse PRIVATE ${SmilesParse_srcs})
target_include_directories(SmilesParse PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(SmilesParse PUBLIC GraphMol CIPLabeler)

add_executable(rdkit_min_smoketest ../rdkit_smoketest.cpp)
target_link_libraries(rdkit_min_smoketest PRIVATE SmilesParse GraphMol)

add_executable(fragment_enum ../fragment_enum.cpp)
target_link_libraries(fragment_enum PRIVATE SmilesParse GraphMol)

add_executable(forward_holdout forward_holdout.cpp)
target_link_libraries(forward_holdout PRIVATE SmilesParse GraphMol)

add_executable(forward_rerank forward_rerank.cpp)
target_link_libraries(forward_rerank PRIVATE SmilesParse GraphMol)

add_executable(pubchem_parse pubchem_parse.cpp)
target_link_libraries(pubchem_parse PRIVATE SmilesParse GraphMol)

add_executable(smiles_scan smiles_scan.cpp)
target_link_libraries(smiles_scan PRIVATE SmilesParse GraphMol)

add_executable(pool_recall_check pool_recall_check.cpp)
target_link_libraries(pool_recall_check PRIVATE SmilesParse GraphMol)

# Tiny dependency-free autograd engine (vendored from yomei-o/mini-yolov5-cpp's
# scratch/, Adam optimizer ported from yomei-o/yolov8_cpp's pure/optim.hpp) -
# used to learn the forward model's fragment-intensity weighting instead of
# the hand-tuned exp(-decay*k^2) rule. See casmi/cpp/autograd/.
add_library(autograd STATIC ../autograd/autograd.cpp ../autograd/nn_ops.cpp)
target_include_directories(autograd PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_executable(train_forward_model_nn train_forward_model_nn.cpp)
target_link_libraries(train_forward_model_nn PRIVATE SmilesParse GraphMol autograd)

add_executable(train_forward_model_rank train_forward_model_rank.cpp)
target_link_libraries(train_forward_model_rank PRIVATE SmilesParse GraphMol autograd)

add_executable(forward_holdout_nn forward_holdout_nn.cpp)
target_link_libraries(forward_holdout_nn PRIVATE SmilesParse GraphMol autograd)

add_executable(forward_rerank_nn forward_rerank_nn.cpp)
target_link_libraries(forward_rerank_nn PRIVATE SmilesParse GraphMol autograd)

add_executable(pool_candidate_rank pool_candidate_rank.cpp)
target_link_libraries(pool_candidate_rank PRIVATE SmilesParse GraphMol autograd)

# Reverse model (spectrum -> fingerprint) data prep: ECFP on-bits per molecule.
add_executable(fp_prep ../rev/fp_prep.cpp)
target_link_libraries(fp_prep PRIVATE SmilesParse GraphMol)

# Runtime candidate retrieval: whole-PubChem window + pool, reverse-model scored.
add_executable(cand_rank ../rev/cand_rank.cpp)
target_link_libraries(cand_rank PRIVATE SmilesParse GraphMol)

# Cascade stage 2: forward-MLP rescoring + score fusion over cand_rank's shortlist.
add_executable(fwd_rescore fwd_rescore.cpp)
target_link_libraries(fwd_rescore PRIVATE SmilesParse GraphMol autograd)

# Molecular formula from MS/MS by sub-formula explanation of fragment peaks
# (no training data).
add_executable(formula_infer formula_infer.cpp)
target_link_libraries(formula_infer PRIVATE SmilesParse GraphMol)

# Physics-shaped fragment intensities (bond-class barriers, proton-affinity
# charge retention, H-shift weights), fitted by coordinate search.
add_executable(phys_rank phys_rank.cpp)
target_link_libraries(phys_rank PRIVATE SmilesParse GraphMol autograd)
