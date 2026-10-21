#!/bin/sh
# macOS/Linux build of the kit's helpers (clang or gcc). Windows: run.ps1.
#   hostsim_strict    no FMA contraction (-ffp-contract=off)
#   hostsim_contract  the compiler's default (clang: contraction within an expression)
#   pp_cli            the portable kernel, as the app builds it
set -e
cd "$(dirname "$0")"
CXX=${CXX:-c++}; CC=${CC:-cc}
$CXX -std=c++20 -O2 -ffp-contract=off -Isim/stubs -Isim/src sim/hostsim.cpp sim/src/game.cpp sim/src/trace.cpp -o hostsim_strict
$CXX -std=c++20 -O2 -Isim/stubs -Isim/src sim/hostsim.cpp sim/src/game.cpp sim/src/trace.cpp -o hostsim_contract
$CC -O2 -ffp-contract=off -fno-fast-math -std=c99 -o pp_cli portable/pp_cli.c portable/policy_portable.c
echo "built hostsim_strict hostsim_contract pp_cli"
