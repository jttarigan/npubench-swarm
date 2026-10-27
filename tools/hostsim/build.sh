#!/bin/sh
# Builds the host simulation harness (see hostsim.cpp). Output: ./hostsim
set -e
cd "$(dirname "$0")"
SRC=../../app/src/main/cpp/src
clang++ -std=c++20 -O2 -Wall -Istubs -I"$SRC" hostsim.cpp "$SRC/game.cpp" "$SRC/trace.cpp" -o hostsim
