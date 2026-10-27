#!/bin/sh
# Host shared library of the portable policy for ctypes (analysis/policy_backends.py "pp");
# same flags as build.sh and the app's copy.
cd "$(dirname "$0")" && cc -O2 -ffp-contract=off -fno-fast-math -std=c99 -shared -fPIC -o libpp.dylib policy_portable.c
