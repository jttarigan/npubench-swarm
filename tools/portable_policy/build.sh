#!/bin/sh
# Host build of the portable policy (same flags as the app's copy).
cd "$(dirname "$0")" && cc -O2 -ffp-contract=off -fno-fast-math -std=c99 -o pp_cli pp_cli.c policy_portable.c
