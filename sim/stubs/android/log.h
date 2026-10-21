#pragma once
// Host stub: route Android logging to stderr.
#include <cstdio>
enum { ANDROID_LOG_INFO = 4, ANDROID_LOG_WARN = 5, ANDROID_LOG_ERROR = 6 };
#define __android_log_print(prio, tag, ...) (fprintf(stderr, __VA_ARGS__), fputc('\n', stderr))
