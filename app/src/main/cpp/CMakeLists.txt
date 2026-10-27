cmake_minimum_required(VERSION 3.22.1)
project(npubench CXX C)

# GameActivity + native_app_glue (glue objects are inside the static lib)
find_package(game-activity REQUIRED CONFIG)

# LiteRT 1.4.2 — headers vendored from the official AARs, .so packaged via jniLibs.
# C API + NNAPI delegate symbols verified exported (see README).
set(LITERT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/litert)
# Gradle stages clean (non-OneDrive) copies of the libs and passes their dir;
# fall back to the in-repo canonical copies for standalone CMake use.
if(NOT DEFINED LITERT_LIB_DIR)
    set(LITERT_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI})
endif()
set(JNILIBS_DIR ${LITERT_LIB_DIR})

# Linked by absolute path (NOT as IMPORTED targets): both .so carry proper
# SONAMEs, live in jniLibs (so they are packaged), and this sidesteps AGP
# trying to snapshot/copy imported libs into its obj folder on Windows.

# Study 3 portable policy (shared with the Mac tools): plain C, fixed summation order, no
# FMA contraction, so it computes the same float32 bits on every platform.
set(PORTABLE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../tools/portable_policy)
set_source_files_properties(${PORTABLE_DIR}/policy_portable.c PROPERTIES
    COMPILE_OPTIONS "-O2;-ffp-contract=off;-fno-fast-math;-std=c99")

add_library(npubench SHARED
    src/main.cpp
    src/engine.cpp
    src/renderer.cpp
    src/sprites.cpp
    src/input.cpp
    src/gesture.cpp
    src/nn.cpp
    src/game.cpp
    src/csvlog.cpp
    src/trace.cpp
    src/prevention.cpp
    ${PORTABLE_DIR}/policy_portable.c)

target_include_directories(npubench PRIVATE ${LITERT_DIR}/include src ${PORTABLE_DIR})
target_compile_features(npubench PRIVATE cxx_std_20)
target_compile_options(npubench PRIVATE -Wall -O2)

# Force the linker to keep GameActivity's JNI entry point from the static lib
# (nothing in our code references that object, so it would be dropped).
target_link_options(npubench PRIVATE
    "LINKER:-u,Java_com_google_androidgamesdk_GameActivity_initializeNativeCode")

target_link_libraries(npubench
    game-activity::game-activity_static
    ${JNILIBS_DIR}/libtensorflowlite_jni.so
    ${JNILIBS_DIR}/libtensorflowlite_gpu_jni.so
    android
    log
    EGL
    GLESv3
    m)
