cmake_minimum_required(VERSION 3.24)

project(kokopop
  VERSION 0.1.0
  DESCRIPTION "Standalone Kokoro GGUF runtime"
  LANGUAGES C CXX)

option(KOKOPOP_BUILD_TESTS "Build kokopop unit tests" ON)
option(KOKOPOP_BUILD_TOOLS "Build kokopop command line tools" ON)
option(KOKOPOP_ENABLE_METAL "Enable Metal (GPU) backend" OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Use -O3 for Release builds instead of the default -O2
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_C_FLAGS_RELEASE "-O3")

include(FetchContent)

FetchContent_Declare(
  ggml
  GIT_REPOSITORY https://github.com/ggml-org/ggml.git
  GIT_TAG        ac6f7b44f60fde0091f0b3d99afde48f8c99b13a
  GIT_SHALLOW    TRUE
  # Fix 3: force scalar (non-float4) NORM kernel — the float4 path corrupts
  #         values for certain token patterns (e.g. comma phrases).
  # Fix 4: dispatch CONV_TRANSPOSE_1D with 32 threads/group and simd_sum
  #         reduction, avoiding GPU watchdog on large generator dims.
  # Apply only when the patch still matches. This keeps re-configure
  # idempotent without leaving .rej files in FetchContent sources.
  PATCH_COMMAND
    ${CMAKE_COMMAND} -DPATCH_FILE=${CMAKE_SOURCE_DIR}/patches/ggml-metal-norm-scalar.patch -P ${CMAKE_SOURCE_DIR}/tools/apply_patch_if_needed.cmake
    COMMAND
    ${CMAKE_COMMAND} -DPATCH_FILE=${CMAKE_SOURCE_DIR}/patches/ggml-conv-transpose-1d-simd.patch -P ${CMAKE_SOURCE_DIR}/tools/apply_patch_if_needed.cmake
)
FetchContent_MakeAvailable(ggml)

FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG        v2.4.11
  GIT_SHALLOW    TRUE
)
# CMake 4.x removed support for cmake_minimum_required < 3.5.
# Set before MakeAvailable so the policy applies when doctest's CMakeLists is processed.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_MakeAvailable(doctest)
unset(CMAKE_POLICY_VERSION_MINIMUM)

FetchContent_Declare(
  yyjson
  GIT_REPOSITORY https://github.com/ibireme/yyjson.git
  GIT_TAG        0.12.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(yyjson)

find_package(Threads REQUIRED)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(ESPEAK_NG QUIET espeak-ng)
endif()
if(ESPEAK_NG_FOUND)
  set(KOKOPOP_ESPEAK_LIBRARIES ${ESPEAK_NG_LIBRARIES})
  set(KOKOPOP_ESPEAK_INCLUDE_DIRS ${ESPEAK_NG_INCLUDE_DIRS})
else()
  find_path(ESPEAK_NG_INCLUDE_DIR espeak-ng/speak_lib.h)
  find_library(ESPEAK_NG_LIBRARY espeak-ng)
  if(NOT ESPEAK_NG_INCLUDE_DIR OR NOT ESPEAK_NG_LIBRARY)
    message(FATAL_ERROR "libespeak-ng is required but was not found")
  endif()
  set(KOKOPOP_ESPEAK_LIBRARIES ${ESPEAK_NG_LIBRARY})
  set(KOKOPOP_ESPEAK_INCLUDE_DIRS ${ESPEAK_NG_INCLUDE_DIR})
endif()

option(KOKOPOP_ENABLE_OPUS "Enable Ogg/Opus streaming output" ON)

if(KOKOPOP_ENABLE_OPUS)
  if(NOT PkgConfig_FOUND)
    find_package(PkgConfig REQUIRED)
  endif()
  pkg_check_modules(OPUS REQUIRED opus)
  pkg_check_modules(OGG REQUIRED ogg)
  pkg_check_modules(OPUSENC REQUIRED libopusenc)
  message(STATUS "kokopop: Ogg/Opus output enabled")
else()
  message(STATUS "kokopop: Ogg/Opus output disabled")
endif()

# Core utilities
set(_kokopop_core_sources
  src/core/error.cpp
  src/core/replace.cpp
  src/core/utf8.cpp
  src/core/wav.cpp
  src/core/ogg_opus_encoder.cpp
)

# Backend module
set(_kokopop_backend_sources
  src/backend/cpu.cpp
  src/backend/registry.cpp
)
if(KOKOPOP_ENABLE_METAL)
  list(APPEND _kokopop_backend_sources src/backend/metal.cpp)
  list(APPEND _kokopop_backend_sources src/backend/metal_lstm.mm)
  list(APPEND _kokopop_backend_sources src/backend/metal_stft.mm)
  list(APPEND _kokopop_backend_sources src/backend/metal_vocoder.mm)
endif()

# Model module (GGUF loading)
set(_kokopop_model_sources
  src/model/model.cpp
)

# Inference module (Kokoro graph operations + audio utilities)
set(_kokopop_inference_sources
  src/inference/kokoro.cpp
  src/inference/graph_ops.cpp
  src/inference/audio_utils.cpp
  src/inference/lstm_fused.cpp
)

# Synthesis module (phonemization + chunking + main synth)
set(_kokopop_synthesis_sources
  src/synthesis/phonemizer.cpp
  src/synthesis/synth.cpp
  src/synthesis/chunker/chunker.cpp
  src/synthesis/chunker/text_normalizer.cpp
  src/synthesis/chunker/text_splitter.cpp
  src/synthesis/g2p/zh_g2p.cpp
)

# Audio post-processing
set(_kokopop_audio_sources
  src/audio/audio_postprocess.cpp
)

# Streaming module
set(_kokopop_streaming_sources
  src/streaming/streaming.cpp
)

# STDIO streaming module
set(_kokopop_stdio_sources
  src/stdio/stdio_stream.cpp
)

# Playback module
set(_kokopop_playback_sources
  src/playback/playback_stdout.cpp
  src/playback/playback_factory.cpp
)

# Core Audio on macOS
if(APPLE)
  list(APPEND _kokopop_playback_sources src/playback/playback_coreaudio.cpp)
  set(KOKOPOP_HAS_COREAUDIO 1)
endif()

add_library(kokopop
  ${_kokopop_core_sources}
  ${_kokopop_backend_sources}
  ${_kokopop_model_sources}
  ${_kokopop_inference_sources}
  ${_kokopop_synthesis_sources}
  ${_kokopop_audio_sources}
  ${_kokopop_streaming_sources}
  ${_kokopop_stdio_sources}
  ${_kokopop_playback_sources}
  src/kokopop.cpp
)

target_include_directories(kokopop
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    $<BUILD_INTERFACE:${yyjson_SOURCE_DIR}/src>
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model
    ${CMAKE_CURRENT_SOURCE_DIR}/src/inference
    ${CMAKE_CURRENT_SOURCE_DIR}/src/synthesis
    ${CMAKE_CURRENT_SOURCE_DIR}/src/synthesis/chunker
    ${CMAKE_CURRENT_SOURCE_DIR}/src/synthesis/g2p
    ${CMAKE_CURRENT_SOURCE_DIR}/src/audio
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend
    ${CMAKE_CURRENT_SOURCE_DIR}/src/streaming
    ${CMAKE_CURRENT_SOURCE_DIR}/src/stdio
    ${CMAKE_CURRENT_SOURCE_DIR}/src/playback
    ${KOKOPOP_ESPEAK_INCLUDE_DIRS}
)

target_link_libraries(kokopop
  PRIVATE
    ggml
    Threads::Threads
    ${KOKOPOP_ESPEAK_LIBRARIES}
    yyjson
)

if(KOKOPOP_ENABLE_OPUS)
  target_compile_definitions(kokopop PUBLIC KOKOPOP_HAS_OPUS)
  target_include_directories(kokopop PRIVATE
    ${OPUSENC_INCLUDE_DIRS}
    ${OPUS_INCLUDE_DIRS}
    ${OGG_INCLUDE_DIRS}
  )
  target_link_libraries(kokopop PRIVATE
    ${OPUSENC_LIBRARIES}
    ${OPUS_LIBRARIES}
    ${OGG_LIBRARIES}
  )
endif()

if(APPLE AND KOKOPOP_HAS_COREAUDIO)
  target_link_libraries(kokopop PRIVATE
    "-framework AudioToolbox"
    "-framework CoreAudio"
  )
  message(STATUS "kokopop: Core Audio playback enabled (macOS)")
endif()

if(KOKOPOP_ENABLE_METAL)
  target_compile_definitions(kokopop PUBLIC KOKOPOP_HAS_METAL)
  target_link_libraries(kokopop PRIVATE
    "-framework Metal"
    "-framework Foundation"
  )
  message(STATUS "kokopop: Metal backend enabled")
else()
  message(STATUS "kokopop: Metal backend disabled (CPU-only)")
endif()

if(MSVC)
  target_compile_options(kokopop PRIVATE /W4)
else()
  target_compile_options(kokopop PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(KOKOPOP_BUILD_TOOLS)
  add_executable(kokopop_say tools/kokopop_say.cpp)
  target_include_directories(kokopop_say PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/playback
  )
  target_link_libraries(kokopop_say PRIVATE kokopop)

  add_executable(kokopop_probe tools/kokopop_probe.cpp)
  target_include_directories(kokopop_probe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model
    ${CMAKE_CURRENT_SOURCE_DIR}/src/inference
    ${CMAKE_CURRENT_SOURCE_DIR}/src/synthesis
    ${CMAKE_CURRENT_SOURCE_DIR}/src/synthesis/chunker
    ${CMAKE_CURRENT_SOURCE_DIR}/src/synthesis/g2p
    ${CMAKE_CURRENT_SOURCE_DIR}/src/audio
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend
    ${CMAKE_CURRENT_SOURCE_DIR}/src/streaming
    ${CMAKE_CURRENT_SOURCE_DIR}/src/playback
    $<BUILD_INTERFACE:${ggml_SOURCE_DIR}/include>
  )
  target_link_libraries(kokopop_probe PRIVATE kokopop)

  # Streaming tool (stdio + HTTP server)
  add_executable(kokopop_stream
    tools/kokopop_stream.cpp
    src/http/http_server.cpp
    src/http/async_server.cpp
    src/http/synthesis_scheduler.cpp
  )
  target_include_directories(kokopop_stream PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model
    ${CMAKE_CURRENT_SOURCE_DIR}/src/inference
    ${CMAKE_CURRENT_SOURCE_DIR}/src/synthesis
    ${CMAKE_CURRENT_SOURCE_DIR}/src/synthesis/chunker
    ${CMAKE_CURRENT_SOURCE_DIR}/src/audio
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend
    ${CMAKE_CURRENT_SOURCE_DIR}/src/streaming
    ${CMAKE_CURRENT_SOURCE_DIR}/src/stdio
    ${CMAKE_CURRENT_SOURCE_DIR}/src/playback
    ${CMAKE_CURRENT_SOURCE_DIR}/src/http
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    $<BUILD_INTERFACE:${ggml_SOURCE_DIR}/include>
  )
  target_link_libraries(kokopop_stream PRIVATE kokopop yyjson)

  # Audio playback tool
  add_executable(kokopop_play tools/kokopop_play.cpp)
  target_include_directories(kokopop_play PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/playback
  )
  target_link_libraries(kokopop_play PRIVATE kokopop)

  # RT benchmark tool (synthesis speed measurement)
  add_executable(kokopop_rt tools/kokopop_rt.cpp)
  target_include_directories(kokopop_rt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model
    ${CMAKE_CURRENT_SOURCE_DIR}/src/streaming
    ${CMAKE_CURRENT_SOURCE_DIR}/src/synthesis
    ${CMAKE_CURRENT_SOURCE_DIR}/src/synthesis/chunker
    $<BUILD_INTERFACE:${ggml_SOURCE_DIR}/include>
  )
  target_link_libraries(kokopop_rt PRIVATE kokopop)
endif()

if(KOKOPOP_BUILD_TESTS)
  enable_testing()
  add_executable(kokopop_tests
    tests/test_main.cpp
  )
  target_include_directories(kokopop_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model
    ${CMAKE_CURRENT_SOURCE_DIR}/src/inference
    ${CMAKE_CURRENT_SOURCE_DIR}/src/synthesis
    ${CMAKE_CURRENT_SOURCE_DIR}/src/synthesis/chunker
    ${CMAKE_CURRENT_SOURCE_DIR}/src/audio
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend
    ${CMAKE_CURRENT_SOURCE_DIR}/src/streaming
    ${CMAKE_CURRENT_SOURCE_DIR}/src/playback
  )
  target_link_libraries(kokopop_tests PRIVATE kokopop doctest::doctest ggml)
  include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
  doctest_discover_tests(kokopop_tests)
endif()

option(KOKOPOP_BUILD_BENCH "Build kokopop benchmarks (requires kokoro.gguf)" OFF)
if(KOKOPOP_BUILD_BENCH)
  add_executable(kokopop_bench tests/bench_main.cpp)
  target_include_directories(kokopop_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model
    ${CMAKE_CURRENT_SOURCE_DIR}/src/inference
    ${CMAKE_CURRENT_SOURCE_DIR}/src/synthesis
    $<BUILD_INTERFACE:${ggml_SOURCE_DIR}/include>
  )
  target_link_libraries(kokopop_bench PRIVATE kokopop ggml)
endif()
