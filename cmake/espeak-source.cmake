# Shared source and API backport for WebAssembly and the native fallback.
FetchContent_Declare(espeak
  GIT_REPOSITORY https://github.com/espeak-ng/espeak-ng.git
  GIT_TAG 4870adfa25b1a32b4361592f1be8a40337c58d6c
  SOURCE_SUBDIR kokopop-populate-only)
FetchContent_MakeAvailable(espeak)
include(${CMAKE_CURRENT_LIST_DIR}/patch-espeak-terminator.cmake)
kokopop_patch_espeak_terminator("${espeak_SOURCE_DIR}")
