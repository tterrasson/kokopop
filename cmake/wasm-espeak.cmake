# Build the library for wasm and its language data with a host executable.
# Both builds deliberately use exactly the same pinned eSpeak version.
FetchContent_Declare(espeak
  GIT_REPOSITORY https://github.com/espeak-ng/espeak-ng.git
  GIT_TAG 4870adfa25b1a32b4361592f1be8a40337c58d6c
  SOURCE_SUBDIR kokopop-populate-only)
FetchContent_MakeAvailable(espeak)
foreach(option COMPILE_INTONATIONS ENABLE_TESTS USE_ASYNC USE_MBROLA
    USE_LIBSONIC USE_LIBPCAUDIO USE_SPEECHPLAYER)
  set(${option} OFF CACHE BOOL "" FORCE)
endforeach()
add_subdirectory(${espeak_SOURCE_DIR} ${espeak_BINARY_DIR} EXCLUDE_FROM_ALL)
set(KOKOPOP_ESPEAK_LIBRARIES espeak-ng)
set(KOKOPOP_ESPEAK_INCLUDE_DIRS ${espeak_SOURCE_DIR}/src/include)

include(ExternalProject)
find_program(KOKOPOP_HOST_CC NAMES clang gcc cc REQUIRED NO_CMAKE_FIND_ROOT_PATH)
find_program(KOKOPOP_HOST_CXX NAMES clang++ g++ c++ REQUIRED NO_CMAKE_FIND_ROOT_PATH)
set(KOKOPOP_ESPEAK_DATA_DIR ${CMAKE_BINARY_DIR}/espeak-native/espeak-ng-data)
ExternalProject_Add(kokopop_espeak_data
  SOURCE_DIR ${espeak_SOURCE_DIR}
  BINARY_DIR ${CMAKE_BINARY_DIR}/espeak-native
  CMAKE_ARGS
    -DCMAKE_TOOLCHAIN_FILE= -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_C_COMPILER=${KOKOPOP_HOST_CC} -DCMAKE_CXX_COMPILER=${KOKOPOP_HOST_CXX}
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    -DBUILD_SHARED_LIBS=OFF -DENABLE_TESTS=OFF -DUSE_ASYNC=OFF
    -DUSE_MBROLA=OFF -DUSE_LIBSONIC=OFF -DUSE_LIBPCAUDIO=OFF -DUSE_SPEECHPLAYER=OFF
  BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target data --parallel 4
  INSTALL_COMMAND "")

# musl declares isw* in wchar.h as well as wctype.h. Read those declarations
# before eSpeak's wctype shim maps the names to its Unicode database functions.
target_compile_options(espeak-ng PRIVATE -include wchar.h)
# eSpeak exposes its libc compatibility shims through espeak-include. They
# must stay private to eSpeak, not shadow the C++ standard library in kokopop.
get_target_property(_espeak_interface espeak-ng INTERFACE_LINK_LIBRARIES)
list(REMOVE_ITEM _espeak_interface espeak-include)
set_property(TARGET espeak-ng PROPERTY INTERFACE_LINK_LIBRARIES "${_espeak_interface}")
