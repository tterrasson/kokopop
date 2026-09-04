#pragma once

// Locating the upstream sanoTTS golden fixtures.
//
// `LICENSE.MIT` in Ampixa/sanoTTS covers the runtime sources, not
// `mcu/test/fixtures/**`, the weight blobs or the rendered PCM. So kokopop does
// not redistribute them: the tests read them from a local checkout and skip
// when there is none, which keeps `ctest` green on a fresh clone while still
// gating a developer who has the data.
//
// Point KOKOPOP_SANOTTS_FIXTURES at a directory containing the per-voice
// fixture directories (en_us_r227f32, en_us_e13b, ...). Without it, the default
// below is tried, matching a plain clone of the pinned upstream revision.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace kokopop::test {

inline std::string sanotts_fixture_root() {
    const char * env = std::getenv("KOKOPOP_SANOTTS_FIXTURES");
    if (env != nullptr && env[0] != '\0') {
        return env;
    }
    const char * home = std::getenv("HOME");
    if (home == nullptr || home[0] == '\0') {
        return {};
    }
    return std::string(home) + "/.cache/sanotts/upstream/mcu/test/fixtures";
}

inline std::string sanotts_fixture_hint() {
    return "sanoTTS golden fixtures not found; set KOKOPOP_SANOTTS_FIXTURES to a "
           "checkout of Ampixa/sanoTTS mcu/test/fixtures "
           "(pinned revision 939d982b9faa54cbcf5d24cc878f5cd514b2646e)";
}

/// Absolute path of `file` inside the `voice` fixture directory, or an empty
/// string when it is not readable.
inline std::string sanotts_fixture_path(const std::string & voice,
                                        const std::string & file) {
    const std::string root = sanotts_fixture_root();
    if (root.empty()) {
        return {};
    }
    const std::string path = root + "/" + voice + "/" + file;
    std::ifstream in(path, std::ios::binary);
    return in ? path : std::string();
}

/// Reads a whole file of little-endian float32 values.
inline bool read_f32_file(const std::string & path, std::vector<float> & out) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        return false;
    }
    const std::streamoff size = in.tellg();
    if (size < 0 || (size % 4) != 0) {
        return false;
    }
    in.seekg(0);
    out.resize(static_cast<size_t>(size) / 4);
    if (out.empty()) {
        return true;
    }
    in.read(reinterpret_cast<char *>(out.data()), size);
    return in.good() || in.eof();
}

/// Reads a whole file of little-endian int32 values.
inline bool read_i32_file(const std::string & path, std::vector<int32_t> & out) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        return false;
    }
    const std::streamoff size = in.tellg();
    if (size < 0 || (size % 4) != 0) {
        return false;
    }
    in.seekg(0);
    out.resize(static_cast<size_t>(size) / 4);
    if (out.empty()) {
        return true;
    }
    in.read(reinterpret_cast<char *>(out.data()), size);
    return in.good() || in.eof();
}

} // namespace kokopop::test
