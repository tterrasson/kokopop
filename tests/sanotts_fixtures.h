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
#include "arch/sanotts/sano_arch.h"
#include "model/model.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
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

/// Root of the piperlite reference fixtures, which are generated locally by
/// `tools/gen_sanotts_fixtures.py` rather than shipped: upstream has no golden
/// PCM for that family, and the voice packs the generator reads are not
/// redistributable either.
inline std::string sanotts_piperlite_fixture_root() {
    const char * env = std::getenv("KOKOPOP_SANOTTS_PIPERLITE_FIXTURES");
    if (env != nullptr && env[0] != '\0') {
        return env;
    }
    const char * home = std::getenv("HOME");
    if (home == nullptr || home[0] == '\0') {
        return {};
    }
    return std::string(home) + "/.cache/sanotts/piperlite-fixtures";
}

inline std::string sanotts_piperlite_fixture_hint() {
    return "piperlite reference fixtures not found; generate them with "
           "`uv run tools/gen_sanotts_fixtures.py`, or point "
           "KOKOPOP_SANOTTS_PIPERLITE_FIXTURES at an existing set";
}

inline std::string sanotts_piperlite_fixture_path(const std::string & voice,
                                                  const std::string & file) {
    const std::string root = sanotts_piperlite_fixture_root();
    if (root.empty()) {
        return {};
    }
    const std::string path = root + "/" + voice + "/" + file;
    std::ifstream in(path, std::ios::binary);
    return in ? path : std::string();
}

/// Path of a converted sanoTTS GGUF, or an empty string when it is absent.
///
/// These are produced by `tools/convert_sanotts_to_gguf.py` from voice packs
/// kokopop does not redistribute, so every test that needs one skips without.
inline std::string sanotts_model_path(const std::string & name) {
    const std::string candidates[] = {
        "models/sanotts-" + name + ".gguf",
        "../models/sanotts-" + name + ".gguf",
    };
    for (const std::string & path : candidates) {
        std::ifstream in(path, std::ios::binary);
        if (in) {
            return path;
        }
    }
    return {};
}

inline std::string sanotts_model_hint(const std::string & name) {
    return "models/sanotts-" + name + ".gguf not found; build it with "
           "`uv run tools/convert_sanotts_to_gguf.py`";
}

/// A loaded sanoTTS GGUF plus its arch, or a reason it could not be loaded.
///
/// Every sanoTTS test needs the same three lines, and every one of them must
/// skip rather than fail when the converted model is missing from a fresh
/// clone.
struct SanoModel {
    std::unique_ptr<kokopop::Model> model;
    kokopop::SanoArch * arch = nullptr;

    bool load(const std::string & name, std::string & why,
              int backend = KOKOPOP_BACKEND_CPU) {
        const std::string path = sanotts_model_path(name);
        if (path.empty()) {
            why = sanotts_model_hint(name);
            return false;
        }
        kokopop_model_options options{};
        options.n_threads = 1;
        options.backend = backend;
        std::string error;
        const bool ok = kokopop::load_model_from_gguf(path, &options, model, error);
        // Missing optional data may skip. A present CPU model failing to load
        // is a regression, and must not turn a broken decoder test green.
        if (backend == KOKOPOP_BACKEND_CPU) {
            REQUIRE_MESSAGE(ok, path << ": " << error);
        }
        if (!ok) {
            why = path + ": " + error;
            return false;
        }
        arch = kokopop::sano_arch(*model);
        if (arch == nullptr) {
            why = path + ": loaded but not a sanoTTS model";
            return false;
        }
        return true;
    }
};

/// Correlation and error metrics between a rendering and its reference,
/// compared sample for sample after length alignment only. No time shift is
/// searched for: a decoder that needs one is not passing.
struct AudioComparison {
    double correlation = 0.0;
    double rms_error = 0.0;
    double rms_ratio = 0.0;
    size_t compared = 0;
};

inline AudioComparison compare_audio(const std::vector<float> & got,
                                     const std::vector<float> & expected) {
    AudioComparison out;
    out.compared = std::min(got.size(), expected.size());
    if (out.compared == 0) {
        return out;
    }
    double energy_got = 0.0;
    double energy_expected = 0.0;
    double cross = 0.0;
    double squared_error = 0.0;
    for (size_t i = 0; i < out.compared; ++i) {
        const double a = got[i];
        const double b = expected[i];
        energy_got += a * a;
        energy_expected += b * b;
        cross += a * b;
        squared_error += (a - b) * (a - b);
    }
    const double denominator = std::sqrt(energy_got * energy_expected);
    out.correlation = denominator > 0.0 ? cross / denominator : 0.0;
    out.rms_error = std::sqrt(squared_error / static_cast<double>(out.compared));
    out.rms_ratio = energy_expected > 0.0
        ? std::sqrt(energy_got / energy_expected) : 0.0;
    return out;
}

} // namespace kokopop::test
