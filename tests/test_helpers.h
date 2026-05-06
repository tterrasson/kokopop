#pragma once

#include "kokopop.h"
#include "model/model.h"
#include "inference/kokoro.h"
#include "synthesis/phonemizer.h"
#include "core/utf8.h"
#include "core/wav.h"
#include "synthesis/synth.h"
#include "backend/backend.h"

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

inline void put_u32(std::vector<uint8_t> & out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v));
    out.push_back(static_cast<uint8_t>(v >> 8u));
    out.push_back(static_cast<uint8_t>(v >> 16u));
    out.push_back(static_cast<uint8_t>(v >> 24u));
}

inline void put_u64(std::vector<uint8_t> & out, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<uint8_t>(v >> (8u * i)));
    }
}

inline void put_f32(std::vector<uint8_t> & out, float v) {
    uint32_t raw = 0;
    static_assert(sizeof(raw) == sizeof(v), "float size");
    std::memcpy(&raw, &v, sizeof(v));
    put_u32(out, raw);
}

inline void put_string(std::vector<uint8_t> & out, const std::string & s) {
    put_u64(out, s.size());
    out.insert(out.end(), s.begin(), s.end());
}

inline void put_kv_u32(std::vector<uint8_t> & out, const std::string & key, uint32_t value) {
    put_string(out, key);
    put_u32(out, 4);
    put_u32(out, value);
}

inline void put_kv_bool(std::vector<uint8_t> & out, const std::string & key, bool value) {
    put_string(out, key);
    put_u32(out, 7);
    out.push_back(value ? 1 : 0);
}

inline void put_kv_str_array(std::vector<uint8_t> & out, const std::string & key, const std::vector<std::string> & values) {
    put_string(out, key);
    put_u32(out, 9);
    put_u32(out, 8);
    put_u64(out, values.size());
    for (const auto & value : values) {
        put_string(out, value);
    }
}

inline void align_to(std::vector<uint8_t> & out, size_t alignment) {
    while (out.size() % alignment != 0) {
        out.push_back(0);
    }
}

inline std::string write_mock_gguf() {
    const std::string path = "kokopop_mock_test.gguf";
    std::vector<uint8_t> bytes;
    bytes.insert(bytes.end(), {'G', 'G', 'U', 'F'});
    put_u32(bytes, 3);
    put_u64(bytes, 1);
    put_u64(bytes, 6);

    put_kv_u32(bytes, "general.alignment", 32);
    put_kv_u32(bytes, "kokopop.kokoro.version", 4);
    put_kv_bool(bytes, "kokopop.mock", true);
    put_kv_u32(bytes, "kokopop.sample_rate", 24000);
    put_kv_str_array(bytes, "tokenizer.ggml.tokens", {"", "a", "b", "c", " ", "ɑ", "ɔ", "ʃ"});
    put_kv_str_array(bytes, "kokopop.voices", {"af_heart"});

    put_string(bytes, "kokopop.voice.af_heart");
    put_u32(bytes, 2);
    put_u64(bytes, 4);
    put_u64(bytes, 2);
    put_u32(bytes, 0);
    put_u64(bytes, 0);

    align_to(bytes, 32);
    for (int i = 0; i < 8; ++i) {
        put_f32(bytes, static_cast<float>(i) / 8.0f);
    }

    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return path;
}

inline std::string real_model_path() {
    const char * candidates[] = {
        "models/kokoro-md.gguf",
        "../models/kokoro-md.gguf",
    };
    for (const char * path : candidates) {
        std::ifstream in(path, std::ios::binary);
        if (in) {
            return path;
        }
    }
    return "";
}

inline bool real_model_available() {
    return !real_model_path().empty();
}

struct Stats {
    double mean = 0.0;
    double rms = 0.0;
    float peak = 0.0f;
};

inline Stats stats(const std::vector<float> & values) {
    Stats out;
    if (values.empty()) {
        return out;
    }
    double sum = 0.0;
    double ss = 0.0;
    for (float v : values) {
        CHECK(std::isfinite(v));
        sum += v;
        ss += static_cast<double>(v) * static_cast<double>(v);
        out.peak = std::max(out.peak, std::fabs(v));
    }
    out.mean = sum / static_cast<double>(values.size());
    out.rms = std::sqrt(ss / static_cast<double>(values.size()));
    return out;
}

inline void load_real_model(std::unique_ptr<kokopop::Model> & model) {
    kokopop_model_options options{};
    options.n_threads = 1;
    options.backend = KOKOPOP_BACKEND_CPU;
    std::string error;
    CHECK(kokopop::load_model_from_gguf(real_model_path(), &options, model, error));
    CHECK(model != nullptr);
}

// Returns a lazily-loaded real model shared across all test cases in a session.
// Loading happens once on first call; returns nullptr if the model file is absent.
inline kokopop::Model * shared_real_model() {
    static std::unique_ptr<kokopop::Model> s_model;
    static bool s_tried = false;
    if (!s_tried) {
        s_tried = true;
        if (real_model_available()) {
            load_real_model(s_model);
        }
    }
    return s_model.get();
}

// Returns a path to a mock GGUF written once per session.
inline const std::string & shared_mock_gguf() {
    static const std::string path = write_mock_gguf();
    return path;
}
