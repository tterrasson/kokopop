#pragma once

// Small typed readers over a `gguf_context`.
//
// A GGUF file is untrusted input: every accessor here returns false when the
// key is absent or has the wrong type so that callers decide between a default
// and a hard error, without triggering a ggml assertion.

#include <cstdint>
#include <string>
#include <vector>

#include <gguf.h>

namespace kokopop {

inline bool gguf_get_u32(gguf_context * ctx, const char * key, uint32_t & out) {
    const int idx = gguf_find_key(ctx, key);
    if (idx < 0 || gguf_get_kv_type(ctx, idx) != GGUF_TYPE_UINT32) {
        return false;
    }
    out = gguf_get_val_u32(ctx, idx);
    return true;
}

inline bool gguf_get_i32(gguf_context * ctx, const char * key, int32_t & out) {
    const int idx = gguf_find_key(ctx, key);
    if (idx < 0 || gguf_get_kv_type(ctx, idx) != GGUF_TYPE_INT32) {
        return false;
    }
    out = gguf_get_val_i32(ctx, idx);
    return true;
}

inline bool gguf_get_bool(gguf_context * ctx, const char * key, bool & out) {
    const int idx = gguf_find_key(ctx, key);
    if (idx < 0 || gguf_get_kv_type(ctx, idx) != GGUF_TYPE_BOOL) {
        return false;
    }
    out = gguf_get_val_bool(ctx, idx);
    return true;
}

inline bool gguf_get_f32(gguf_context * ctx, const char * key, float & out) {
    const int idx = gguf_find_key(ctx, key);
    if (idx < 0 || gguf_get_kv_type(ctx, idx) != GGUF_TYPE_FLOAT32) {
        return false;
    }
    out = gguf_get_val_f32(ctx, idx);
    return true;
}

inline bool gguf_get_str(gguf_context * ctx, const char * key, std::string & out) {
    const int idx = gguf_find_key(ctx, key);
    if (idx < 0 || gguf_get_kv_type(ctx, idx) != GGUF_TYPE_STRING) {
        return false;
    }
    const char * s = gguf_get_val_str(ctx, idx);
    out = s != nullptr ? s : "";
    return true;
}

inline bool gguf_get_str_array(gguf_context * ctx, const char * key,
                               std::vector<std::string> & out) {
    const int idx = gguf_find_key(ctx, key);
    if (idx < 0 || gguf_get_kv_type(ctx, idx) != GGUF_TYPE_ARRAY) {
        return false;
    }
    if (gguf_get_arr_type(ctx, idx) != GGUF_TYPE_STRING) {
        return false;
    }
    const int64_t n = gguf_get_arr_n(ctx, idx);
    out.clear();
    out.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const char * s = gguf_get_arr_str(ctx, idx, i);
        out.emplace_back(s != nullptr ? s : "");
    }
    return true;
}

inline bool gguf_get_u32_array(gguf_context * ctx, const char * key,
                               std::vector<uint32_t> & out) {
    const int idx = gguf_find_key(ctx, key);
    if (idx < 0 || gguf_get_kv_type(ctx, idx) != GGUF_TYPE_ARRAY) {
        return false;
    }
    if (gguf_get_arr_type(ctx, idx) != GGUF_TYPE_UINT32) {
        return false;
    }
    const int64_t n = gguf_get_arr_n(ctx, idx);
    const uint32_t * data = static_cast<const uint32_t *>(gguf_get_arr_data(ctx, idx));
    out.clear();
    if (n != 0) out.assign(data, data + n);
    return true;
}

} // namespace kokopop
