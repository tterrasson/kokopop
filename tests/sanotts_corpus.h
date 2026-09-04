#pragma once

// Loads tests/data/sanotts_g2p_corpus.json.
//
// The corpus is the reference side of the G2P parity gate: text, the phonemes
// the reference implementation produces, and the ids it frames them into. It
// also carries the token tables and the slice of the canonical-decomposition
// table those ids depend on, so the gate needs neither a converted GGUF nor a
// downloaded voice pack.
//
// Regenerate with tools/gen_sanotts_g2p_corpus.py.

#include "arch/sanotts/sano_tokenizer.h"

#include <yyjson.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace kokopop::test {

struct G2pEntry {
    std::string frontend;      // "misaki" | "piper"
    std::string voice;
    std::string espeak_voice;
    std::string text;
    std::string phonemes;
    std::vector<uint32_t> ids;
};

struct G2pCorpus {
    bool loaded = false;
    std::string load_error;

    std::string espeak_version;
    std::string espeak_source;
    std::string unicode_version;
    std::string sanotts_revision;

    std::vector<G2pEntry> entries;
    std::map<std::string, sano::TokenTable> token_tables;

    // Backing storage for the NFD views. The table itself holds raw pointers,
    // which is how it reads GGUF metadata in production; here they point at
    // these vectors instead.
    std::vector<uint32_t> nfd_codepoints;
    std::vector<uint32_t> nfd_offsets;
    std::vector<uint32_t> nfd_values;
    std::vector<uint32_t> nfd_ccc_codepoints;
    std::vector<uint32_t> nfd_ccc_classes;
    sano::NfdTable nfd;
};

namespace detail {

inline std::string corpus_path() {
    // KOKOPOP_TEST_DATA_DIR is the source tree's tests/data, injected by CMake,
    // so the suite passes from any build directory. The relative candidates are
    // the fallback for a build that did not define it.
    std::vector<std::string> candidates;
    const char * env = std::getenv("KOKOPOP_SANOTTS_CORPUS");
    if (env != nullptr && env[0] != '\0') {
        candidates.emplace_back(env);
    }
#ifdef KOKOPOP_TEST_DATA_DIR
    candidates.emplace_back(std::string(KOKOPOP_TEST_DATA_DIR) + "/sanotts_g2p_corpus.json");
#endif
    candidates.emplace_back("tests/data/sanotts_g2p_corpus.json");
    candidates.emplace_back("../tests/data/sanotts_g2p_corpus.json");

    for (const std::string & path : candidates) {
        std::ifstream in(path, std::ios::binary);
        if (in) {
            return path;
        }
    }
    return {};
}

inline bool read_u32_array(yyjson_val * object, const char * key,
                           std::vector<uint32_t> & out, std::string & error) {
    yyjson_val * array = yyjson_obj_get(object, key);
    if (array == nullptr || !yyjson_is_arr(array)) {
        error = std::string("corpus: missing or non-array ") + key;
        return false;
    }
    out.clear();
    out.reserve(yyjson_arr_size(array));
    size_t index = 0;
    size_t max = 0;
    yyjson_val * item = nullptr;
    yyjson_arr_foreach(array, index, max, item) {
        if (!yyjson_is_int(item)) {
            error = std::string("corpus: non-integer in ") + key;
            return false;
        }
        const int64_t value = yyjson_get_sint(item);
        if (value < 0) {
            error = std::string("corpus: negative value in ") + key;
            return false;
        }
        out.push_back(static_cast<uint32_t>(value));
    }
    return true;
}

inline std::string read_string(yyjson_val * object, const char * key) {
    yyjson_val * value = yyjson_obj_get(object, key);
    const char * text = yyjson_get_str(value);
    return text != nullptr ? std::string(text) : std::string();
}

inline bool parse_corpus(const std::string & path, G2pCorpus & corpus) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream buffer;
    buffer << in.rdbuf();
    const std::string json = buffer.str();

    yyjson_doc * doc = yyjson_read(json.data(), json.size(), 0);
    if (doc == nullptr) {
        corpus.load_error = "corpus: " + path + " is not valid JSON";
        return false;
    }
    const std::unique_ptr<yyjson_doc, void (*)(yyjson_doc *)> guard(doc, yyjson_doc_free);

    yyjson_val * root = yyjson_doc_get_root(doc);
    if (root == nullptr || !yyjson_is_obj(root)) {
        corpus.load_error = "corpus: root is not an object";
        return false;
    }

    corpus.espeak_version = read_string(root, "espeak_ng_version");
    corpus.espeak_source = read_string(root, "espeak_source");
    corpus.unicode_version = read_string(root, "unicode_version");
    corpus.sanotts_revision = read_string(root, "sanotts_revision");

    // -- the NFD subset ----------------------------------------------------
    yyjson_val * nfd = yyjson_obj_get(root, "nfd");
    if (nfd == nullptr || !yyjson_is_obj(nfd)) {
        corpus.load_error = "corpus: missing nfd object";
        return false;
    }
    if (!read_u32_array(nfd, "codepoints", corpus.nfd_codepoints, corpus.load_error) ||
        !read_u32_array(nfd, "offsets", corpus.nfd_offsets, corpus.load_error) ||
        !read_u32_array(nfd, "values", corpus.nfd_values, corpus.load_error) ||
        !read_u32_array(nfd, "ccc_codepoints", corpus.nfd_ccc_codepoints, corpus.load_error) ||
        !read_u32_array(nfd, "ccc_classes", corpus.nfd_ccc_classes, corpus.load_error)) {
        return false;
    }
    corpus.nfd.codepoints = corpus.nfd_codepoints.data();
    corpus.nfd.offsets = corpus.nfd_offsets.data();
    corpus.nfd.values = corpus.nfd_values.data();
    corpus.nfd.count = corpus.nfd_codepoints.size();
    corpus.nfd.n_offsets = corpus.nfd_offsets.size();
    corpus.nfd.n_values = corpus.nfd_values.size();
    corpus.nfd.ccc_codepoints = corpus.nfd_ccc_codepoints.data();
    corpus.nfd.ccc_classes = corpus.nfd_ccc_classes.data();
    corpus.nfd.ccc_count = corpus.nfd_ccc_codepoints.size();
    corpus.nfd.n_ccc_classes = corpus.nfd_ccc_classes.size();
    if (!corpus.nfd.validate(corpus.load_error)) {
        return false;
    }

    // -- token tables ------------------------------------------------------
    yyjson_val * tables = yyjson_obj_get(root, "token_tables");
    if (tables == nullptr || !yyjson_is_obj(tables)) {
        corpus.load_error = "corpus: missing token_tables object";
        return false;
    }
    size_t table_index = 0;
    size_t table_max = 0;
    yyjson_val * table_key = nullptr;
    yyjson_val * table_value = nullptr;
    yyjson_obj_foreach(tables, table_index, table_max, table_key, table_value) {
        const char * voice = yyjson_get_str(table_key);
        if (voice == nullptr || !yyjson_is_obj(table_value)) {
            corpus.load_error = "corpus: malformed token_tables entry";
            return false;
        }
        yyjson_val * symbols = yyjson_obj_get(table_value, "symbols");
        std::vector<uint32_t> ids;
        if (symbols == nullptr || !yyjson_is_arr(symbols) ||
            !read_u32_array(table_value, "ids", ids, corpus.load_error)) {
            corpus.load_error = std::string("corpus: token table ") + voice +
                                " is missing symbols or ids";
            return false;
        }
        if (yyjson_arr_size(symbols) != ids.size()) {
            corpus.load_error = std::string("corpus: token table ") + voice +
                                " has mismatched symbol and id counts";
            return false;
        }

        sano::TokenTable built;
        size_t index = 0;
        size_t max = 0;
        yyjson_val * item = nullptr;
        yyjson_arr_foreach(symbols, index, max, item) {
            const char * symbol = yyjson_get_str(item);
            if (symbol == nullptr) {
                corpus.load_error = "corpus: non-string symbol";
                return false;
            }
            built.to_id.emplace(symbol, ids[index]);
        }
        built.bos_id = static_cast<uint32_t>(
            yyjson_get_sint(yyjson_obj_get(table_value, "bos_id")));
        built.eos_id = static_cast<uint32_t>(
            yyjson_get_sint(yyjson_obj_get(table_value, "eos_id")));
        built.pad_id = static_cast<int32_t>(
            yyjson_get_sint(yyjson_obj_get(table_value, "pad_id")));
        built.fallback_id = static_cast<int32_t>(
            yyjson_get_sint(yyjson_obj_get(table_value, "fallback_id")));

        yyjson_val * specials = yyjson_obj_get(table_value, "special_symbols");
        if (specials != nullptr && yyjson_is_arr(specials)) {
            index = 0;
            max = 0;
            item = nullptr;
            yyjson_arr_foreach(specials, index, max, item) {
                const char * symbol = yyjson_get_str(item);
                if (symbol != nullptr) {
                    built.special_symbols.emplace(symbol);
                }
            }
        }
        if (!built.validate(corpus.load_error)) {
            corpus.load_error = std::string("corpus: token table ") + voice + ": " +
                                corpus.load_error;
            return false;
        }
        corpus.token_tables.emplace(voice, std::move(built));
    }

    // -- entries -----------------------------------------------------------
    yyjson_val * entries = yyjson_obj_get(root, "entries");
    if (entries == nullptr || !yyjson_is_arr(entries)) {
        corpus.load_error = "corpus: missing entries array";
        return false;
    }
    size_t index = 0;
    size_t max = 0;
    yyjson_val * item = nullptr;
    yyjson_arr_foreach(entries, index, max, item) {
        if (!yyjson_is_obj(item)) {
            corpus.load_error = "corpus: entry is not an object";
            return false;
        }
        G2pEntry entry;
        entry.frontend = read_string(item, "frontend");
        entry.voice = read_string(item, "voice");
        entry.espeak_voice = read_string(item, "espeak_voice");
        entry.text = read_string(item, "text");
        entry.phonemes = read_string(item, "phonemes");
        if (!read_u32_array(item, "ids", entry.ids, corpus.load_error)) {
            return false;
        }
        if (entry.frontend != "misaki" && entry.frontend != "piper") {
            corpus.load_error = "corpus: unknown frontend " + entry.frontend;
            return false;
        }
        corpus.entries.push_back(std::move(entry));
    }

    corpus.loaded = true;
    return true;
}

} // namespace detail

/// The corpus, parsed once per test run.
inline const G2pCorpus & sanotts_g2p_corpus() {
    static const G2pCorpus corpus = [] {
        G2pCorpus out;
        const std::string path = detail::corpus_path();
        if (path.empty()) {
            out.load_error = "tests/data/sanotts_g2p_corpus.json not found; run "
                             "tools/gen_sanotts_g2p_corpus.py";
            return out;
        }
        detail::parse_corpus(path, out);
        return out;
    }();
    return corpus;
}

inline const sano::NfdTable & sanotts_nfd_table() {
    return sanotts_g2p_corpus().nfd;
}

inline const sano::TokenTable * sanotts_token_table(const std::string & voice) {
    const G2pCorpus & corpus = sanotts_g2p_corpus();
    const auto it = corpus.token_tables.find(voice);
    return it == corpus.token_tables.end() ? nullptr : &it->second;
}

} // namespace kokopop::test
