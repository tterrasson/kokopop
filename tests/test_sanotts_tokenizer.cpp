#include "arch/sanotts/sano_tokenizer.h"
#include "sanotts_corpus.h"
#include "sanotts_fixtures.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace sano = kokopop::sano;

namespace {

/// A small hand-built Piper-style table: framing at 0/1/2, then a few symbols.
sano::TokenTable piper_table() {
    sano::TokenTable table;
    table.to_id = {
        {"_", 0}, {"^", 1}, {"$", 2}, {" ", 3}, {",", 4}, {".", 5},
        {"a", 10}, {"b", 11}, {"c", 12},
        {"\xc9\x99", 20},                     // U+0259 schwa
        {"\xcc\x81", 21},                     // U+0301 combining acute
        {"\xcc\xa7", 22},                     // U+0327 combining cedilla
        {"\xc3\xa7", 23},                     // U+00E7 c-with-cedilla, precomposed
    };
    table.bos_id = 1;
    table.eos_id = 2;
    table.pad_id = 0;
    table.fallback_id = 20;  // schwa
    table.special_symbols = {"_", "^", "$"};
    return table;
}

/// A misaki-style table: framing symbols are multi-code-point names, no PAD.
sano::TokenTable misaki_table() {
    sano::TokenTable table;
    table.to_id = {
        {"<pad>", 0}, {"<bos>", 1}, {"<eos>", 2},
        {" ", 3}, {".", 4},
        {"a", 10}, {"b", 11},
        {"\xc9\x99", 20},  // U+0259 schwa
        {"\xe2\x80\x94", 30},  // U+2014 em dash
    };
    table.bos_id = 1;
    table.eos_id = 2;
    table.pad_id = -1;
    table.fallback_id = -1;
    table.special_symbols = {"<pad>", "<bos>", "<eos>"};
    return table;
}

/// The NFD subset the corpus ships, which covers the cases used here.
const sano::NfdTable & nfd() {
    return kokopop::test::sanotts_nfd_table();
}

std::vector<uint32_t> ids_of(const std::string & phonemes,
                             const sano::TokenTable & table) {
    std::vector<uint32_t> ids;
    std::string error;
    REQUIRE_MESSAGE(sano::tokenize_piper(phonemes, table, nfd(), ids, error), error);
    return ids;
}

} // namespace

// ---------------------------------------------------------------------------
// Framing
// ---------------------------------------------------------------------------

TEST_CASE("sanotts_piper_framing_interleaves_pad") {
    const sano::TokenTable table = piper_table();
    // [BOS, PAD] + (id, PAD) per phoneme + [EOS].
    CHECK_EQ(ids_of("abc", table),
             std::vector<uint32_t>{1, 0, 10, 0, 11, 0, 12, 0, 2});
    CHECK_EQ(ids_of("a", table), std::vector<uint32_t>{1, 0, 10, 0, 2});
}

TEST_CASE("sanotts_misaki_framing_interleaves_nothing") {
    const sano::TokenTable table = misaki_table();
    std::vector<uint32_t> ids;
    std::string error;
    REQUIRE(sano::tokenize_misaki("ab", table, ids, error));
    CHECK_EQ(ids, std::vector<uint32_t>{1, 10, 11, 2});
}

// The Piper budget is counted in *final* ids, and PAD interleaving roughly
// doubles them: a limit read as "phonemes" would let through twice the tokens
// the duration model was trained for.
TEST_CASE("sanotts_piper_budget_counts_final_ids_not_phonemes") {
    sano::TokenTable table = piper_table();
    // "abc" is 3 phonemes but 9 final ids.
    table.max_tokens = 9;
    std::vector<uint32_t> ids;
    std::string error;
    CHECK(sano::tokenize_piper("abc", table, nfd(), ids, error));
    CHECK_EQ(ids.size(), 9u);

    table.max_tokens = 8;
    CHECK_FALSE(sano::tokenize_piper("abc", table, nfd(), ids, error));
    CHECK(error.find("9 token ids") != std::string::npos);
    CHECK(ids.empty());
}

TEST_CASE("sanotts_misaki_budget_counts_final_ids") {
    sano::TokenTable table = misaki_table();
    table.max_tokens = 4;  // BOS + 2 + EOS
    std::vector<uint32_t> ids;
    std::string error;
    CHECK(sano::tokenize_misaki("ab", table, ids, error));
    table.max_tokens = 3;
    CHECK_FALSE(sano::tokenize_misaki("ab", table, ids, error));
    CHECK(ids.empty());
}

TEST_CASE("sanotts_piper_needs_a_pad_id") {
    sano::TokenTable table = piper_table();
    table.pad_id = -1;
    std::vector<uint32_t> ids;
    std::string error;
    CHECK_FALSE(sano::tokenize_piper("a", table, nfd(), ids, error));
    CHECK(error.find("PAD") != std::string::npos);
}

// ---------------------------------------------------------------------------
// NFD
// ---------------------------------------------------------------------------

// The precomposed ç is in the table with its own id, but the reference
// decomposes first, so it must come out as c + combining cedilla — two ids, not
// one. Getting this wrong shifts every later id.
TEST_CASE("sanotts_piper_decomposes_before_mapping") {
    const sano::TokenTable table = piper_table();
    CHECK_EQ(ids_of("\xc3\xa7", table),
             std::vector<uint32_t>{1, 0, 12, 0, 22, 0, 2});
    // The already-decomposed form gives the same ids, which is the point of
    // normalising.
    CHECK_EQ(ids_of("c\xcc\xa7", table),
             std::vector<uint32_t>{1, 0, 12, 0, 22, 0, 2});
}

TEST_CASE("sanotts_misaki_does_not_decompose") {
    // The misaki path maps code points as they are: its vocabulary is
    // character-level over already-normalised phonemes.
    sano::TokenTable table = misaki_table();
    table.to_id.emplace("\xc3\xa7", 40);
    std::vector<uint32_t> ids;
    std::string error;
    REQUIRE(sano::tokenize_misaki("\xc3\xa7", table, ids, error));
    CHECK_EQ(ids, std::vector<uint32_t>{1, 40, 2});
}

TEST_CASE("sanotts_nfd_table_decomposes_and_orders") {
    const sano::NfdTable & table = nfd();
    REQUIRE(table.present());

    std::vector<uint32_t> out;
    // e-acute -> e + U+0301
    table.decompose(0x00E9, out);
    CHECK_EQ(out, std::vector<uint32_t>{0x0065, 0x0301});

    // A code point absent from the table decomposes to itself.
    out.clear();
    table.decompose(0x0062, out);
    CHECK_EQ(out, std::vector<uint32_t>{0x0062});

    // Combining classes come from the same table.
    CHECK_EQ(table.combining_class(0x0301), 230u);  // acute, above
    CHECK_EQ(table.combining_class(0x0327), 202u);  // cedilla, below
    CHECK_EQ(table.combining_class(0x0062), 0u);    // 'b' is a starter
}

// NFD is decomposition *and* canonical ordering. Two marks of different classes
// must come out in class order regardless of how they were written, or the same
// text yields two different id sequences.
TEST_CASE("sanotts_nfd_canonically_orders_combining_marks") {
    const sano::NfdTable & table = nfd();
    std::string error;

    // c + acute(230) + cedilla(202) must reorder to cedilla, then acute.
    std::vector<uint32_t> reordered;
    REQUIRE(table.normalize("c\xcc\x81\xcc\xa7", reordered, error));
    CHECK_EQ(reordered, std::vector<uint32_t>{0x0063, 0x0327, 0x0301});

    // Written the other way round, the result is identical.
    std::vector<uint32_t> already;
    REQUIRE(table.normalize("c\xcc\xa7\xcc\x81", already, error));
    CHECK_EQ(already, reordered);

    // The tokenizer therefore gives the same ids either way.
    const sano::TokenTable tokens = piper_table();
    CHECK_EQ(ids_of("c\xcc\x81\xcc\xa7", tokens), ids_of("c\xcc\xa7\xcc\x81", tokens));
}

TEST_CASE("sanotts_nfd_table_validates_its_structure") {
    std::string error;
    // A well-formed table.
    CHECK(nfd().validate(error));

    // Non-monotonic offsets would let decompose() read backwards.
    const uint32_t codepoints[] = {100, 200};
    const uint32_t offsets[] = {0, 2, 1};
    const uint32_t values[] = {1, 2};
    sano::NfdTable broken;
    broken.codepoints = codepoints;
    broken.offsets = offsets;
    broken.values = values;
    broken.count = 2;
    broken.n_offsets = 3;
    broken.n_values = 2;
    CHECK_FALSE(broken.validate(error));

    // Unsorted code points would break the binary search silently.
    const uint32_t unsorted[] = {200, 100};
    const uint32_t good_offsets[] = {0, 1, 2};
    sano::NfdTable out_of_order;
    out_of_order.codepoints = unsorted;
    out_of_order.offsets = good_offsets;
    out_of_order.values = values;
    out_of_order.count = 2;
    out_of_order.n_offsets = 3;
    out_of_order.n_values = 2;
    CHECK_FALSE(out_of_order.validate(error));

    // A truncated offsets array is the one defect validate() cannot infer from
    // the data, so the loader reports the length and validate() checks it —
    // otherwise reading offsets[count] is already out of bounds.
    sano::NfdTable truncated;
    truncated.codepoints = codepoints;
    truncated.offsets = good_offsets;
    truncated.values = values;
    truncated.count = 2;
    truncated.n_offsets = 2;
    truncated.n_values = 2;
    CHECK_FALSE(truncated.validate(error));

    // Same for the combining classes, which are indexed by ccc_codepoints.
    const uint32_t ccc_codepoints[] = {0x0301, 0x0327};
    const uint32_t ccc_classes[] = {230};
    sano::NfdTable short_ccc;
    short_ccc.codepoints = codepoints;
    short_ccc.offsets = good_offsets;
    short_ccc.values = values;
    short_ccc.count = 2;
    short_ccc.n_offsets = 3;
    short_ccc.n_values = 2;
    short_ccc.ccc_codepoints = ccc_codepoints;
    short_ccc.ccc_classes = ccc_classes;
    short_ccc.ccc_count = 2;
    short_ccc.n_ccc_classes = 1;
    CHECK_FALSE(short_ccc.validate(error));

    // An empty table is valid: it means nothing decomposes.
    sano::NfdTable empty;
    CHECK(empty.validate(error));
    CHECK_FALSE(empty.present());
}

TEST_CASE("sanotts_nfd_rejects_orphan_arrays_and_invalid_unicode") {
    std::string error;
    sano::NfdTable table;
    const uint32_t cp[] = {0xD800};
    const uint32_t offsets[] = {0, 1};
    const uint32_t values[] = {0x61};
    table.n_ccc_classes = 1;
    CHECK_FALSE(table.validate(error));
    table.n_ccc_classes = 0;
    table.n_offsets = 2;
    CHECK_FALSE(table.validate(error));
    table.codepoints = cp;
    table.offsets = offsets;
    table.values = values;
    table.count = table.n_values = 1;
    CHECK_FALSE(table.validate(error));
    table.codepoints = values;
    table.values = cp;
    CHECK_FALSE(table.validate(error));
}

TEST_CASE("sanotts_special_ids_must_exist_even_in_sparse_tables") {
    std::string error;
    auto table = piper_table();
    table.bos_id = 8;
    CHECK_FALSE(table.validate(error));
    table = piper_table();
    table.pad_id = 8;
    CHECK_FALSE(table.validate(error));
    table = piper_table();
    table.fallback_id = 8;
    CHECK_FALSE(table.validate(error));
    table = piper_table();
    table.pad_id = -2;
    CHECK_FALSE(table.validate(error));
}

// With no table, nothing decomposes — which is wrong for real input but must
// not crash or mis-map.
TEST_CASE("sanotts_piper_without_an_nfd_table_maps_code_points_directly") {
    const sano::TokenTable table = piper_table();
    sano::NfdTable empty;
    std::vector<uint32_t> ids;
    std::string error;
    REQUIRE(sano::tokenize_piper("\xc3\xa7", table, empty, ids, error));
    // The precomposed form keeps its own id instead of decomposing.
    CHECK_EQ(ids, std::vector<uint32_t>{1, 0, 23, 0, 2});
}

// ---------------------------------------------------------------------------
// Unknown symbols and framing symbols in the input
// ---------------------------------------------------------------------------

TEST_CASE("sanotts_unknown_symbols_are_dropped_not_mapped") {
    const sano::TokenTable piper = piper_table();
    // 'z' is not in the table: it disappears, and the surrounding ids keep
    // their positions.
    CHECK_EQ(ids_of("azb", piper),
             std::vector<uint32_t>{1, 0, 10, 0, 11, 0, 2});

    const sano::TokenTable misaki = misaki_table();
    std::vector<uint32_t> ids;
    std::string error;
    REQUIRE(sano::tokenize_misaki("azb", misaki, ids, error));
    CHECK_EQ(ids, std::vector<uint32_t>{1, 10, 11, 2});
}

TEST_CASE("sanotts_framing_symbols_in_the_input_cannot_forge_a_sentinel") {
    const sano::TokenTable table = piper_table();
    // "^" and "$" are the BOS/EOS symbols. Text containing them must not
    // inject extra sentinels mid-sequence.
    const std::vector<uint32_t> ids = ids_of("a^b$c_", table);
    CHECK_EQ(ids, std::vector<uint32_t>{1, 0, 10, 0, 11, 0, 12, 0, 2});
}

// The misaki-style tables name their sentinels with multi-code-point strings
// ("<bos>"), which the per-code-point loop can never match. The guarantee has
// to rest on the id, not the symbol.
TEST_CASE("sanotts_a_symbol_that_resolves_to_a_sentinel_id_is_dropped") {
    sano::TokenTable table = misaki_table();
    // A voice whose vocabulary happens to map a printable symbol onto the BOS
    // id: tokenizing text containing it must not emit a second BOS.
    table.to_id["|"] = table.bos_id;
    table.to_id["~"] = table.eos_id;

    std::vector<uint32_t> ids;
    std::string error;
    REQUIRE_MESSAGE(sano::tokenize_misaki("a|b~", table, ids, error), error);
    CHECK_EQ(ids, std::vector<uint32_t>{1, 10, 11, 2});

    // Same for the Piper framing, where PAD is a sentinel too.
    sano::TokenTable piper = piper_table();
    piper.special_symbols.clear();  // rely on the id check alone
    const std::vector<uint32_t> piper_ids = ids_of("a^b$c_", piper);
    CHECK_EQ(piper_ids, std::vector<uint32_t>{1, 0, 10, 0, 11, 0, 12, 0, 2});
}

TEST_CASE("sanotts_tokenizers_reject_input_with_nothing_mappable") {
    const sano::TokenTable piper = piper_table();
    std::vector<uint32_t> ids;
    std::string error;
    CHECK_FALSE(sano::tokenize_piper("zzz", piper, nfd(), ids, error));
    CHECK(error.find("no symbol") != std::string::npos);
    CHECK(ids.empty());

    const sano::TokenTable misaki = misaki_table();
    CHECK_FALSE(sano::tokenize_misaki("zzz", misaki, ids, error));
    CHECK(ids.empty());
}

TEST_CASE("sanotts_piper_right_strips_its_input") {
    const sano::TokenTable table = piper_table();
    // phonemizer emits a trailing separator space that Piper does not; keeping
    // it would add a (space, PAD) pair to every utterance.
    CHECK_EQ(ids_of("a  ", table), ids_of("a", table));
    // An interior space is a real phoneme and stays.
    CHECK_EQ(ids_of("a b", table),
             std::vector<uint32_t>{1, 0, 10, 0, 3, 0, 11, 0, 2});
}

TEST_CASE("sanotts_tokenizers_reject_invalid_utf8") {
    const sano::TokenTable table = piper_table();
    std::vector<uint32_t> ids;
    std::string error;
    const std::string truncated = std::string("a\xc9");  // lone lead byte
    CHECK_FALSE(sano::tokenize_piper(truncated, table, nfd(), ids, error));
    CHECK(error.find("UTF-8") != std::string::npos);

    const sano::TokenTable misaki = misaki_table();
    CHECK_FALSE(sano::tokenize_misaki(truncated, misaki, ids, error));
}

// ---------------------------------------------------------------------------
// Component vocabulary clamping — Piper only
// ---------------------------------------------------------------------------

TEST_CASE("sanotts_piper_clamps_out_of_vocab_ids_to_schwa") {
    const sano::TokenTable table = piper_table();
    // The shared code-point table reaches id 23; a component trained with 21
    // symbols cannot take those.
    const std::vector<uint32_t> ids{1, 0, 10, 0, 22, 0, 23, 0, 2};
    std::vector<uint32_t> clamped;
    std::string error;
    REQUIRE(sano::clamp_ids_to_vocab(ids, 21, table, "duration", clamped, error));
    CHECK_EQ(clamped, std::vector<uint32_t>{1, 0, 10, 0, 20, 0, 20, 0, 2});

    // In-range ids pass through untouched.
    REQUIRE(sano::clamp_ids_to_vocab(ids, 64, table, "duration", clamped, error));
    CHECK_EQ(clamped, ids);
}

// The vocos vocabulary has no schwa fallback: id 59 is the em dash there, so
// remapping to it would splice punctuation into the phoneme stream.
TEST_CASE("sanotts_misaki_refuses_to_clamp_instead_of_inventing_a_dash") {
    const sano::TokenTable table = misaki_table();
    CHECK_EQ(table.fallback_id, -1);
    const std::vector<uint32_t> ids{1, 30, 2};  // 30 is the em dash here
    std::vector<uint32_t> clamped;
    std::string error;
    CHECK_FALSE(sano::clamp_ids_to_vocab(ids, 21, table, "acoustic", clamped, error));
    CHECK(error.find("forbids a fallback") != std::string::npos);
    CHECK(clamped.empty());
}

TEST_CASE("sanotts_clamping_refuses_an_out_of_range_fallback") {
    sano::TokenTable table = piper_table();
    table.fallback_id = 20;
    std::vector<uint32_t> clamped;
    std::string error;
    // A vocabulary too small to hold the fallback itself.
    CHECK_FALSE(sano::clamp_ids_to_vocab({1, 0, 23, 2}, 20, table, "duration",
                                          clamped, error));
    CHECK(error.find("itself outside") != std::string::npos);

    CHECK_FALSE(sano::clamp_ids_to_vocab({1, 2}, 0, table, "duration", clamped, error));
    CHECK(error.find("zero vocabulary") != std::string::npos);
}

// ---------------------------------------------------------------------------
// Table validation
// ---------------------------------------------------------------------------

TEST_CASE("sanotts_token_table_validation_catches_broken_framing") {
    std::string error;
    CHECK(piper_table().validate(error));
    CHECK(misaki_table().validate(error));

    sano::TokenTable empty;
    CHECK_FALSE(empty.validate(error));

    sano::TokenTable same = piper_table();
    same.eos_id = same.bos_id;
    CHECK_FALSE(same.validate(error));

    sano::TokenTable pad_collides = piper_table();
    pad_collides.pad_id = static_cast<int32_t>(pad_collides.bos_id);
    CHECK_FALSE(pad_collides.validate(error));

    sano::TokenTable out_of_range = piper_table();
    out_of_range.bos_id = 9999;
    CHECK_FALSE(out_of_range.validate(error));

    sano::TokenTable tiny_budget = piper_table();
    tiny_budget.max_tokens = 2;
    CHECK_FALSE(tiny_budget.validate(error));
}

TEST_CASE("sanotts_corpus_token_tables_are_the_real_ones") {
    const sano::TokenTable * heart = kokopop::test::sanotts_token_table("heart");
    const sano::TokenTable * amy = kokopop::test::sanotts_token_table("amy");
    REQUIRE(heart != nullptr);
    REQUIRE(amy != nullptr);

    // The 62-symbol vocos vocabulary, framed without PAD and with no fallback.
    CHECK_EQ(heart->to_id.size(), 62u);
    CHECK_EQ(heart->pad_id, -1);
    CHECK_EQ(heart->fallback_id, -1);

    // Piper's shared code-point table, with PAD at 0 and schwa as the fallback.
    CHECK(amy->to_id.size() > 140u);
    CHECK_EQ(amy->pad_id, 0);
    CHECK_EQ(amy->bos_id, 1u);
    CHECK_EQ(amy->eos_id, 2u);
    const uint32_t * schwa = amy->find("\xc9\x99");
    REQUIRE(schwa != nullptr);
    CHECK_EQ(amy->fallback_id, static_cast<int32_t>(*schwa));
}

// ---------------------------------------------------------------------------
// The golden fixtures
//
// r*_ids.bin is the id sequence upstream fed its reference stack. Its *text* is
// not shipped, so what is gated here is the tokenizer: decoding the ids back to
// phonemes and re-tokenizing them has to return the same sequence. That pins
// the framing, the vocabulary and the drop rules against real data.
// ---------------------------------------------------------------------------

TEST_CASE("sanotts_misaki_tokenizer_reproduces_the_golden_ids") {
    const sano::TokenTable * table = kokopop::test::sanotts_token_table("heart");
    REQUIRE(table != nullptr);

    // Invert the table so the fixture ids can be turned back into phonemes.
    std::vector<std::string> by_id(64);
    for (const auto & entry : table->to_id) {
        REQUIRE(entry.second < by_id.size());
        by_id[entry.second] = entry.first;
    }

    size_t rows = 0;
    for (int row = 0; row < 8; ++row) {
        char name[32];
        std::snprintf(name, sizeof(name), "r%02d_ids.bin", row);
        const std::string path =
            kokopop::test::sanotts_fixture_path("en_us_r227f32", name);
        if (path.empty()) {
            continue;
        }
        std::vector<int32_t> ids;
        REQUIRE(kokopop::test::read_i32_file(path, ids));
        REQUIRE(ids.size() > 8u);

        // The framing the vocos contract uses.
        CHECK_EQ(ids.front(), static_cast<int32_t>(table->bos_id));
        CHECK_EQ(ids.back(), static_cast<int32_t>(table->eos_id));

        std::string phonemes;
        for (size_t i = 1; i + 1 < ids.size(); ++i) {
            const int32_t id = ids[i];
            REQUIRE(id >= 0);
            REQUIRE(static_cast<size_t>(id) < by_id.size());
            const std::string & symbol = by_id[static_cast<size_t>(id)];
            REQUIRE_FALSE(symbol.empty());
            // No framing symbol may appear inside the body.
            CHECK_FALSE(table->is_special(symbol));
            phonemes += symbol;
        }

        std::vector<uint32_t> round_trip;
        std::string error;
        REQUIRE_MESSAGE(sano::tokenize_misaki(phonemes, *table, round_trip, error), error);

        std::vector<uint32_t> expected(ids.begin(), ids.end());
        CHECK_EQ(round_trip, expected);
        ++rows;
    }
    if (rows == 0) {
        MESSAGE("skipping: " << kokopop::test::sanotts_fixture_hint());
    } else {
        CHECK_EQ(rows, 8u);
    }
}

// The two fixture families share the vocabulary, so the same round trip must
// hold for the int8 voice's rows.
TEST_CASE("sanotts_misaki_tokenizer_reproduces_the_heartnano_golden_ids") {
    const sano::TokenTable * table = kokopop::test::sanotts_token_table("heartnano");
    REQUIRE(table != nullptr);

    std::vector<std::string> by_id(64);
    for (const auto & entry : table->to_id) {
        by_id[entry.second] = entry.first;
    }

    size_t rows = 0;
    for (int row = 0; row < 8; ++row) {
        char name[32];
        std::snprintf(name, sizeof(name), "r%02d_ids.bin", row);
        const std::string path = kokopop::test::sanotts_fixture_path("en_us_e13b", name);
        if (path.empty()) {
            continue;
        }
        std::vector<int32_t> ids;
        REQUIRE(kokopop::test::read_i32_file(path, ids));

        std::string phonemes;
        for (size_t i = 1; i + 1 < ids.size(); ++i) {
            phonemes += by_id[static_cast<size_t>(ids[i])];
        }
        std::vector<uint32_t> round_trip;
        std::string error;
        REQUIRE_MESSAGE(sano::tokenize_misaki(phonemes, *table, round_trip, error), error);
        CHECK_EQ(round_trip, std::vector<uint32_t>(ids.begin(), ids.end()));
        ++rows;
    }
    if (rows == 0) {
        MESSAGE("skipping: " << kokopop::test::sanotts_fixture_hint());
    } else {
        CHECK_EQ(rows, 8u);
    }
}
