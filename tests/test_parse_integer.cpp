#include "core/parse_integer.h"

TEST_CASE("unsigned_seed_parser_preserves_all_64_bits_and_rejects_partial_values") {
    uint64_t seed = 7;
    for (const char * text : {"", "-1", "+1", " 1", "1 ", "1.5", "1x", "18446744073709551616"}) {
        CHECK_FALSE(kokopop::parse_u64(text, seed));
        CHECK_EQ(seed, 7u);
    }
    CHECK(kokopop::parse_u64("18446744073709551615", seed));
    CHECK_EQ(seed, UINT64_MAX);
    CHECK(kokopop::parse_u64("0", seed));
    CHECK_EQ(seed, 0u);
}
