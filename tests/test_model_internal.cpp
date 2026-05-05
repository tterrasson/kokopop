#include "test_helpers.h"
#include "model/model.h"

// ---- ScratchArena ----

TEST_CASE("scratch_arena_first_call") {
    kokopop::ScratchArena arena;
    CHECK_EQ(arena.capacity(), 0u);

    auto* ptr = arena.data(64);
    CHECK(ptr != nullptr);
    CHECK_EQ(arena.capacity(), 64u);
    CHECK_EQ(arena.high_water, 64u);

    ptr[0] = 0xAB;
    ptr[63] = 0xCD;
}

TEST_CASE("scratch_arena_growth") {
    kokopop::ScratchArena arena;

    auto* ptr1 = arena.data(100);
    CHECK_EQ(arena.capacity(), 100u);

    // Write a pattern
    for (int i = 0; i < 100; ++i) {
        ptr1[i] = static_cast<uint8_t>(i);
    }

    auto* ptr2 = arena.data(200);
    CHECK_EQ(arena.capacity(), 200u);
    // high_water tracks the largest data() call
    CHECK_EQ(arena.high_water, 200u);
    // Data should be preserved (vector preserves existing elements on resize)
    for (int i = 0; i < 100; ++i) {
        CHECK_EQ(ptr2[i], static_cast<uint8_t>(i));
    }
}

TEST_CASE("scratch_arena_high_water_tracking") {
    kokopop::ScratchArena arena;

    arena.data(1000);
    CHECK_EQ(arena.high_water, 1000u);

    // Calling with smaller size doesn't change high_water
    arena.data(50);
    CHECK_EQ(arena.high_water, 1000u);
}
