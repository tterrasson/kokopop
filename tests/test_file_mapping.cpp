#include "test_helpers.h"
#include "core/file_mapping.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <utility>

namespace {

// Writes `bytes` to a temp path; returns the path. The file is left on disk
// for the test to read; caller removes it via std::remove().
std::string write_temp_file(const std::vector<uint8_t> & bytes, const std::string & tag) {
    const std::string path = "kokopop_file_mapping_" + tag + ".bin";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char *>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    out.close();
    return path;
}

} // namespace

// ---- FileMapping basic open & content integrity ----

TEST_CASE("file_mapping_opens_existing_file_and_reads_bytes") {
    std::vector<uint8_t> contents(8192);
    std::mt19937 rng(0xC0FFEE);
    for (auto & b : contents) b = static_cast<uint8_t>(rng() & 0xff);
    const std::string path = write_temp_file(contents, "basic");

    kokopop::FileMapping fm(path);
    CHECK(fm.ok());
    CHECK_EQ(fm.size(), contents.size());
    REQUIRE(fm.data() != nullptr);
    // Bit-exact compare: file → mapping must not corrupt a single byte.
    CHECK_EQ(std::memcmp(fm.data(), contents.data(), contents.size()), 0);

    std::remove(path.c_str());
}

TEST_CASE("file_mapping_fails_gracefully_on_missing_file") {
    kokopop::FileMapping fm("kokopop_file_mapping_does_not_exist.bin");
    CHECK_FALSE(fm.ok());
    CHECK(fm.data() == nullptr);
    CHECK_EQ(fm.size(), 0u);
    CHECK_FALSE(fm.error().empty());
}

TEST_CASE("file_mapping_zero_byte_file_is_not_ok") {
    // Both POSIX mmap and Win32 CreateFileMapping refuse zero-length files;
    // FileMapping must report this as an error rather than returning an
    // empty mapping (the GGUF loader needs the error path to fall back to
    // fread, not silently produce no tensor data).
    const std::string path = write_temp_file({}, "empty");
    kokopop::FileMapping fm(path);
    CHECK_FALSE(fm.ok());
    CHECK_FALSE(fm.error().empty());
    std::remove(path.c_str());
}

// ---- Move semantics ----

TEST_CASE("file_mapping_move_transfers_ownership") {
    std::vector<uint8_t> contents = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::string path = write_temp_file(contents, "move");

    kokopop::FileMapping a(path);
    REQUIRE(a.ok());
    const uint8_t * a_ptr = a.data();
    const size_t a_size   = a.size();

    kokopop::FileMapping b(std::move(a));
    CHECK(b.ok());
    CHECK_EQ(b.data(), a_ptr);
    CHECK_EQ(b.size(), a_size);
    // Moved-from is empty / non-ok; double-close on destruction must be safe.
    CHECK_FALSE(a.ok());
    CHECK(a.data() == nullptr);

    // Move-assign onto an already-mapped target frees the previous mapping.
    kokopop::FileMapping c(path);
    REQUIRE(c.ok());
    c = std::move(b);
    CHECK(c.ok());
    CHECK_FALSE(b.ok());

    std::remove(path.c_str());
}

TEST_CASE("file_mapping_advise_random_is_safe_on_unmapped") {
    kokopop::FileMapping fm;
    CHECK_FALSE(fm.ok());
    // Must not crash even though no mapping is active.
    fm.advise_random();
}

TEST_CASE("file_mapping_advise_random_is_safe_on_mapped") {
    std::vector<uint8_t> contents(4096, 0xAB);
    const std::string path = write_temp_file(contents, "advise");
    kokopop::FileMapping fm(path);
    REQUIRE(fm.ok());
    fm.advise_random();
    // Mapping must still be readable after switching the access hint.
    CHECK_EQ(fm.data()[0],     0xAB);
    CHECK_EQ(fm.data()[2048],  0xAB);
    CHECK_EQ(fm.data()[4095],  0xAB);
    std::remove(path.c_str());
}
