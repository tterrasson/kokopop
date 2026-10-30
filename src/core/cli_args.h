#pragma once

// Shared command-line parsing helpers for the kokopop CLI tools.
// Every helper validates its input: a missing or malformed value exits with
// code 2 and a message instead of dereferencing nullptr or throwing.

#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cmath>

namespace kokopop::cli {

/// Returns argv[++i], or nullptr when the option is the last argument.
inline const char * arg_value(int & i, int argc, char ** argv) {
    if (i + 1 >= argc) return nullptr;
    ++i;
    return argv[i];
}

/// Like arg_value(), but exits with an error when the value is missing.
[[noreturn]] inline void arg_missing(const char * option, const char * argv0) {
    std::fprintf(stderr, "%s: option %s requires a value\n", argv0, option);
    std::exit(2);
}

inline const char * arg_required(int & i, int argc, char ** argv,
                                 const char * option, const char * argv0) {
    const char * v = arg_value(i, argc, argv);
    if (v == nullptr) arg_missing(option, argv0);
    return v;
}

/// Parses an integer option value in [lo, hi]; exits with an error otherwise.
inline long arg_int(int & i, int argc, char ** argv,
                    const char * option, const char * argv0,
                    long lo, long hi) {
    const char * v = arg_required(i, argc, argv, option, argv0);
    char * end = nullptr;
    errno = 0;
    const long value = std::strtol(v, &end, 10);
    if (errno == ERANGE || end == v || *end != '\0' || value < lo || value > hi) {
        std::fprintf(stderr, "%s: option %s expects an integer in [%ld, %ld], got '%s'\n",
                     argv0, option, lo, hi, v);
        std::exit(2);
    }
    return value;
}

/// Parses a floating-point option value in [lo, hi]; exits on error.
inline double arg_double(int & i, int argc, char ** argv,
                         const char * option, const char * argv0,
                         double lo, double hi) {
    const char * v = arg_required(i, argc, argv, option, argv0);
    char * end = nullptr;
    errno = 0;
    const double value = std::strtod(v, &end);
    if (errno == ERANGE || !std::isfinite(value) || end == v || *end != '\0' || value < lo || value > hi) {
        std::fprintf(stderr, "%s: option %s expects a number in [%g, %g], got '%s'\n",
                     argv0, option, lo, hi, v);
        std::exit(2);
    }
    return value;
}

} // namespace kokopop::cli
