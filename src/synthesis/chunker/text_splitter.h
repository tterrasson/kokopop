#pragma once

#include "synthesis/chunker/chunker.h"

#include <string>
#include <vector>

namespace kokopop {

// ---------------------------------------------------------------------------
// Text splitter — split normalized text into candidate units
//
// Hierarchy of boundaries (strongest first):
//   1. Paragraphs (double newline)
//   2. Sentences (. ! ? … followed by space/end)
//   3. Strong clauses (; :)
//   4. Weak clauses (, — parenthetical)
//   5. Spaces between words
//   6. Character fallback for very long tokens
//
// The splitter respects protected tokens (abbreviations, decimals, URLs)
// and does not split on their internal dots.
// ---------------------------------------------------------------------------

/// Infer the boundary type based on how a text fragment ends
Boundary infer_boundary_type(const std::string & text);

/// Split text into candidate units (paragraphs → sentences → clauses)
/// Each unit keeps its delimiter
std::vector<std::string> split_into_candidate_units(const std::string & text);

/// Split a single oversized unit by punctuation boundaries
std::vector<std::string> force_split_unit(const std::string & text);

/// Split text by words (keeping punctuation attached to words)
std::vector<std::string> split_by_words(const std::string & text);

/// Split text into sentences, keeping delimiters
std::vector<std::string> split_sentences(const std::string & text);

/// Split text into paragraphs, keeping delimiters
std::vector<std::string> split_paragraphs(const std::string & text);

// ---------------------------------------------------------------------------
// Utility: split while keeping delimiter with the preceding fragment
// ---------------------------------------------------------------------------
std::vector<std::string> split_keep_delimiter(
    const std::string & text,
    const std::vector<std::string> & delimiters);

// ---------------------------------------------------------------------------
// Check if a position in text is a valid sentence boundary
// ---------------------------------------------------------------------------
bool is_sentence_boundary(const std::string & text, size_t pos);

// ---------------------------------------------------------------------------
// Check if a character at position is a protected dot (should not split)
// ---------------------------------------------------------------------------
bool is_protected_dot(const std::string & text, size_t pos);

} // namespace kokopop
