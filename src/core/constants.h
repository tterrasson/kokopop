#pragma once

// ---------------------------------------------------------------------------
// Kokoro model architecture constants
// ---------------------------------------------------------------------------
constexpr int        KOKOPOP_HIDDEN_SIZE = 768;
constexpr int        KOKOPOP_NUM_HEADS   = 12;
constexpr int        KOKOPOP_HEAD_SIZE   = KOKOPOP_HIDDEN_SIZE / KOKOPOP_NUM_HEADS;
constexpr float      KOKOPOP_ATTN_SCALE  = 0.125f;

// ---------------------------------------------------------------------------
// Generator ResBlock constants
// ---------------------------------------------------------------------------
constexpr int KOKOPOP_RESBLOCK_DILATIONS[] = {1, 3, 5};
constexpr int KOKOPOP_RESBLOCK_KERNELS[]   = {3, 7, 11};

// ---------------------------------------------------------------------------
// STFT / audio synthesis constants
// ---------------------------------------------------------------------------
constexpr int   KOKOPOP_STFT_N            = 20;
constexpr int   KOKOPOP_STFT_HOP          = 5;
constexpr int   KOKOPOP_SAMPLE_RATE       = 24000;
constexpr int   KOKOPOP_HARMONIC_COUNT    = 9;
constexpr int   KOKOPOP_UPSAMPLE          = 300;
constexpr float KOKOPOP_SINE_AMP          = 0.1f;

// ---------------------------------------------------------------------------
// Miscellaneous math constants
// ---------------------------------------------------------------------------
// M_PI from <math.h>/<cmath>; provide fallback if the platform omits it.
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// ---------------------------------------------------------------------------
// Model tensor name prefixes
// ---------------------------------------------------------------------------
constexpr const char * KOKOPOP_PREFIX_ALBERT_LAYER =
    "kokopop.albert.encoder.albert_layer_groups.0.albert_layers.0.";
constexpr const char * KOKOPOP_PREFIX_VOICE = "kokopop.voice.";

// ---------------------------------------------------------------------------
// IO helpers
// ---------------------------------------------------------------------------
constexpr size_t KOKOPOP_IO_BUF_SIZE = 4 * 1024 * 1024;  // 4 MB read buffer

// ---------------------------------------------------------------------------
// Helper constants
// ---------------------------------------------------------------------------
constexpr float KOKOPOP_INV_SQRT2 = 0.7071067811865476f;  // 1 / sqrt(2)
