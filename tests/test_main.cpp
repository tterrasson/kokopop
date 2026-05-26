#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#define CHECK_NEAR(a, b, eps) CHECK(std::fabs(static_cast<double>(a) - static_cast<double>(b)) <= (eps))

// Include all test files
#include "test_api.cpp"
#include "test_phonemizer.cpp"
#include "test_utf8.cpp"
#include "test_wav.cpp"
#include "test_tokenizer.cpp"
#include "test_backend.cpp"
#include "test_kokoro_real.cpp"
#include "test_mock_synthesis.cpp"
#include "test_text_normalizer.cpp"
#include "test_audio_postprocess.cpp"
#include "test_replace.cpp"
#include "test_model_internal.cpp"
#include "test_audio_utils.cpp"
#include "test_lstm_fused.cpp"
#include "test_diffusion_kernels.cpp"
#include "test_text_splitter.cpp"
#include "test_zh_g2p.cpp"
#include "test_unicode_inputs.cpp"
#include "test_chunker.cpp"
#include "test_file_mapping.cpp"
#include "test_streaming_pipeline.cpp"
#include "test_http_async_parser.cpp"
#include "test_http_server.cpp"
#include "test_ogg_opus.cpp"
#include "test_last_error.cpp"
