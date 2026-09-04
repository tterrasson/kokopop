#pragma once

#include "model/model.h"
#include "streaming/streaming.h"
#include "synthesis/chunker/chunker.h"

#include <cstddef>
#include <string>
#include <vector>

namespace kokopop {

struct SynthesisSessionOptions {
    std::string voice;
    float speed = 1.0f;
    StreamMode mode = StreamMode::Adaptative;
    ChunkConfig chunk_config{};
    bool has_chunk_config = false;
    bool use_exact_chunk_config = false;
    KokoroDiffusionOptions diffusion{};

    /// sanoTTS's deterministic noise seed, absent by default. The flag is
    /// separate so that 0 stays a usable explicit seed.
    bool     has_noise_seed = false;
    uint64_t noise_seed     = 0;
};

struct SynthesisAudioChunk {
    std::vector<float> samples;
    int chunk_index = 0;
    bool is_final = false;
};

class SynthesisSession {
public:
    SynthesisSession(Model & model, SynthesisSessionOptions options);

    bool push_text(const std::string & text, std::string & error);
    bool finish_input(std::string & error);
    bool prepare(std::string & error);
    bool next(size_t max_chunks, size_t queued_requests,
              std::vector<SynthesisAudioChunk> & out,
              std::string & error);

    bool done() const { return _done; }
    int chunks_total() const { return _chunks_total; }
    /// The rate of *this session's* voice. A model may carry voices at
    /// different rates (a sanoTTS pack mixes 22050 and 24000 Hz), so the
    /// model-level accessor which answers for the default voice is the
    /// wrong one for anything that has resolved a voice.
    int sample_rate() const { return _model.sample_rate(_options.voice); }

private:
    ChunkConfig selected_config() const;

    Model & _model;
    SynthesisSessionOptions _options;
    std::string _text;
    bool _input_finished = false;
    bool _started = false;
    bool _prepared = false;
    bool _done = false;

    SynthesisPlan _plan;
    /// The resolved voice's tokenizer, bound in `prepare()`. Captures `_model`
    /// by reference, which outlives the session.
    TokenizeFn _tokenize;
    std::vector<Unit> _adaptative_units;
    size_t _adaptative_next_unit = 0;
    AdaptativeChunkController _adaptative_controller;
    std::vector<float> _prev_tail;
    int _chunks_completed = 0;
    int _chunks_total = 0; // 0 means unknown for adaptative until the final chunk.
};

} // namespace kokopop
