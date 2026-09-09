#include "kokopop.h"
#include "core/backend_names.h"
#include "model/model.h"
#include "streaming/streaming.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

// ============================================================================
// Language-specific corpora selected from the voice's eSpeak identifier.
// Add twelve roughly parallel sentences per language for comparable runs.
// ============================================================================
namespace {

struct Corpus {
    std::string_view              label;      // human-readable name
    std::vector<std::string_view> codes;      // eSpeak language codes it covers
    std::vector<std::string_view> sentences;  // exactly 12
};

const std::vector<Corpus> & corpora() {
    static const std::vector<Corpus> kCorpora = {
        {"English", {"en"}, {
            "The quick brown fox jumps over the lazy dog.",
            "In a world where technology evolves faster than we can adapt, we must learn to embrace change.",
            "She asked, 'Is this the real life? Is this just fantasy?'",
            "Hello world, this is a simple test of the text-to-speech synthesis engine.",
            "The temperature outside is currently twenty-three degrees Celsius with a slight chance of rain.",
            "Why did the chicken cross the road? To get to the other side, of course!",
            "Artificial intelligence, pronounced as A-I, has revolutionized many industries in recent years.",
            "Please note that the conference room has been moved to the third floor, room three-forty-two.",
            "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness.",
            "Can you repeat that one more time, please?",
            "The project deadline is next Friday, but we might need an extension depending on the review feedback.",
            "Numbers like 42, 3.14159, and one hundred million should be handled correctly by the phonemizer.",
        }},
        {"French", {"fr"}, {
            "Le vif renard brun saute par-dessus le chien paresseux.",
            "Dans un monde où la technologie évolue plus vite que nous ne pouvons nous adapter, il faut apprendre à accepter le changement.",
            "Elle a demandé : « Est-ce la vraie vie ? Est-ce seulement un rêve ? »",
            "Bonjour tout le monde, ceci est un simple test du moteur de synthèse vocale.",
            "La température extérieure est actuellement de vingt-trois degrés Celsius, avec un léger risque de pluie.",
            "Pourquoi la poule a-t-elle traversé la route ? Pour aller de l'autre côté, bien sûr !",
            "L'intelligence artificielle, que l'on prononce I-A, a transformé de nombreux secteurs ces dernières années.",
            "Veuillez noter que la salle de réunion a été déplacée au troisième étage, salle trois cent quarante-deux.",
            "C'était le meilleur des temps, c'était le pire des temps, c'était l'âge de la sagesse, c'était l'âge de la folie.",
            "Peux-tu répéter encore une fois, s'il te plaît ?",
            "La date limite du projet est vendredi prochain, mais nous aurons peut-être besoin d'un délai selon les retours.",
            "Des nombres comme 42, 3,14159 et cent millions doivent être traités correctement par le phonémiseur.",
        }},
        {"Spanish", {"es"}, {
            "El veloz zorro marrón salta sobre el perro perezoso.",
            "En un mundo donde la tecnología avanza más rápido de lo que podemos adaptarnos, debemos aprender a aceptar el cambio.",
            "Ella preguntó: «¿Es esto la vida real? ¿Es solo una fantasía?»",
            "Hola a todos, esta es una simple prueba del motor de síntesis de voz.",
            "La temperatura exterior es de veintitrés grados centígrados, con una ligera probabilidad de lluvia.",
            "¿Por qué cruzó la gallina la carretera? Para llegar al otro lado, por supuesto.",
            "La inteligencia artificial, que se pronuncia I-A, ha transformado muchos sectores en los últimos años.",
            "Tenga en cuenta que la sala de reuniones se ha trasladado al tercer piso, sala trescientos cuarenta y dos.",
            "Era el mejor de los tiempos, era el peor de los tiempos, era la edad de la sabiduría, era la edad de la locura.",
            "¿Puedes repetir eso una vez más, por favor?",
            "La fecha límite del proyecto es el próximo viernes, pero quizá necesitemos una prórroga según los comentarios.",
            "Números como 42, 3,14159 y cien millones deben ser tratados correctamente por el fonemizador.",
        }},
        {"Italian", {"it"}, {
            "La rapida volpe marrone salta sopra il cane pigro.",
            "In un mondo in cui la tecnologia evolve più in fretta di quanto riusciamo ad adattarci, dobbiamo imparare ad accettare il cambiamento.",
            "Lei ha chiesto: «È questa la vita vera? È soltanto una fantasia?»",
            "Ciao a tutti, questo è un semplice test del motore di sintesi vocale.",
            "La temperatura esterna è attualmente di ventitré gradi centigradi, con una leggera probabilità di pioggia.",
            "Perché la gallina ha attraversato la strada? Per arrivare dall'altra parte, ovviamente!",
            "L'intelligenza artificiale, pronunciata I-A, ha trasformato molti settori negli ultimi anni.",
            "Si noti che la sala riunioni è stata spostata al terzo piano, stanza trecentoquarantadue.",
            "Era il migliore dei tempi, era il peggiore dei tempi, era l'età della saggezza, era l'età della follia.",
            "Puoi ripetere ancora una volta, per favore?",
            "La scadenza del progetto è venerdì prossimo, ma potremmo aver bisogno di una proroga in base ai riscontri.",
            "Numeri come 42, 3,14159 e cento milioni devono essere gestiti correttamente dal fonemizzatore.",
        }},
        {"Portuguese", {"pt"}, {
            "A rápida raposa marrom salta sobre o cachorro preguiçoso.",
            "Num mundo em que a tecnologia evolui mais rápido do que conseguimos acompanhar, precisamos aprender a aceitar a mudança.",
            "Ela perguntou: «Isto é a vida real? É apenas fantasia?»",
            "Olá a todos, este é um teste simples do motor de síntese de voz.",
            "A temperatura lá fora está em vinte e três graus Celsius, com uma pequena chance de chuva.",
            "Por que a galinha atravessou a estrada? Para chegar ao outro lado, é claro!",
            "A inteligência artificial, pronunciada I-A, transformou muitos setores nos últimos anos.",
            "Observe que a sala de reunião foi transferida para o terceiro andar, sala trezentos e quarenta e dois.",
            "Era o melhor dos tempos, era o pior dos tempos, era a era da sabedoria, era a era da tolice.",
            "Você pode repetir mais uma vez, por favor?",
            "O prazo do projeto é a próxima sexta-feira, mas podemos precisar de uma extensão dependendo do retorno.",
            "Números como 42, 3,14159 e cem milhões devem ser tratados corretamente pelo fonemizador.",
        }},
        {"Mandarin", {"zh", "cmn"}, {
            "那只敏捷的棕色狐狸跳过了那条懒狗。",
            "在这个技术发展快过我们适应速度的世界里，我们必须学会接受变化。",
            "她问道：“这是真实的人生吗？还是只是一场幻想？”",
            "大家好，这是语音合成引擎的一个简单测试。",
            "室外气温目前是二十三摄氏度，有小雨的可能。",
            "小鸡为什么要过马路？当然是为了到马路的另一边！",
            "人工智能，读作 A I，近年来改变了许多行业。",
            "请注意，会议室已经改到三楼，三百四十二房间。",
            "那是最好的时代，也是最坏的时代，是智慧的时代，也是愚蠢的时代。",
            "你能再重复一遍吗？",
            "项目的截止日期是下周五，但根据评审意见我们可能需要延期。",
            "像 42、3.14159 和一亿这样的数字应当被音素转换器正确处理。",
        }},
    };
    return kCorpora;
}

constexpr int kSentencesPerCorpus = 12;

/// eSpeak identifier -> bare language code: "gmw/en-US" -> "en",
/// "roa/pt-BR" -> "pt", "sit/cmn" -> "cmn".
std::string language_code(std::string_view espeak_voice) {
    const size_t slash = espeak_voice.rfind('/');
    std::string_view tail = (slash == std::string_view::npos)
        ? espeak_voice : espeak_voice.substr(slash + 1);
    const size_t dash = tail.find('-');
    if (dash != std::string_view::npos) {
        tail = tail.substr(0, dash);
    }
    std::string out(tail);
    for (char & c : out) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

const Corpus * find_corpus(std::string_view code) {
    for (const Corpus & c : corpora()) {
        for (std::string_view known : c.codes) {
            if (known == code) return &c;
        }
    }
    return nullptr;
}

std::string known_language_list() {
    std::ostringstream oss;
    for (const Corpus & c : corpora()) {
        if (oss.tellp() > 0) oss << ", ";
        oss << c.codes.front();
    }
    return oss.str();
}

const char * frontend_name(kokopop::FrontendKind kind) {
    switch (kind) {
    case kokopop::FrontendKind::Misaki: return "misaki";
    case kokopop::FrontendKind::Piper:  return "piper";
    }
    return "unknown";
}

const char * decoder_name(kokopop::DecoderKind kind) {
    switch (kind) {
    case kokopop::DecoderKind::Kokoro:    return "kokoro";
    case kokopop::DecoderKind::PiperLite: return "piperlite";
    case kokopop::DecoderKind::Vocos:     return "vocos";
    }
    return "unknown";
}

} // namespace

// ============================================================================
// Usage & argument parsing
// ============================================================================
namespace {

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model PATH [options]\n"
        "\n"
        "Measure real-time (RT) synthesis speed for a Kokoro or sanoTTS model.\n"
        "Text comes from a built-in corpus in the voice's own language, or from\n"
        "--text / --text-file. Reports per-chunk and overall RT ratios\n"
        "(1.0x = real-time, >1 = faster).\n"
        "\n"
        "Options:\n"
        "  --model PATH        Path to the GGUF model (required)\n"
        "  --voice NAME        Voice name (default: the model's default voice)\n"
        "  --backend %s Inference backend (default: auto)\n"
        "  --threads N         Thread count (default: min(4, hw_concurrency))\n"
        "  --n-sentences N     Number of corpus sentences (default: 10, range: 3-%d)\n"
        "  --lang CODE         Force a corpus language instead of the voice's own\n"
        "                      (known: %s)\n"
        "  --text TEXT         Benchmark this text instead of a corpus\n"
        "  --text-file PATH    Benchmark the contents of this file\n"
        "  --list-languages    Print the built-in corpora and exit\n"
        "  --speed F           Synthesis speed multiplier (default: 1.0)\n"
        "  --seed N            Seed for sentence selection (default: 0 = time-based)\n"
        "  --help, -h          Show this message\n"
        "\n"
        "Comparing two models fairly:\n"
        "  Use text in each voice's language. RT and compute per audio second\n"
        "  both depend on the text and chunking.\n"
        "\n"
        "Examples:\n"
        "  %s --model models/sanotts-en.gguf --voice heart\n"
        "  %s --model models/sanotts-fr-upmc-large.gguf --backend metal --threads 8\n"
        "  %s --model models/kokoro.gguf --text-file bench.txt\n",
        argv0, kokopop::backend_name_list(), kSentencesPerCorpus,
        known_language_list().c_str(), argv0, argv0, argv0);
}

void list_languages() {
    std::printf("Built-in corpora (%d sentences each):\n", kSentencesPerCorpus);
    for (const Corpus & c : corpora()) {
        std::printf("  %-4s %-12s eSpeak codes:", std::string(c.codes.front()).c_str(),
                    std::string(c.label).c_str());
        for (std::string_view code : c.codes) {
            std::printf(" %s", std::string(code).c_str());
        }
        std::printf("\n");
    }
}

const char * arg_value(int & i, int argc, char ** argv) {
    if (i + 1 >= argc) return nullptr;
    ++i;
    return argv[i];
}

bool read_text_file(const std::string & path, std::string & out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::ostringstream buf;
    buf << in.rdbuf();
    out = buf.str();
    return true;
}

struct Options {
    std::string model_path;
    std::string voice = "";   // empty = the model's default voice
    std::string lang = "";    // empty = derive from the voice
    std::string text = "";
    std::string text_source = "";  // label for the header when text is custom
    float speed = 1.0f;
    int threads = 0;          // 0 = auto
    int n_sentences = 10;
    int32_t backend = KOKOPOP_BACKEND_AUTO;
    unsigned int seed = 0;
};

bool parse_args(int argc, char ** argv, Options & opts) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0) {
            const char * v = arg_value(i, argc, argv);
            if (!v) { usage(argv[0]); return false; }
            opts.model_path = v;
        } else if (std::strcmp(argv[i], "--voice") == 0) {
            const char * v = arg_value(i, argc, argv);
            if (!v) { usage(argv[0]); return false; }
            opts.voice = v;
        } else if (std::strcmp(argv[i], "--lang") == 0) {
            const char * v = arg_value(i, argc, argv);
            if (!v) { usage(argv[0]); return false; }
            opts.lang = v;
        } else if (std::strcmp(argv[i], "--text") == 0) {
            const char * v = arg_value(i, argc, argv);
            if (!v) { usage(argv[0]); return false; }
            opts.text = v;
            opts.text_source = "--text";
        } else if (std::strcmp(argv[i], "--text-file") == 0) {
            const char * v = arg_value(i, argc, argv);
            if (!v) { usage(argv[0]); return false; }
            if (!read_text_file(v, opts.text)) {
                std::fprintf(stderr, "error: cannot read --text-file '%s'\n", v);
                return false;
            }
            opts.text_source = v;
        } else if (std::strcmp(argv[i], "--list-languages") == 0) {
            list_languages();
            std::exit(0);
        } else if (std::strcmp(argv[i], "--speed") == 0) {
            const char * v = arg_value(i, argc, argv);
            if (!v) { usage(argv[0]); return false; }
            opts.speed = std::stof(v);
        } else if (std::strcmp(argv[i], "--threads") == 0) {
            const char * v = arg_value(i, argc, argv);
            if (!v) { usage(argv[0]); return false; }
            opts.threads = std::stoi(v);
        } else if (std::strcmp(argv[i], "--n-sentences") == 0) {
            const char * v = arg_value(i, argc, argv);
            if (!v) { usage(argv[0]); return false; }
            opts.n_sentences = std::stoi(v);
        } else if (std::strcmp(argv[i], "--backend") == 0) {
            const char * v = arg_value(i, argc, argv);
            if (!v) { usage(argv[0]); return false; }
            if (!kokopop::backend_from_name(v, opts.backend)) {
                std::fprintf(stderr, "error: invalid backend '%s' (use %s)\n",
                             v, kokopop::backend_name_list());
                return false;
            }
        } else if (std::strcmp(argv[i], "--seed") == 0) {
            const char * v = arg_value(i, argc, argv);
            if (!v) { usage(argv[0]); return false; }
            opts.seed = static_cast<unsigned int>(std::stoi(v));
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "error: unknown argument '%s'\n", argv[i]);
            usage(argv[0]);
            return false;
        }
    }

    if (opts.model_path.empty()) {
        std::fprintf(stderr, "error: --model is required\n");
        usage(argv[0]);
        return false;
    }

    if (!opts.text_source.empty() && !opts.lang.empty()) {
        std::fprintf(stderr, "error: --lang only selects a built-in corpus; it does "
                             "nothing with --text/--text-file\n");
        return false;
    }

    if (!opts.text_source.empty() &&
        opts.text.find_first_not_of(" \t\r\n\v\f") == std::string::npos) {
        std::fprintf(stderr, "error: the supplied text is empty\n");
        return false;
    }

    if (opts.n_sentences < 3 || opts.n_sentences > kSentencesPerCorpus) {
        std::fprintf(stderr, "error: --n-sentences must be between 3 and %d\n",
                     kSentencesPerCorpus);
        return false;
    }

    return true;
}

} // namespace

// ============================================================================
// Timing helpers
// ============================================================================
namespace {

// ggml_time_us() wrapper — works on all platforms
int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

double ms(double us) { return us / 1000.0; }

} // namespace

// ============================================================================
// Chunk timing record
// ============================================================================
namespace {

struct ChunkResult {
    int chunk_index = 0;
    int n_tokens = 0;
    size_t n_samples = 0;
    double gen_time_ms = 0.0;
    double audio_duration_s = 0.0;
    double rt_ratio = 0.0;  // audio_duration / gen_time
};

} // namespace

// ============================================================================
// Text selection
// ============================================================================
namespace {

std::string select_sentences(const Corpus & corpus, int n_sentences, unsigned int seed) {
    const int available = static_cast<int>(corpus.sentences.size());
    n_sentences = std::clamp(n_sentences, 3, available);

    std::mt19937 rng(seed);

    std::vector<int> indices(corpus.sentences.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    // Take first n_sentences and sort for deterministic display (not shuffle order)
    std::vector<int> selected(indices.begin(), indices.begin() + n_sentences);
    std::sort(selected.begin(), selected.end());

    std::ostringstream oss;
    for (int idx : selected) {
        if (oss.tellp() > 0) {
            oss << "\n";
        }
        oss << corpus.sentences[idx];
    }
    return oss.str();
}

int count_lines(const std::string & text) {
    int lines = 0;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        if (line.find_first_not_of(" \t\r") != std::string::npos) {
            ++lines;
        }
    }
    return lines;
}

} // namespace

// ============================================================================
// Table printing helpers
// ============================================================================
namespace {

struct RunInfo {
    const char * backend_label = "";
    std::string voice;
    std::string espeak_voice;
    const char * frontend = "";
    const char * decoder = "";
    std::string text_label;   // corpus name or --text-file path
    int threads = 0;
    int sample_rate = 0;
    int n_units = 0;          // sentences, or non-empty lines of custom text
    int n_chunks = 0;
    int total_tokens = 0;
};

void print_header(const RunInfo & info) {
    std::printf("\n");
    std::printf("  Backend:     %s\n", info.backend_label);
    std::printf("  Voice:       %s (%s, frontend=%s, decoder=%s)\n",
                info.voice.c_str(), info.espeak_voice.c_str(),
                info.frontend, info.decoder);
    std::printf("  Text:        %s\n", info.text_label.c_str());
    std::printf("  Threads:     %d\n", info.threads);
    std::printf("  Sample Rate: %d Hz\n", info.sample_rate);
    std::printf("  Sentences:   %d → %d chunk(s) (%d tokens)\n",
                info.n_units, info.n_chunks, info.total_tokens);
    std::printf("\n");
}

// Column widths: 6 | 8 | 12 | 10 | 10 | 10 = 56
void print_table_header() {
    std::printf("  ┌──────┬────────┬────────────┬──────────┬──────────┬──────────┐\n");
    std::printf("  │ Chunk│ Tokens │  Gen Time  │ Duration │    RT    │ Samples  │\n");
    std::printf("  ├──────┼────────┼────────────┼──────────┼──────────┼──────────┤\n");
}

void print_table_row(const ChunkResult & cr) {
    std::printf("  │%6d│%8d│%9.1fms │%9.2fs│%9.2fx│%10zu│\n",
               cr.chunk_index + 1,
               cr.n_tokens,
               cr.gen_time_ms,
               cr.audio_duration_s,
               cr.rt_ratio,
               cr.n_samples);
}

void print_table_footer() {
    std::printf("  └──────┴────────┴────────────┴──────────┴──────────┴──────────┘\n");
}

void print_summary(const std::vector<ChunkResult> & results,
                   double total_gen_ms, double total_audio_s,
                   double overall_rt, double ttfb_warm_ms, double ttfb_cold_ms,
                   int sample_rate, int total_tokens) {
    const int n_chunks = static_cast<int>(results.size());

    std::printf("\n");
    std::printf("  Total Generation:  %6.1f ms\n", total_gen_ms);
    std::printf("  Total Audio:       %6.2f s  (%7zu samples @ %d Hz)\n",
               total_audio_s,
               std::accumulate(results.begin(), results.end(), size_t{0},
                   [](size_t acc, const ChunkResult & cr) { return acc + cr.n_samples; }),
               sample_rate);
    std::printf("  Overall RT:        %5.2fx\n", overall_rt);
    // Reciprocal of RT, in milliseconds per audio second.
    if (total_audio_s > 0.0) {
        std::printf("  Compute per audio second: %5.1f ms/s\n",
                   total_gen_ms / total_audio_s);
    }
    if (n_chunks > 0) {
        std::printf("  Chunking:          %d chunk(s), %.0f tokens/chunk avg\n",
                   n_chunks, static_cast<double>(total_tokens) / n_chunks);
    }
    std::printf("  TTFB warm-start:   %6.1f ms (1st chunk inference)\n", ttfb_warm_ms);
    std::printf("  TTFB cold-start:   %6.1f ms (load + prepare + 1st chunk)\n", ttfb_cold_ms);

    if (total_audio_s <= 0.0) {
        std::printf("  → no audio generated (check voice/model compatibility)\n");
    } else if (overall_rt > 1.0) {
        std::printf("  → %.1fx faster than real-time\n", overall_rt);
    } else if (overall_rt < 1.0) {
        std::printf("  → %.1fx slower than real-time\n", 1.0 / overall_rt);
    } else {
        std::printf("  → exactly real-time\n");
    }
    std::printf("\n");
}

} // namespace

// ============================================================================
// Main
// ============================================================================
int main(int argc, char ** argv) {
    Options opts;
    if (!parse_args(argc, argv, opts)) {
        return 2;
    }

    // Auto-detect threads
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (opts.threads <= 0) {
        opts.threads = std::min(4, hw > 0 ? hw : 1);
    }

    // ---- Load model ----
    kokopop_model_options model_opts{};
    model_opts.n_threads = opts.threads;
    model_opts.backend = opts.backend;

    const auto t_cold_start = now_us();
    const auto t_load_start = t_cold_start;
    kokopop_model * model_handle = nullptr;
    {
        std::fprintf(stderr, "[kokopop_rt] Loading model: %s\n", opts.model_path.c_str());
        const int rc = kokopop_model_load(opts.model_path.c_str(), &model_opts, &model_handle);
        if (rc != KOKOPOP_OK) {
            std::fprintf(stderr, "[kokopop_rt] model load failed: %s\n", kokopop_last_error());
            return 1;
        }
    }
    const double t_load_ms = ms(now_us() - t_load_start);

    auto model_guard = std::unique_ptr<kokopop_model, void(*)(kokopop_model *)>(
        model_handle, kokopop_model_free);

    kokopop::Model * model = kokopop_model_get_impl(model_handle);
    if (!model) {
        std::fprintf(stderr, "[kokopop_rt] failed to get model implementation\n");
        return 1;
    }

    const char * backend_name = kokopop::backend_display_name(kokopop_model_backend(model_handle));

    // ---- Resolve voice ----
    const std::string voice = kokopop::resolve_voice_name(opts.voice, *model);
    if (opts.voice.empty()) {
        std::fprintf(stderr, "[kokopop_rt] Auto-selected voice: %s\n", voice.c_str());
    }

    const auto voice_it = model->voices.find(voice);
    if (voice_it == model->voices.end()) {
        std::fprintf(stderr, "[kokopop_rt] unknown voice '%s'\n", voice.c_str());
        return 1;
    }
    const kokopop::VoiceDesc & desc = voice_it->second;

    // Multi-voice packs can mix sample rates.
    const int sample_rate = model->sample_rate(voice);

    // ---- Pick the text ----
    std::string text = opts.text;
    std::string text_label = opts.text_source;
    int n_units = 0;

    if (opts.text_source.empty()) {
        const std::string code = opts.lang.empty() ? language_code(desc.espeak_voice) : opts.lang;
        const Corpus * corpus = find_corpus(code);
        if (!corpus) {
            std::fprintf(stderr,
                "[kokopop_rt] no built-in corpus for language '%s' (voice '%s', eSpeak '%s').\n"
                "             Benchmarking it with another language's text would report\n"
                "             meaningless chunk counts and RT ratios. Pass --text/--text-file\n"
                "             with %s text, or --lang one of: %s\n",
                code.c_str(), voice.c_str(), desc.espeak_voice.c_str(),
                code.c_str(), known_language_list().c_str());
            return 2;
        }
        text = select_sentences(*corpus, opts.n_sentences, opts.seed);
        n_units = std::clamp(opts.n_sentences, 3, static_cast<int>(corpus->sentences.size()));

        std::ostringstream label;
        label << "built-in " << corpus->label << " corpus (" << corpus->codes.front() << ")";
        if (!opts.lang.empty()) {
            label << " [forced via --lang]";
        }
        text_label = label.str();
    } else {
        n_units = count_lines(text);
        if (n_units == 0) {
            std::fprintf(stderr, "[kokopop_rt] the supplied text is empty\n");
            return 2;
        }
    }

    // ---- Prepare synthesis plan (Phase 1) ----
    std::string error;
    const auto t_prepare_start = now_us();
    auto plan = kokopop::prepare_synthesis(
        *model, text, voice, opts.speed,
        kokopop::StreamMode::Adaptative, error);
    const double t_prepare_ms = ms(now_us() - t_prepare_start);

    if (plan.chunks.empty()) {
        std::fprintf(stderr, "[kokopop_rt] prepare_synthesis failed: %s\n", error.c_str());
        return 1;
    }

    const int n_chunks = static_cast<int>(plan.chunks.size());
    int total_tokens = 0;
    for (const auto & chunk : plan.chunks) {
        total_tokens += chunk.n_tokens;
    }

    // ---- Print header ----
    RunInfo info;
    info.backend_label = backend_name;
    info.voice = voice;
    info.espeak_voice = desc.espeak_voice;
    info.frontend = frontend_name(desc.frontend);
    info.decoder = decoder_name(desc.decoder);
    info.text_label = text_label;
    info.threads = opts.threads;
    info.sample_rate = sample_rate;
    info.n_units = n_units;
    info.n_chunks = n_chunks;
    info.total_tokens = total_tokens;
    print_header(info);

    std::printf("  Model load time: %6.1f ms\n", t_load_ms);
    std::printf("  Prepare time:    %6.1f ms (chunking + phonemization)\n\n", t_prepare_ms);

    // ---- Capture cold-start breakdown ----
    const double t_load_prepare_ms = ms(now_us() - t_cold_start);
    std::fprintf(stderr, "[kokopop_rt] cold-start breakdown: load=%.1fms prepare=%.1fms total=%.1fms\n",
                t_load_ms, t_prepare_ms, t_load_prepare_ms);

    // ---- Infer each chunk (Phase 2) with timing ----
    print_table_header();

    std::vector<ChunkResult> results;
    results.reserve(n_chunks);

    double ttfb_warm_ms = 0.0;  // 1st chunk inference only
    double ttfb_cold_ms = 0.0;  // model load → 1st chunk complete
    double total_gen_ms = 0.0;
    double total_audio_s = 0.0;
    int    timed_tokens = 0;    // tokens of the chunks that actually ran

    std::vector<float> prev_tail;

    for (int i = 0; i < n_chunks; ++i) {
        const auto t_chunk_start = now_us();

        std::vector<float> out_tail;
        auto audio = kokopop::infer_chunk(
            *model, plan, i, prev_tail, out_tail, error);

        const double t_chunk_ms = ms(now_us() - t_chunk_start);

        if (audio.empty()) {
            std::fprintf(stderr, "\n  [WARN] chunk[%d] failed: %s — skipped\n", i + 1, error.c_str());
            continue;
        }

        const double audio_dur_s = static_cast<double>(audio.size()) / sample_rate;
        const double gen_sec = t_chunk_ms / 1000.0;
        const double rt_ratio = (gen_sec > 0.0) ? (audio_dur_s / gen_sec) : 0.0;

        ChunkResult cr;
        cr.chunk_index = i;
        cr.n_tokens = plan.chunks[i].n_tokens;
        cr.n_samples = audio.size();
        cr.gen_time_ms = t_chunk_ms;
        cr.audio_duration_s = audio_dur_s;
        cr.rt_ratio = rt_ratio;

        if (i == 0) {
            ttfb_warm_ms = t_chunk_ms;
            ttfb_cold_ms = t_load_prepare_ms + t_chunk_ms;
        }

        total_gen_ms += t_chunk_ms;
        total_audio_s += audio_dur_s;
        timed_tokens += cr.n_tokens;

        results.push_back(std::move(cr));
        prev_tail = std::move(out_tail);

        print_table_row(results.back());
    }

    print_table_footer();

    // ---- Summary ----
    const double overall_rt = (total_gen_ms > 0.0)
        ? (total_audio_s / (total_gen_ms / 1000.0))
        : 0.0;

    print_summary(results, total_gen_ms, total_audio_s, overall_rt,
                  ttfb_warm_ms, ttfb_cold_ms, sample_rate, timed_tokens);

    return 0;
}
