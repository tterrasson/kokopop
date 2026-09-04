#include "model/arch.h"

#include "arch/kokoro/kokoro_arch.h"
#include "arch/sanotts/sano_arch.h"
#include "model/gguf_util.h"

#include <gguf.h>

namespace kokopop {

const char * arch_name(Arch arch) {
    switch (arch) {
        case Arch::Kokoro:  return "kokoro-82m";
        case Arch::SanoTTS: return "sanotts";
        case Arch::Unknown: break;
    }
    return "unknown";
}

Arch peek_arch(gguf_context * meta) {
    if (meta == nullptr) {
        return Arch::Unknown;
    }

    std::string name;
    if (gguf_get_str(meta, "kokopop.arch", name)) {
        if (name == "kokoro-82m" || name == "kokoro") {
            return Arch::Kokoro;
        }
        if (name == "sanotts") {
            return Arch::SanoTTS;
        }
        return Arch::Unknown;
    }

    // Kokoro GGUFs predate `kokopop.arch`. They all carry a version key, and
    // reconverting the files already in the wild is not an option.
    uint32_t kokoro_version = 0;
    if (gguf_get_u32(meta, "kokopop.kokoro.version", kokoro_version)) {
        return Arch::Kokoro;
    }

    return Arch::Unknown;
}

std::unique_ptr<ModelArch> create_arch(gguf_context * meta, std::string & error) {
    switch (peek_arch(meta)) {
        case Arch::Kokoro:
            return std::make_unique<KokoroArch>();
        case Arch::SanoTTS:
            return std::make_unique<SanoArch>();
        case Arch::Unknown:
            break;
    }

    std::string declared;
    if (gguf_get_str(meta, "kokopop.arch", declared)) {
        error = "GGUF declares an unknown kokopop.arch=\"" + declared
              + "\"; known architectures: kokoro-82m, sanotts";
    } else {
        error = "GGUF has neither kokopop.arch nor kokopop.kokoro.version; "
                "it is not a kokopop model";
    }
    return nullptr;
}

} // namespace kokopop
