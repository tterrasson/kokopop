#include "replace.h"

#include <algorithm>
#include <string>

namespace kokopop {
namespace {

void replace_in(std::string & s, const std::string & from, const std::string & to) {
    if (from.empty()) return;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

} // namespace

void replace_all(std::string & s, const std::string & from, const std::string & to) {
    replace_in(s, from, to);
}

} // namespace kokopop
