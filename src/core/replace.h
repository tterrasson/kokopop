#pragma once

#include <string>

namespace kokopop {

/// Replace all occurrences of 'from' with 'to' in 's'
void replace_all(std::string & s, const std::string & from, const std::string & to);

} // namespace kokopop
