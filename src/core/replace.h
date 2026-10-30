#pragma once

#include <string>
#include <string_view>

namespace kokopop {

/// Replace all occurrences of 'from' with 'to' in 's'
void replace_all(std::string & s, std::string_view from, std::string_view to);

} // namespace kokopop
