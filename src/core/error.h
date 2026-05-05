#pragma once

#include <string>

namespace kokopop {

void set_error(std::string message);
const char * last_error();

} // namespace kokopop

