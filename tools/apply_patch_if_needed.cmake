# Usage: cmake -DPATCH_FILE=<path> -P apply_patch_if_needed.cmake
# Applies the patch if not yet applied; skips silently if already applied.
if(NOT DEFINED PATCH_FILE)
  message(FATAL_ERROR "PATCH_FILE not defined")
endif()

execute_process(
  COMMAND git apply --check "${PATCH_FILE}"
  RESULT_VARIABLE _check
  ERROR_QUIET OUTPUT_QUIET
)
if(_check EQUAL 0)
  execute_process(
    COMMAND git apply "${PATCH_FILE}"
    RESULT_VARIABLE _apply
  )
  if(NOT _apply EQUAL 0)
    message(FATAL_ERROR "failed to apply patch: ${PATCH_FILE}")
  endif()
  return()
endif()

execute_process(
  COMMAND git apply --reverse --check "${PATCH_FILE}"
  RESULT_VARIABLE _rev_check
  ERROR_QUIET OUTPUT_QUIET
)
if(_rev_check EQUAL 0)
  return()  # already applied
endif()

message(FATAL_ERROR "failed to apply patch: ${PATCH_FILE}")
