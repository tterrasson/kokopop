set(KOKOPOP_WEB_LANGUAGES "all" CACHE STRING
  "eSpeak dictionary codes to bundle (e.g. en;fr), or all")

set(_data_dir "${CMAKE_CURRENT_BINARY_DIR}/espeak-ng-data")
configure_file(espeak-selection.cmake.in espeak-selection.cmake @ONLY)
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/espeak-data.stamp"
  COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_BINARY_DIR}/espeak-selection.cmake"
  COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_CURRENT_BINARY_DIR}/espeak-data.stamp"
  DEPENDS kokopop_espeak_data "${CMAKE_CURRENT_BINARY_DIR}/espeak-selection.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/select-espeak-data.cmake"
    "${PROJECT_SOURCE_DIR}/cmake/wasm-espeak.cmake"
  VERBATIM)
add_custom_target(kokopop_web_data DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/espeak-data.stamp")

add_executable(kokopop_web bindings.cpp)
target_link_libraries(kokopop_web PRIVATE kokopop)
add_dependencies(kokopop_web kokopop_web_data)

set_target_properties(kokopop_web PROPERTIES
  OUTPUT_NAME kokopop-runtime SUFFIX .js
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/dist
  LINK_DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/espeak-data.stamp")

target_link_options(kokopop_web PRIVATE
  --no-entry -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=web,worker,node
  -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=2147483648 -sSTACK_SIZE=4194304
  -sFORCE_FILESYSTEM=1
  "-sEXPORTED_RUNTIME_METHODS=['ccall','FS','UTF8ToString','HEAPF32']"
  "SHELL:--preload-file \"${_data_dir}@/espeak-ng-data\"")

if(KOKOPOP_ENABLE_WEBGPU)
  target_link_options(kokopop_web PRIVATE
    "-sJSPI_EXPORTS=['kp_load','kp_synthesize','kp_free']")
endif()

foreach(file index.js index.d.ts worker.js runtime.js package.json README.md)
  configure_file(${file} ${CMAKE_CURRENT_BINARY_DIR}/dist/${file} COPYONLY)
endforeach()

if(KOKOPOP_ENABLE_WEBGPU)
  set(KOKOPOP_WEBGPU_JS true)
else()
  set(KOKOPOP_WEBGPU_JS false)
endif()
configure_file(build-info.js.in ${CMAKE_CURRENT_BINARY_DIR}/dist/build-info.js @ONLY)

configure_file(${espeak_SOURCE_DIR}/COPYING ${CMAKE_CURRENT_BINARY_DIR}/dist/LICENSE.espeak-ng COPYONLY)
configure_file(${PROJECT_SOURCE_DIR}/LICENSE ${CMAKE_CURRENT_BINARY_DIR}/dist/LICENSE COPYONLY)
configure_file(${ggml_SOURCE_DIR}/LICENSE ${CMAKE_CURRENT_BINARY_DIR}/dist/LICENSE.ggml COPYONLY)
configure_file(${yyjson_SOURCE_DIR}/LICENSE ${CMAKE_CURRENT_BINARY_DIR}/dist/LICENSE.yyjson COPYONLY)
