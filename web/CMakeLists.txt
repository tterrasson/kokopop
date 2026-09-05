add_executable(kokopop_web bindings.cpp)
target_link_libraries(kokopop_web PRIVATE kokopop)
add_dependencies(kokopop_web kokopop_espeak_data)

set_target_properties(kokopop_web PROPERTIES
  OUTPUT_NAME kokopop-runtime SUFFIX .js
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/dist)

target_link_options(kokopop_web PRIVATE
  --no-entry -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=web,worker,node
  -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=2147483648 -sSTACK_SIZE=4194304
  -sFORCE_FILESYSTEM=1
  "-sEXPORTED_RUNTIME_METHODS=['ccall','FS','UTF8ToString','HEAPF32']"
  "SHELL:--preload-file ${KOKOPOP_ESPEAK_DATA_DIR}@/espeak-ng-data")

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
