# Browser builds use a single compute thread inside a dedicated JS Worker.
# No SharedArrayBuffer or cross-origin isolation headers are required.
foreach(option KOKOPOP_BUILD_TOOLS KOKOPOP_BUILD_TESTS KOKOPOP_BUILD_BENCH
    KOKOPOP_ENABLE_OPUS KOKOPOP_INSTALL BUILD_SHARED_LIBS GGML_BACKEND_DL
    GGML_NATIVE GGML_OPENMP GGML_BLAS GGML_METAL GGML_ACCELERATE)
  set(${option} OFF CACHE BOOL "" FORCE)
endforeach()
foreach(backend METAL CUDA VULKAN OPENCL)
  if(KOKOPOP_ENABLE_${backend})
    message(FATAL_ERROR "${backend} is unavailable with Emscripten; use WEBGPU or CPU")
  endif()
endforeach()
set(GGML_WEBGPU_JSPI ON CACHE BOOL "" FORCE)
add_compile_options(-msimd128 -fwasm-exceptions -sSUPPORT_LONGJMP=wasm)
add_link_options(-fwasm-exceptions -sSUPPORT_LONGJMP=wasm)
