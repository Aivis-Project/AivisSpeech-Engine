cmake_minimum_required(VERSION 3.24)

project(aivis_ggml_onnx_ep LANGUAGES CXX)

set(ORT_INCLUDE_DIR "" CACHE PATH "Path to ONNX Runtime public headers")

if(NOT ORT_INCLUDE_DIR)
  message(FATAL_ERROR "Set ORT_INCLUDE_DIR to the directory containing onnxruntime_cxx_api.h")
endif()

add_library(aivis_ggml_onnx_ep SHARED
  src/aivis_ggml_onnx_ep.cc
)

target_compile_features(aivis_ggml_onnx_ep PRIVATE cxx_std_17)
target_include_directories(aivis_ggml_onnx_ep PRIVATE "${ORT_INCLUDE_DIR}")
target_link_libraries(aivis_ggml_onnx_ep PRIVATE ${CMAKE_DL_LIBS})
target_compile_definitions(aivis_ggml_onnx_ep PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)

set_target_properties(aivis_ggml_onnx_ep PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  OUTPUT_NAME aivis_ggml_onnx_ep
)

install(TARGETS aivis_ggml_onnx_ep
  LIBRARY DESTINATION onnxruntime_ep_aivis_ggml/lib
  RUNTIME DESTINATION onnxruntime_ep_aivis_ggml/lib
)
