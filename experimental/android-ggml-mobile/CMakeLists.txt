cmake_minimum_required(VERSION 3.22)

project(aivis_android_ggml_mobile LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TTS_CPP_ROOT "" CACHE PATH "Path to a TTS.cpp checkout.")
set(VULKAN_HPP_INCLUDE_DIR "/usr/include" CACHE PATH "Host Vulkan-Hpp include root used when the Android NDK does not ship vulkan/vulkan.hpp.")
set(SPIRV_HEADERS_INCLUDE_DIR "/usr/include" CACHE PATH "Host SPIR-V headers include root containing spirv/unified1/spirv.hpp.")
option(AIVIS_ANDROID_PATCH_GGML_VULKAN_CORE_EXTENSIONS "Patch ggml-vulkan Android core-extension createDevice compatibility in the benchmark build." ON)
if (NOT TTS_CPP_ROOT)
    message(FATAL_ERROR "Set -DTTS_CPP_ROOT=/path/to/TTS.cpp")
endif()
get_filename_component(TTS_CPP_ROOT "${TTS_CPP_ROOT}" ABSOLUTE)
if (NOT EXISTS "${TTS_CPP_ROOT}/ggml/CMakeLists.txt")
    message(FATAL_ERROR "TTS_CPP_ROOT does not look like a TTS.cpp checkout: ${TTS_CPP_ROOT}")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
set(GGML_ALL_WARNINGS_3RD_PARTY OFF CACHE BOOL "" FORCE)
set(GGML_FATAL_WARNINGS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN ON CACHE BOOL "" FORCE)
set(TTS_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
set(TTS_ALL_WARNINGS_3RD_PARTY OFF CACHE BOOL "" FORCE)
set(TTS_FATAL_WARNINGS OFF CACHE BOOL "" FORCE)
set(TTS_BUILD_KOKORO OFF CACHE BOOL "" FORCE)

if (EXISTS "${VULKAN_HPP_INCLUDE_DIR}/vulkan/vulkan.hpp")
    file(GLOB AIVIS_ANDROID_VULKAN_HPP_HEADERS "${VULKAN_HPP_INCLUDE_DIR}/vulkan/*.hpp")
    set(AIVIS_ANDROID_VULKAN_HPP_STAGING_DIR "${CMAKE_BINARY_DIR}/vulkan-hpp")
    file(MAKE_DIRECTORY "${AIVIS_ANDROID_VULKAN_HPP_STAGING_DIR}/vulkan")
    file(COPY ${AIVIS_ANDROID_VULKAN_HPP_HEADERS} DESTINATION "${AIVIS_ANDROID_VULKAN_HPP_STAGING_DIR}/vulkan")
else()
    message(FATAL_ERROR "Set -DVULKAN_HPP_INCLUDE_DIR to a Vulkan-Hpp include root containing vulkan/vulkan.hpp.")
endif()
if (EXISTS "${SPIRV_HEADERS_INCLUDE_DIR}/spirv/unified1/spirv.hpp")
    file(COPY "${SPIRV_HEADERS_INCLUDE_DIR}/spirv" DESTINATION "${AIVIS_ANDROID_VULKAN_HPP_STAGING_DIR}")
else()
    message(FATAL_ERROR "Set -DSPIRV_HEADERS_INCLUDE_DIR to an include root containing spirv/unified1/spirv.hpp.")
endif()

add_subdirectory("${TTS_CPP_ROOT}/ggml" "${CMAKE_BINARY_DIR}/tts-cpp-ggml")
if (TARGET ggml-vulkan)
    target_include_directories(ggml-vulkan PRIVATE "${AIVIS_ANDROID_VULKAN_HPP_STAGING_DIR}")
    if (AIVIS_ANDROID_PATCH_GGML_VULKAN_CORE_EXTENSIONS)
        set(AIVIS_ANDROID_GGML_VULKAN_ORIGINAL "${TTS_CPP_ROOT}/ggml/src/ggml-vulkan/ggml-vulkan.cpp")
        set(AIVIS_ANDROID_GGML_VULKAN_PATCHED "${CMAKE_BINARY_DIR}/patched-sources/ggml-vulkan.cpp")
        file(READ "${AIVIS_ANDROID_GGML_VULKAN_ORIGINAL}" AIVIS_ANDROID_GGML_VULKAN_SOURCE)
        string(REPLACE
            "        device_extensions.push_back(\"VK_KHR_16bit_storage\");"
            "        if (fp16_storage) {\n            device_extensions.push_back(\"VK_KHR_16bit_storage\");\n        }"
            AIVIS_ANDROID_GGML_VULKAN_SOURCE
            "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}"
        )
        string(REPLACE
            "        device->fp16 = !force_disable_f16 && fp16_storage && fp16_compute;"
            "        device->fp16 = !force_disable_f16;"
            AIVIS_ANDROID_GGML_VULKAN_SOURCE
            "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}"
        )
        string(REPLACE
            "        bool fp16 = !force_disable_f16 && fp16_storage && fp16_compute;"
            "        bool fp16 = !force_disable_f16;"
            AIVIS_ANDROID_GGML_VULKAN_SOURCE
            "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}"
        )
        string(REPLACE
            "    bool fp16 = !force_disable_f16 && fp16_storage && fp16_compute;"
            "    bool fp16 = !force_disable_f16;"
            AIVIS_ANDROID_GGML_VULKAN_SOURCE
            "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}"
        )
        string(REPLACE
            "        device->fp16 = device->fp16 && vk12_features.shaderFloat16;"
            "        device->fp16 = device->fp16 && vk11_features.storageBuffer16BitAccess && vk12_features.shaderFloat16;"
            AIVIS_ANDROID_GGML_VULKAN_SOURCE
            "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}"
        )
        string(REPLACE
            "    fp16 = fp16 && vk12_features.shaderFloat16;"
            "    fp16 = fp16 && vk11_features.storageBuffer16BitAccess && vk12_features.shaderFloat16;"
            AIVIS_ANDROID_GGML_VULKAN_SOURCE
            "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}"
        )
        string(REPLACE
            "        if (device->fp16) {\n            device_extensions.push_back(\"VK_KHR_shader_float16_int8\");\n        }"
            "        if (device->fp16 && fp16_compute) {\n            device_extensions.push_back(\"VK_KHR_shader_float16_int8\");\n        }"
            AIVIS_ANDROID_GGML_VULKAN_SOURCE
            "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}"
        )
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/patched-sources")
        file(WRITE "${AIVIS_ANDROID_GGML_VULKAN_PATCHED}" "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}")
        get_target_property(AIVIS_ANDROID_GGML_VULKAN_SOURCES ggml-vulkan SOURCES)
        set(AIVIS_ANDROID_GGML_VULKAN_REPLACED_SOURCES)
        foreach(AIVIS_ANDROID_GGML_VULKAN_SOURCE_FILE IN LISTS AIVIS_ANDROID_GGML_VULKAN_SOURCES)
            if (AIVIS_ANDROID_GGML_VULKAN_SOURCE_FILE STREQUAL "ggml-vulkan.cpp")
                list(APPEND AIVIS_ANDROID_GGML_VULKAN_REPLACED_SOURCES "${AIVIS_ANDROID_GGML_VULKAN_PATCHED}")
            else()
                list(APPEND AIVIS_ANDROID_GGML_VULKAN_REPLACED_SOURCES "${AIVIS_ANDROID_GGML_VULKAN_SOURCE_FILE}")
            endif()
        endforeach()
        set_property(TARGET ggml-vulkan PROPERTY SOURCES "${AIVIS_ANDROID_GGML_VULKAN_REPLACED_SOURCES}")
    endif()
endif()
add_subdirectory("${TTS_CPP_ROOT}/ggml-patches" "${CMAKE_BINARY_DIR}/tts-cpp-ggml-patches")
add_subdirectory("${TTS_CPP_ROOT}/src" "${CMAKE_BINARY_DIR}/tts-cpp-src")

add_executable(aivis_ggml_mobile_bench
    src/aivis_ggml_mobile_bench.cpp
)
target_include_directories(aivis_ggml_mobile_bench PRIVATE
    "${TTS_CPP_ROOT}/include"
)
target_link_libraries(aivis_ggml_mobile_bench PRIVATE tts)
