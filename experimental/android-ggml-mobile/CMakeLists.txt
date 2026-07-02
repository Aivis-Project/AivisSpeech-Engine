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

find_package(Vulkan REQUIRED)

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
        string(REPLACE
            "            device->mul_mat_l_int[i]    = device->mul_mat_l[i];"
            "            if (getenv(\"GGML_VK_ALLOW_F16_COOPMAT_ONLY\") != nullptr &&\n                device->coopmat_support &&\n                device->coopmat_acc_f16_support &&\n                !device->coopmat_acc_f32_support) {\n                device->mul_mat_l[i] = false;\n                device->mul_mat_m[i] = false;\n                device->mul_mat_s[i] = true;\n                device->mul_mat_id_l[i] = false;\n                device->mul_mat_id_m[i] = false;\n                device->mul_mat_id_s[i] = true;\n            }\n\n            device->mul_mat_l_int[i]    = device->mul_mat_l[i];"
            AIVIS_ANDROID_GGML_VULKAN_SOURCE
            "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}"
        )
        string(CONCAT AIVIS_ANDROID_ALLOW_F16_ONLY_COOPMAT_REPLACEMENT
            "            const bool aivis_allow_f16_only_coopmat = getenv(\"GGML_VK_ALLOW_F16_COOPMAT_ONLY\") != nullptr;\n"
            "            if (device->coopmat_m == 0 || "
            "(!device->coopmat_acc_f32_support && "
            "!(aivis_allow_f16_only_coopmat && device->coopmat_acc_f16_support))) {"
        )
        string(REPLACE
            "            if (device->coopmat_m == 0 || !device->coopmat_acc_f32_support) {"
            "${AIVIS_ANDROID_ALLOW_F16_ONLY_COOPMAT_REPLACEMENT}"
            AIVIS_ANDROID_GGML_VULKAN_SOURCE
            "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}"
        )
        string(REPLACE
            "    if (device->coopmat_support) {\n        // Create 6 variants, {s,m,l}x{unaligned,aligned}"
            "    if (device->coopmat_support) {\n        const bool aivis_f16_only_coopmat = getenv(\"GGML_VK_ALLOW_F16_COOPMAT_ONLY\") != nullptr &&\n                                                device->coopmat_acc_f16_support &&\n                                                !device->coopmat_acc_f32_support;\n        // Create 6 variants, {s,m,l}x{unaligned,aligned}"
            AIVIS_ANDROID_GGML_VULKAN_SOURCE
            "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}"
        )
        string(REPLACE
            "#define CREATE_MM2(TYPE, PIPELINE_NAME, NAMELC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \\\n        if (device->coopmat_acc_f16_support) { \\\n            CREATE_MM(TYPE, PIPELINE_NAME . f16acc, NAMELC, _f16acc, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \\\n        } \\\n        if (device->coopmat_acc_f32_support) { \\\n            CREATE_MM(TYPE, PIPELINE_NAME . f32acc, NAMELC, , WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \\\n        } \\"
            "#define AIVIS_KHR_F16_ONLY_COOPMAT_NAMELC_OK(NAMELC) \\\n        (std::strcmp(#NAMELC, \"matmul_f16\") == 0 || \\\n         std::strcmp(#NAMELC, \"matmul_id_subgroup_f16\") == 0 || \\\n         std::strcmp(#NAMELC, \"matmul_id_f16\") == 0)\n#define CREATE_MM2(TYPE, PIPELINE_NAME, NAMELC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \\\n        if (device->coopmat_acc_f16_support && (!aivis_f16_only_coopmat || AIVIS_KHR_F16_ONLY_COOPMAT_NAMELC_OK(NAMELC))) { \\\n            CREATE_MM(TYPE, PIPELINE_NAME . f16acc, NAMELC, _f16acc, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \\\n        } \\\n        if (device->coopmat_acc_f32_support) { \\\n            CREATE_MM(TYPE, PIPELINE_NAME . f32acc, NAMELC, , WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \\\n        } \\"
            AIVIS_ANDROID_GGML_VULKAN_SOURCE
            "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}"
        )
        string(REPLACE
            "        CREATE_MM(GGML_TYPE_F32, pipeline_matmul_f32, matmul_f32_f32, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, );\n        CREATE_MM(GGML_TYPE_F32, pipeline_matmul_f32_f16, matmul_f32_f16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, );"
            "        if (!aivis_f16_only_coopmat) {\n            CREATE_MM(GGML_TYPE_F32, pipeline_matmul_f32, matmul_f32_f32, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, );\n            CREATE_MM(GGML_TYPE_F32, pipeline_matmul_f32_f16, matmul_f32_f16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, );\n        }"
            AIVIS_ANDROID_GGML_VULKAN_SOURCE
            "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}"
        )
        string(REPLACE
            "        if (device->mul_mat ## ID ## _l[TYPE]) \\\n            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_l, #NAMELC #F16ACC \"_aligned_l\", NAMELC ## _aligned ## F16ACC ## _cm1_len, NAMELC ## _aligned ## F16ACC ## _cm1_data, \"main\", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, l_align, false, true);"
            "        if (!aivis_f16_only_coopmat && device->mul_mat ## ID ## _l[TYPE]) \\\n            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_l, #NAMELC #F16ACC \"_aligned_l\", NAMELC ## _aligned ## F16ACC ## _cm1_len, NAMELC ## _aligned ## F16ACC ## _cm1_data, \"main\", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, l_align, false, true);"
            AIVIS_ANDROID_GGML_VULKAN_SOURCE
            "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}"
        )
        string(REPLACE
            "        if (device->mul_mat ## ID ## _m[TYPE]) \\\n            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_m, #NAMELC #F16ACC \"_aligned_m\", NAMELC ## _aligned ## F16ACC ## _cm1_len, NAMELC ## _aligned ## F16ACC ## _cm1_data, \"main\", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, m_align, false, true);"
            "        if (!aivis_f16_only_coopmat && device->mul_mat ## ID ## _m[TYPE]) \\\n            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_m, #NAMELC #F16ACC \"_aligned_m\", NAMELC ## _aligned ## F16ACC ## _cm1_len, NAMELC ## _aligned ## F16ACC ## _cm1_data, \"main\", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, m_align, false, true);"
            AIVIS_ANDROID_GGML_VULKAN_SOURCE
            "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}"
        )
        string(REPLACE
            "        if (device->mul_mat ## ID ## _s[TYPE]) \\\n            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_s, #NAMELC #F16ACC \"_aligned_s\", NAMELC ## _aligned ## F16ACC ## _cm1_len, NAMELC ## _aligned ## F16ACC ## _cm1_data, \"main\", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, s_align, false, true);"
            "        if (!aivis_f16_only_coopmat && device->mul_mat ## ID ## _s[TYPE]) \\\n            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_s, #NAMELC #F16ACC \"_aligned_s\", NAMELC ## _aligned ## F16ACC ## _cm1_len, NAMELC ## _aligned ## F16ACC ## _cm1_data, \"main\", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, s_align, false, true);"
            AIVIS_ANDROID_GGML_VULKAN_SOURCE
            "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}"
        )
        string(REPLACE
            "#undef CREATE_MM2\n#undef CREATE_MM"
            "#undef CREATE_MM2\n#undef AIVIS_KHR_F16_ONLY_COOPMAT_NAMELC_OK\n#undef CREATE_MM"
            AIVIS_ANDROID_GGML_VULKAN_SOURCE
            "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}"
        )
        string(REPLACE
            "static vk_matmul_pipeline ggml_vk_get_mul_mat_mat_pipeline(ggml_backend_vk_context * ctx, ggml_type src0_type, ggml_type src1_type, ggml_prec prec) {\n    VK_LOG_DEBUG(\"ggml_vk_get_mul_mat_mat_pipeline(\" << ggml_type_name(src0_type) << \", \" << ggml_type_name(src1_type) << \", \" << prec << \")\");\n    if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {\n        return ctx->device->pipeline_matmul_f32;\n    }"
            "static vk_matmul_pipeline ggml_vk_get_mul_mat_mat_pipeline(ggml_backend_vk_context * ctx, ggml_type src0_type, ggml_type src1_type, ggml_prec prec) {\n    VK_LOG_DEBUG(\"ggml_vk_get_mul_mat_mat_pipeline(\" << ggml_type_name(src0_type) << \", \" << ggml_type_name(src1_type) << \", \" << prec << \")\");\n    const bool aivis_f16_only_coopmat = getenv(\"GGML_VK_ALLOW_F16_COOPMAT_ONLY\") != nullptr &&\n                                        ctx->device->coopmat_support &&\n                                        ctx->device->coopmat_acc_f16_support &&\n                                        !ctx->device->coopmat_acc_f32_support;\n    if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {\n        return aivis_f16_only_coopmat ? nullptr : ctx->device->pipeline_matmul_f32;\n    }"
            AIVIS_ANDROID_GGML_VULKAN_SOURCE
            "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}"
        )
        string(REPLACE
            "    if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F16) {\n        return ctx->device->pipeline_matmul_f32_f16;\n    }"
            "    if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F16) {\n        return aivis_f16_only_coopmat ? nullptr : ctx->device->pipeline_matmul_f32_f16;\n    }"
            AIVIS_ANDROID_GGML_VULKAN_SOURCE
            "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}"
        )
        string(REPLACE
            "        if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {\n            return ctx->device->pipeline_matmul_f16_f32.f16acc;\n        }"
            "        if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {\n            return aivis_f16_only_coopmat ? nullptr : ctx->device->pipeline_matmul_f16_f32.f16acc;\n        }"
            AIVIS_ANDROID_GGML_VULKAN_SOURCE
            "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}"
        )
        string(REPLACE
            "        if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {\n            return ctx->device->pipeline_matmul_f16_f32.f32acc;\n        }"
            "        if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {\n            return aivis_f16_only_coopmat ? nullptr : ctx->device->pipeline_matmul_f16_f32.f32acc;\n        }"
            AIVIS_ANDROID_GGML_VULKAN_SOURCE
            "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}"
        )
        string(REPLACE
            "    if (ctx->device->coopmat_support) {\n        return (ctx->device->fp16 && ctx->device->coopmat_acc_f16_support && prec == GGML_PREC_DEFAULT) ? ctx->device->pipeline_dequant_mul_mat_mat[src0_type].f16acc : ctx->device->pipeline_dequant_mul_mat_mat[src0_type].f32acc;\n    }"
            "    if (ctx->device->coopmat_support) {\n        if (aivis_f16_only_coopmat) {\n            return nullptr;\n        }\n        return (ctx->device->fp16 && ctx->device->coopmat_acc_f16_support && prec == GGML_PREC_DEFAULT) ? ctx->device->pipeline_dequant_mul_mat_mat[src0_type].f16acc : ctx->device->pipeline_dequant_mul_mat_mat[src0_type].f32acc;\n    }"
            AIVIS_ANDROID_GGML_VULKAN_SOURCE
            "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}"
        )
        string(REPLACE
            "static vk_matmul_pipeline ggml_vk_get_mul_mat_mat_id_pipeline(ggml_backend_vk_context * ctx, ggml_type src0_type, ggml_type src1_type, ggml_prec prec) {\n    VK_LOG_DEBUG(\"ggml_vk_get_mul_mat_mat_id_pipeline()\");\n    if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {\n        return ctx->device->pipeline_matmul_id_f32;\n    }"
            "static vk_matmul_pipeline ggml_vk_get_mul_mat_mat_id_pipeline(ggml_backend_vk_context * ctx, ggml_type src0_type, ggml_type src1_type, ggml_prec prec) {\n    VK_LOG_DEBUG(\"ggml_vk_get_mul_mat_mat_id_pipeline()\");\n    const bool aivis_f16_only_coopmat = getenv(\"GGML_VK_ALLOW_F16_COOPMAT_ONLY\") != nullptr &&\n                                        ctx->device->coopmat_support &&\n                                        ctx->device->coopmat_acc_f16_support &&\n                                        !ctx->device->coopmat_acc_f32_support;\n    if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {\n        return aivis_f16_only_coopmat ? nullptr : ctx->device->pipeline_matmul_id_f32;\n    }"
            AIVIS_ANDROID_GGML_VULKAN_SOURCE
            "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}"
        )
        string(REPLACE
            "        if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {\n            return ctx->device->pipeline_matmul_id_f16_f32.f16acc;\n        }"
            "        if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {\n            return aivis_f16_only_coopmat ? nullptr : ctx->device->pipeline_matmul_id_f16_f32.f16acc;\n        }"
            AIVIS_ANDROID_GGML_VULKAN_SOURCE
            "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}"
        )
        string(REPLACE
            "        if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {\n            return ctx->device->pipeline_matmul_id_f16_f32.f32acc;\n        }"
            "        if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {\n            return aivis_f16_only_coopmat ? nullptr : ctx->device->pipeline_matmul_id_f16_f32.f32acc;\n        }"
            AIVIS_ANDROID_GGML_VULKAN_SOURCE
            "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}"
        )
        string(REPLACE
            "    vk_matmul_pipeline2& mmp = ctx->device->pipeline_dequant_mul_mat_mat_id[src0_type];\n    // XXX TODO 'prec' is not actually allowed in mul_mat_id."
            "    if (aivis_f16_only_coopmat) {\n        return nullptr;\n    }\n\n    vk_matmul_pipeline2& mmp = ctx->device->pipeline_dequant_mul_mat_mat_id[src0_type];\n    // XXX TODO 'prec' is not actually allowed in mul_mat_id."
            AIVIS_ANDROID_GGML_VULKAN_SOURCE
            "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}"
        )
        string(REPLACE
            "    const bool y_non_contig = (ctx->device->coopmat2 && src1->type == GGML_TYPE_F32) ||\n                              (src0->type == GGML_TYPE_BF16 && src1->type != GGML_TYPE_BF16) ||"
            "    const bool aivis_f16_only_coopmat = getenv(\"GGML_VK_ALLOW_F16_COOPMAT_ONLY\") != nullptr &&\n                                        ctx->device->coopmat_support &&\n                                        ctx->device->coopmat_acc_f16_support &&\n                                        !ctx->device->coopmat_acc_f32_support;\n    const bool y_non_contig = (aivis_f16_only_coopmat && src1->type == GGML_TYPE_F32) ||\n                              (ctx->device->coopmat2 && src1->type == GGML_TYPE_F32) ||\n                              (src0->type == GGML_TYPE_BF16 && src1->type != GGML_TYPE_BF16) ||"
            AIVIS_ANDROID_GGML_VULKAN_SOURCE
            "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}"
        )
        string(REPLACE
            "    const bool y_non_contig = y_decode_vector_staging ||\n                              (ctx->device->coopmat2 && src1->type == GGML_TYPE_F32) ||\n                              (src0->type == GGML_TYPE_BF16 && src1->type != GGML_TYPE_BF16) ||"
            "    const bool aivis_f16_only_coopmat_id = getenv(\"GGML_VK_ALLOW_F16_COOPMAT_ONLY\") != nullptr &&\n                                           ctx->device->coopmat_support &&\n                                           ctx->device->coopmat_acc_f16_support &&\n                                           !ctx->device->coopmat_acc_f32_support;\n    const bool y_non_contig = y_decode_vector_staging ||\n                              (aivis_f16_only_coopmat_id && src1->type == GGML_TYPE_F32) ||\n                              (ctx->device->coopmat2 && src1->type == GGML_TYPE_F32) ||\n                              (src0->type == GGML_TYPE_BF16 && src1->type != GGML_TYPE_BF16) ||"
            AIVIS_ANDROID_GGML_VULKAN_SOURCE
            "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}"
        )
        string(REPLACE
            "static vk_pipeline ggml_vk_guess_matmul_pipeline(ggml_backend_vk_context * ctx, vk_matmul_pipeline& mmp, uint32_t m, uint32_t n, bool aligned, ggml_type src0_type, ggml_type src1_type) {\n    VK_LOG_DEBUG(\"ggml_vk_guess_matmul_pipeline(\" << m << \", \" << n << \", \" << aligned << \", \" << ggml_type_name(src0_type) << \", \" << ggml_type_name(src1_type) << \")\");"
            "static vk_pipeline ggml_vk_guess_matmul_pipeline(ggml_backend_vk_context * ctx, vk_matmul_pipeline& mmp, uint32_t m, uint32_t n, bool aligned, ggml_type src0_type, ggml_type src1_type) {\n    VK_LOG_DEBUG(\"ggml_vk_guess_matmul_pipeline(\" << m << \", \" << n << \", \" << aligned << \", \" << ggml_type_name(src0_type) << \", \" << ggml_type_name(src1_type) << \")\");\n    if (aligned && (mmp->a_l == nullptr || mmp->a_m == nullptr || mmp->a_s == nullptr)) {\n        aligned = false;\n    }"
            AIVIS_ANDROID_GGML_VULKAN_SOURCE
            "${AIVIS_ANDROID_GGML_VULKAN_SOURCE}"
        )
        string(REPLACE
            "static vk_pipeline ggml_vk_guess_matmul_id_pipeline(ggml_backend_vk_context * ctx, vk_matmul_pipeline& mmp, uint32_t m, uint32_t n, bool aligned, ggml_type src0_type, ggml_type src1_type) {\n    VK_LOG_DEBUG(\"ggml_vk_guess_matmul_id_pipeline(\" << m << \", \" << n << \", \" << aligned << \", \" << ggml_type_name(src0_type) << \", \" << ggml_type_name(src1_type) << \")\");"
            "static vk_pipeline ggml_vk_guess_matmul_id_pipeline(ggml_backend_vk_context * ctx, vk_matmul_pipeline& mmp, uint32_t m, uint32_t n, bool aligned, ggml_type src0_type, ggml_type src1_type) {\n    VK_LOG_DEBUG(\"ggml_vk_guess_matmul_id_pipeline(\" << m << \", \" << n << \", \" << aligned << \", \" << ggml_type_name(src0_type) << \", \" << ggml_type_name(src1_type) << \")\");\n    if (aligned && (mmp->a_l == nullptr || mmp->a_m == nullptr || mmp->a_s == nullptr)) {\n        aligned = false;\n    }"
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

add_executable(aivis_vulkan_coopmat_probe
    src/aivis_vulkan_coopmat_probe.cpp
)
target_link_libraries(aivis_vulkan_coopmat_probe PRIVATE Vulkan::Vulkan)
