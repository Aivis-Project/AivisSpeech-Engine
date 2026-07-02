#include <vulkan/vulkan.h>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const char * component_type_name(VkComponentTypeKHR type) {
    switch (type) {
    case VK_COMPONENT_TYPE_FLOAT16_KHR:
        return "float16";
    case VK_COMPONENT_TYPE_FLOAT32_KHR:
        return "float32";
    case VK_COMPONENT_TYPE_FLOAT64_KHR:
        return "float64";
    case VK_COMPONENT_TYPE_SINT8_KHR:
        return "sint8";
    case VK_COMPONENT_TYPE_SINT16_KHR:
        return "sint16";
    case VK_COMPONENT_TYPE_SINT32_KHR:
        return "sint32";
    case VK_COMPONENT_TYPE_SINT64_KHR:
        return "sint64";
    case VK_COMPONENT_TYPE_UINT8_KHR:
        return "uint8";
    case VK_COMPONENT_TYPE_UINT16_KHR:
        return "uint16";
    case VK_COMPONENT_TYPE_UINT32_KHR:
        return "uint32";
    case VK_COMPONENT_TYPE_UINT64_KHR:
        return "uint64";
    default:
        return "unknown";
    }
}

const char * scope_name(VkScopeKHR scope) {
    switch (scope) {
    case VK_SCOPE_DEVICE_KHR:
        return "device";
    case VK_SCOPE_WORKGROUP_KHR:
        return "workgroup";
    case VK_SCOPE_SUBGROUP_KHR:
        return "subgroup";
    case VK_SCOPE_QUEUE_FAMILY_KHR:
        return "queue_family";
    default:
        return "unknown";
    }
}

std::string stage_flags(VkShaderStageFlags flags) {
    std::string out;
    auto append = [&](VkShaderStageFlagBits bit, const char * name) {
        if ((flags & bit) == 0) {
            return;
        }
        if (!out.empty()) {
            out += "|";
        }
        out += name;
    };
    append(VK_SHADER_STAGE_COMPUTE_BIT, "compute");
    append(VK_SHADER_STAGE_VERTEX_BIT, "vertex");
    append(VK_SHADER_STAGE_FRAGMENT_BIT, "fragment");
    append(VK_SHADER_STAGE_ALL_GRAPHICS, "all_graphics");
    append(VK_SHADER_STAGE_ALL, "all");
    if (out.empty()) {
        out = "none";
    }
    return out;
}

bool has_extension(VkPhysicalDevice device, const char * name) {
    uint32_t count = 0;
    VkResult result = vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("vkEnumerateDeviceExtensionProperties count failed");
    }
    std::vector<VkExtensionProperties> extensions(count);
    result = vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions.data());
    if (result != VK_SUCCESS) {
        throw std::runtime_error("vkEnumerateDeviceExtensionProperties failed");
    }
    for (const VkExtensionProperties & extension : extensions) {
        if (std::strcmp(extension.extensionName, name) == 0) {
            return true;
        }
    }
    return false;
}

uint32_t choose_instance_api_version() {
    uint32_t available = VK_API_VERSION_1_0;
    auto enumerate_instance_version =
        reinterpret_cast<PFN_vkEnumerateInstanceVersion>(vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"));
    if (enumerate_instance_version) {
        enumerate_instance_version(&available);
    }
    return available >= VK_API_VERSION_1_3 ? VK_API_VERSION_1_3 : available;
}

} // namespace

int main() {
    try {
        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "aivis-vulkan-coopmat-probe";
        app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.pEngineName = "aivis";
        app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.apiVersion = choose_instance_api_version();

        VkInstanceCreateInfo instance_info{};
        instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instance_info.pApplicationInfo = &app_info;

        VkInstance instance = VK_NULL_HANDLE;
        VkResult result = vkCreateInstance(&instance_info, nullptr, &instance);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("vkCreateInstance failed: " + std::to_string(result));
        }

        uint32_t device_count = 0;
        result = vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("vkEnumeratePhysicalDevices count failed");
        }
        std::vector<VkPhysicalDevice> devices(device_count);
        result = vkEnumeratePhysicalDevices(instance, &device_count, devices.data());
        if (result != VK_SUCCESS) {
            throw std::runtime_error("vkEnumeratePhysicalDevices failed");
        }

        auto get_coopmat_properties =
            reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
                vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));

        std::cout << "physical_devices=" << device_count << "\n";
        for (uint32_t i = 0; i < device_count; ++i) {
            VkPhysicalDevice device = devices[i];

            VkPhysicalDeviceCooperativeMatrixPropertiesKHR coopmat_device_props{};
            coopmat_device_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
            VkPhysicalDeviceProperties2 props2{};
            props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props2.pNext = &coopmat_device_props;
            vkGetPhysicalDeviceProperties2(device, &props2);

            const bool extension_present = has_extension(device, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
            VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopmat_features{};
            coopmat_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
            VkPhysicalDeviceFeatures2 features2{};
            features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            if (extension_present) {
                features2.pNext = &coopmat_features;
            }
            vkGetPhysicalDeviceFeatures2(device, &features2);

            std::cout << "device[" << i << "] name=\"" << props2.properties.deviceName
                      << "\" vendor=0x" << std::hex << props2.properties.vendorID
                      << " device=0x" << props2.properties.deviceID << std::dec
                      << " api=" << VK_VERSION_MAJOR(props2.properties.apiVersion) << "."
                      << VK_VERSION_MINOR(props2.properties.apiVersion) << "."
                      << VK_VERSION_PATCH(props2.properties.apiVersion) << "\n";
            std::cout << "  " << VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME
                      << "=" << (extension_present ? "yes" : "no")
                      << " feature.cooperativeMatrix=" << coopmat_features.cooperativeMatrix
                      << " robustBufferAccess=" << coopmat_features.cooperativeMatrixRobustBufferAccess
                      << " supportedStages=" << stage_flags(coopmat_device_props.cooperativeMatrixSupportedStages)
                      << "\n";

            if (!extension_present || !coopmat_features.cooperativeMatrix || !get_coopmat_properties) {
                continue;
            }

            uint32_t property_count = 0;
            result = get_coopmat_properties(device, &property_count, nullptr);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR count failed: " + std::to_string(result));
            }
            std::vector<VkCooperativeMatrixPropertiesKHR> properties(property_count);
            for (VkCooperativeMatrixPropertiesKHR & property : properties) {
                property.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
            }
            result = get_coopmat_properties(device, &property_count, properties.data());
            if (result != VK_SUCCESS) {
                throw std::runtime_error("vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR failed: " + std::to_string(result));
            }

            std::cout << "  cooperative_matrix_properties=" << property_count << "\n";
            for (uint32_t p = 0; p < property_count; ++p) {
                const VkCooperativeMatrixPropertiesKHR & property = properties[p];
                const bool ggml_f16_f32acc =
                    property.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                    property.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                    property.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    property.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    property.scope == VK_SCOPE_SUBGROUP_KHR;
                const bool ggml_f16_f16acc =
                    property.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                    property.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                    property.CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                    property.ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                    property.scope == VK_SCOPE_SUBGROUP_KHR;
                const bool ggml_f32_f32 =
                    property.AType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    property.BType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    property.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    property.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                    property.scope == VK_SCOPE_SUBGROUP_KHR;
                const bool ggml_s8_s32 =
                    property.AType == VK_COMPONENT_TYPE_SINT8_KHR &&
                    property.BType == VK_COMPONENT_TYPE_SINT8_KHR &&
                    property.CType == VK_COMPONENT_TYPE_SINT32_KHR &&
                    property.ResultType == VK_COMPONENT_TYPE_SINT32_KHR &&
                    property.scope == VK_SCOPE_SUBGROUP_KHR;

                std::cout << "    [" << p << "] "
                          << "M=" << property.MSize
                          << " N=" << property.NSize
                          << " K=" << property.KSize
                          << " A=" << component_type_name(property.AType)
                          << " B=" << component_type_name(property.BType)
                          << " C=" << component_type_name(property.CType)
                          << " Result=" << component_type_name(property.ResultType)
                          << " saturatingAccumulation=" << property.saturatingAccumulation
                          << " scope=" << scope_name(property.scope)
                          << " ggml_f16_f32acc=" << (ggml_f16_f32acc ? "yes" : "no")
                          << " ggml_f16_f16acc=" << (ggml_f16_f16acc ? "yes" : "no")
                          << " ggml_f32_f32=" << (ggml_f32_f32 ? "yes" : "no")
                          << " ggml_s8_s32=" << (ggml_s8_s32 ? "yes" : "no")
                          << "\n";
            }
        }

        vkDestroyInstance(instance, nullptr);
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
