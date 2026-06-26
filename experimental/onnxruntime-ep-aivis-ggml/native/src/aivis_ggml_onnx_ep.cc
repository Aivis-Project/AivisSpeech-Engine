#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <memory>
#include <new>
#include <stdexcept>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#if defined(_WIN32)
#define AIVIS_GGML_EP_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define AIVIS_GGML_EP_EXPORT __attribute__((visibility("default")))
#else
#define AIVIS_GGML_EP_EXPORT
#endif

namespace {

constexpr const char* kEpName = "AivisGgmlExecutionProvider";
constexpr const char* kVendor = "Aivis Project";
constexpr const char* kVersion = "0.1.0";
constexpr const char* kStage = "synthesis-jp-bert-bridge";
constexpr const char* kRuntimeRegistryContract = "aivis-ggml-runtime-registry-v1";
constexpr const char* kTtsCppRuntimeContract = "tts-style-bert-vits2-c-api-v1";
constexpr const char* kSignatureContract = "aivis-ggml-signature-contract-v1";
constexpr const char* kOfficialEpContextVersion = "aivis-ggml-official-ep-context-v1";
constexpr const char* kCompiledModelCompatibilityVersion = "aivis-ggml-compiled-model-compatibility-v1";
constexpr uint32_t kExpectedTtsCppRuntimeAbiVersion = 1;
constexpr uint32_t kExpectedTtsCppGgufSchemaVersion = 1;
constexpr int64_t kExpectedIrVersion = 8;
constexpr int64_t kExpectedOpsetVersion = 18;
constexpr const char* kExpectedGraphName = "main_graph";
constexpr size_t kExpectedOutputCount = 7;
constexpr const char* kExpectedFirstOutputName = "output";
constexpr size_t kExpectedNodeCount = 5334;
constexpr size_t kExpectedOptimizedNodeCount = 5332;
constexpr size_t kExpectedInitializerCount = 948;
constexpr size_t kExpectedOptimizedInitializerCount = 949;
constexpr const char* kDefaultBackend = "vulkan";
constexpr const char* kDefaultPrecision = "accurate";
constexpr int64_t kExpectedJpBertOpsetVersion = 17;
constexpr size_t kExpectedJpBertInputCount = 2;
constexpr size_t kExpectedJpBertOutputCount = 1;
constexpr size_t kExpectedJpBertNodeCount = 3619;
constexpr size_t kExpectedJpBertInitializerCount = 432;
constexpr std::array<size_t, 3> kAcceptedJpBertNodeCounts = {3619, 3092, 3180};
constexpr std::array<size_t, 3> kAcceptedJpBertInitializerCounts = {432, 521, 543};
constexpr const char* kExpectedJpBertOutputName = "output";

constexpr std::array<const char*, 11> kExpectedInputNames = {
    "x_tst",
    "x_tst_lengths",
    "sid",
    "tones",
    "language",
    "bert",
    "style_vec",
    "length_scale",
    "sdp_ratio",
    "noise_scale",
    "noise_scale_w",
};

constexpr std::array<ONNXTensorElementDataType, 11> kExpectedInputTypes = {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
};

constexpr std::array<const char*, 4> kRequiredInitializerNames = {
    "enc_p.emb.weight",
    "enc_p.tone_emb.weight",
    "enc_p.language_emb.weight",
    "enc_p.bert_proj.weight",
};

constexpr std::array<const char*, 5> kRequiredOpTypes = {
    "Conv",
    "Gather",
    "MatMul",
    "RandomNormalLike",
    "Tanh",
};

constexpr std::array<const char*, 2> kExpectedJpBertInputNames = {
    "input_ids",
    "attention_mask",
};

constexpr std::array<ONNXTensorElementDataType, 2> kExpectedJpBertInputTypes = {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
};

constexpr std::array<const char*, 5> kRequiredJpBertOpTypes = {
    "Gather",
    "LayerNormalization",
    "MatMul",
    "Reshape",
    "Where",
};

OrtStatus* CreateStatus(const OrtApi& api, OrtErrorCode code, const char* message) noexcept {
  return api.CreateStatus(code, message);
}

std::string ToLowerAscii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

void LogMessage(
    const OrtApi& api,
    const OrtLogger* logger,
    OrtLoggingLevel level,
    const std::string& message) noexcept {
  if (logger == nullptr) {
    return;
  }

  OrtStatus* status = api.Logger_LogMessage(
      logger,
      level,
      message.c_str(),
      ORT_FILE,
      __LINE__,
      __FUNCTION__);
  if (status != nullptr) {
    api.ReleaseStatus(status);
  }
}

bool IsTraceEnabled() {
  const char* value = std::getenv("AIVIS_GGML_EP_TRACE");
  return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

void TraceMessage(const std::string& message) {
  if (IsTraceEnabled()) {
    std::cerr << "AIVIS_GGML_EP_TRACE " << message << std::endl;
  }
}

const char* TensorElementTypeName(ONNXTensorElementDataType type) noexcept {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return "FLOAT";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return "INT64";
    default:
      return "OTHER";
  }
}

ONNXTensorElementDataType TensorElementType(Ort::ConstValueInfo value_info) {
  Ort::ConstTypeInfo type_info = value_info.TypeInfo();
  if (type_info.GetONNXType() != ONNX_TYPE_TENSOR) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
  return type_info.GetTensorTypeAndShapeInfo().GetElementType();
}

std::string JoinReasons(const std::vector<std::string>& reasons) {
  std::ostringstream out;
  for (size_t i = 0; i < reasons.size(); ++i) {
    if (i != 0) {
      out << "; ";
    }
    out << reasons[i];
  }
  return out.str();
}

bool IsAcceptedGraphName(const std::string& graph_name) {
  return graph_name == kExpectedGraphName || graph_name.rfind(kEpName, 0) == 0;
}

template <size_t N>
std::vector<size_t> BuildInputIndices(
    const OrtGraph* ort_graph,
    const std::array<const char*, N>& expected_input_names) {
  Ort::ConstGraph graph{ort_graph};
  const std::vector<Ort::ConstValueInfo> inputs = graph.GetInputs();
  std::vector<size_t> indices;
  indices.reserve(N);
  for (const char* expected_name : expected_input_names) {
    bool found = false;
    for (size_t index = 0; index < inputs.size(); ++index) {
      if (inputs[index].GetName() == expected_name) {
        indices.push_back(index);
        found = true;
        break;
      }
    }
    if (!found) {
      throw std::runtime_error(std::string("compiled graph input is missing: ") + expected_name);
    }
  }
  return indices;
}

size_t BuildOutputIndex(const OrtGraph* ort_graph, const char* expected_output_name) {
  Ort::ConstGraph graph{ort_graph};
  const std::vector<Ort::ConstValueInfo> outputs = graph.GetOutputs();
  for (size_t index = 0; index < outputs.size(); ++index) {
    if (outputs[index].GetName() == expected_output_name) {
      return index;
    }
  }
  throw std::runtime_error(std::string("compiled graph output is missing: ") + expected_output_name);
}

struct AivisGgmlEpConfig {
  std::string backend = kDefaultBackend;
  std::string device;
  std::string precision = kDefaultPrecision;
  std::string cache_dir;
  std::string cache_manifest_path;
  std::string gguf_path;
  std::string jp_bert_gguf_path;
  std::string tts_cpp_library_path;
  bool eager_load_model = false;
  bool claim_synthesis_graph = false;
  bool claim_jp_bert_graph = false;
  bool ort_ep_context_enable = false;
  bool ort_ep_context_embed_mode = false;
  std::string ort_ep_context_file_path;
  std::string ort_ep_context_node_name_prefix;
  int n_threads = 0;
};

std::string ReadSessionOption(
    const OrtSessionOptions* session_options,
    const std::string& option_name,
    const std::string& default_value) {
  if (session_options == nullptr) {
    return default_value;
  }
  Ort::ConstSessionOptions options{session_options};
  if (options.HasConfigEntry(option_name.c_str())) {
    return options.GetConfigEntry(option_name.c_str());
  }
  return default_value;
}

std::string ReadEpOption(
    const OrtSessionOptions* session_options,
    const std::string& option_name,
    const std::string& default_value) {
  if (session_options == nullptr) {
    return default_value;
  }

  Ort::ConstSessionOptions options{session_options};
  const std::string ep_name = kEpName;
  const std::string lower_ep_name = ToLowerAscii(ep_name);
  const std::array<std::string, 4> candidates = {
      "ep." + lower_ep_name + "." + option_name,
      "ep." + ep_name + "." + option_name,
      "ep_factory." + lower_ep_name + "." + option_name,
      "ep_factory." + ep_name + "." + option_name,
  };

  for (const std::string& candidate : candidates) {
    if (options.HasConfigEntry(candidate.c_str())) {
      return options.GetConfigEntry(candidate.c_str());
    }
  }
  return default_value;
}

bool ParseBoolOption(const std::string& value, bool default_value, const char* option_name) {
  if (value.empty()) {
    return default_value;
  }
  const std::string lowered = ToLowerAscii(value);
  if (lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on") {
    return true;
  }
  if (lowered == "0" || lowered == "false" || lowered == "no" || lowered == "off") {
    return false;
  }
  throw std::invalid_argument(std::string("Invalid boolean value for ") + option_name + ": " + value);
}

int ParseNonNegativeIntOption(const std::string& value, int default_value, const char* option_name) {
  if (value.empty()) {
    return default_value;
  }
  size_t consumed = 0;
  unsigned long parsed = 0;
  try {
    parsed = std::stoul(value, &consumed, 10);
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string("Invalid integer value for ") + option_name + ": " + value);
  }
  if (consumed != value.size() || parsed > static_cast<unsigned long>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument(std::string("Invalid integer value for ") + option_name + ": " + value);
  }
  return static_cast<int>(parsed);
}

AivisGgmlEpConfig ReadEpConfig(const OrtSessionOptions* session_options) {
  AivisGgmlEpConfig config;
  config.backend = ReadEpOption(session_options, "backend", kDefaultBackend);
  config.device = ReadEpOption(session_options, "device", "");
  config.precision = ReadEpOption(session_options, "precision", kDefaultPrecision);
  config.cache_dir = ReadEpOption(session_options, "cache_dir", "");
  config.cache_manifest_path = ReadEpOption(session_options, "cache_manifest_path", "");
  config.gguf_path = ReadEpOption(session_options, "gguf_path", "");
  config.jp_bert_gguf_path = ReadEpOption(session_options, "jp_bert_gguf_path", "");
  config.tts_cpp_library_path = ReadEpOption(session_options, "tts_cpp_library_path", "");
  config.eager_load_model = ParseBoolOption(
      ReadEpOption(session_options, "eager_load_model", "0"),
      false,
      "eager_load_model");
  config.claim_synthesis_graph = ParseBoolOption(
      ReadEpOption(session_options, "claim_synthesis_graph", "0"),
      false,
      "claim_synthesis_graph");
  config.claim_jp_bert_graph = ParseBoolOption(
      ReadEpOption(session_options, "claim_jp_bert_graph", "0"),
      false,
      "claim_jp_bert_graph");
  config.ort_ep_context_enable = ParseBoolOption(
      ReadSessionOption(session_options, "ep.context_enable", "0"),
      false,
      "ep.context_enable");
  config.ort_ep_context_file_path = ReadSessionOption(
      session_options,
      "ep.context_file_path",
      "");
  config.ort_ep_context_embed_mode = ParseBoolOption(
      ReadSessionOption(session_options, "ep.context_embed_mode", "0"),
      false,
      "ep.context_embed_mode");
  config.ort_ep_context_node_name_prefix = ReadSessionOption(
      session_options,
      "ep.context_node_name_prefix",
      "");
  config.n_threads = ParseNonNegativeIntOption(
      ReadEpOption(session_options, "n_threads", "0"),
      0,
      "n_threads");
  return config;
}

bool IsSupportedBackend(const std::string& backend) {
  return backend == "vulkan" || backend == "metal" || backend == "cpu";
}

bool IsSupportedPrecision(const std::string& precision) {
  return precision == "accurate" || precision == "fast";
}

bool PathExists(const std::string& path) {
  if (path.empty()) {
    return false;
  }
  std::error_code error;
  return std::filesystem::exists(std::filesystem::path(path), error) && !error;
}

std::string ReadSmallTextFile(const std::string& path, size_t max_bytes) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("could not open file: " + path);
  }
  std::string data;
  data.resize(max_bytes + 1);
  file.read(data.data(), static_cast<std::streamsize>(data.size()));
  const std::streamsize bytes_read = file.gcount();
  if (bytes_read > static_cast<std::streamsize>(max_bytes)) {
    throw std::runtime_error("file is too large for bootstrap validation: " + path);
  }
  data.resize(static_cast<size_t>(bytes_read));
  return data;
}

bool ManifestContainsReadyCache(const std::string& manifest_text) {
  return manifest_text.find("aivis-ggml-onnx-cache-v1") != std::string::npos &&
         manifest_text.find("\"status\": \"ready\"") != std::string::npos &&
         manifest_text.find("\"can_write_gguf\": true") != std::string::npos;
}

OrtStatus* ValidateEpConfig(const OrtApi& api, const AivisGgmlEpConfig& config) noexcept {
  if (!IsSupportedBackend(config.backend)) {
    return CreateStatus(
        api,
        ORT_INVALID_ARGUMENT,
        "AivisGgmlExecutionProvider option backend must be one of: vulkan, metal, cpu.");
  }
  if (!IsSupportedPrecision(config.precision)) {
    return CreateStatus(
        api,
        ORT_INVALID_ARGUMENT,
        "AivisGgmlExecutionProvider option precision must be one of: accurate, fast.");
  }
  try {
    if (!config.cache_manifest_path.empty()) {
      if (!PathExists(config.cache_manifest_path)) {
        return CreateStatus(
            api,
            ORT_INVALID_ARGUMENT,
            "AivisGgmlExecutionProvider option cache_manifest_path does not exist.");
      }
      const std::string manifest = ReadSmallTextFile(config.cache_manifest_path, 1024 * 1024);
      if (!ManifestContainsReadyCache(manifest)) {
        return CreateStatus(
            api,
            ORT_INVALID_ARGUMENT,
            "AivisGgmlExecutionProvider cache_manifest_path is not a ready Aivis GGML cache manifest.");
      }
    }
    if (config.eager_load_model) {
      if (config.tts_cpp_library_path.empty() || !PathExists(config.tts_cpp_library_path)) {
        return CreateStatus(
            api,
            ORT_INVALID_ARGUMENT,
            "AivisGgmlExecutionProvider eager_load_model requires an existing tts_cpp_library_path.");
      }
      if (config.gguf_path.empty() && config.jp_bert_gguf_path.empty()) {
        return CreateStatus(
            api,
            ORT_INVALID_ARGUMENT,
            "AivisGgmlExecutionProvider eager_load_model requires gguf_path or jp_bert_gguf_path.");
      }
      if (!config.gguf_path.empty() && !PathExists(config.gguf_path)) {
        return CreateStatus(
            api,
            ORT_INVALID_ARGUMENT,
            "AivisGgmlExecutionProvider option gguf_path does not exist.");
      }
      if (!config.jp_bert_gguf_path.empty() && !PathExists(config.jp_bert_gguf_path)) {
        return CreateStatus(
            api,
            ORT_INVALID_ARGUMENT,
            "AivisGgmlExecutionProvider option jp_bert_gguf_path does not exist.");
      }
    }
    if ((config.claim_synthesis_graph || config.claim_jp_bert_graph) &&
        !config.eager_load_model &&
        config.tts_cpp_library_path.empty()) {
      return CreateStatus(
          api,
          ORT_INVALID_ARGUMENT,
          "AivisGgmlExecutionProvider graph claim requires either eager_load_model=1 "
          "or tts_cpp_library_path for EPContext lazy restore.");
    }
    if ((config.claim_synthesis_graph || config.claim_jp_bert_graph) &&
        !config.eager_load_model &&
        !PathExists(config.tts_cpp_library_path)) {
      return CreateStatus(
          api,
          ORT_INVALID_ARGUMENT,
          "AivisGgmlExecutionProvider graph claim requires an existing tts_cpp_library_path.");
    }
    if (config.claim_synthesis_graph && config.eager_load_model && config.gguf_path.empty()) {
      return CreateStatus(
          api,
          ORT_INVALID_ARGUMENT,
          "AivisGgmlExecutionProvider claim_synthesis_graph with eager_load_model=1 requires gguf_path.");
    }
    if (config.claim_jp_bert_graph && config.eager_load_model && config.jp_bert_gguf_path.empty()) {
      return CreateStatus(
          api,
          ORT_INVALID_ARGUMENT,
          "AivisGgmlExecutionProvider claim_jp_bert_graph with eager_load_model=1 requires jp_bert_gguf_path.");
    }
  } catch (const std::exception& ex) {
    return CreateStatus(api, ORT_INVALID_ARGUMENT, ex.what());
  }
  return nullptr;
}

std::string ConfigSummary(const AivisGgmlEpConfig& config) {
  std::ostringstream out;
  out << "backend=" << config.backend
      << ", device=" << (config.device.empty() ? "default" : config.device)
      << ", precision=" << config.precision
      << ", cache_dir_set=" << (config.cache_dir.empty() ? "false" : "true")
      << ", cache_manifest_set=" << (config.cache_manifest_path.empty() ? "false" : "true")
      << ", gguf_path_set=" << (config.gguf_path.empty() ? "false" : "true")
      << ", jp_bert_gguf_path_set=" << (config.jp_bert_gguf_path.empty() ? "false" : "true")
      << ", tts_cpp_library_set=" << (config.tts_cpp_library_path.empty() ? "false" : "true")
      << ", eager_load_model=" << (config.eager_load_model ? "true" : "false")
      << ", claim_synthesis_graph=" << (config.claim_synthesis_graph ? "true" : "false")
      << ", claim_jp_bert_graph=" << (config.claim_jp_bert_graph ? "true" : "false")
      << ", ort_ep_context_enable=" << (config.ort_ep_context_enable ? "true" : "false")
      << ", ort_ep_context_embed_mode=" << (config.ort_ep_context_embed_mode ? "true" : "false")
      << ", n_threads=" << config.n_threads;
  return out.str();
}

class DynamicLibrary final {
 public:
  static std::unique_ptr<DynamicLibrary> Load(const std::string& path) {
#if defined(_WIN32)
    HMODULE handle = LoadLibraryA(path.c_str());
    if (handle == nullptr) {
      throw std::runtime_error("could not load dynamic library: " + path);
    }
#else
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      const char* error = dlerror();
      throw std::runtime_error(
          std::string("could not load dynamic library: ") + path +
          (error != nullptr ? std::string(": ") + error : std::string()));
    }
#endif
    return std::unique_ptr<DynamicLibrary>(new DynamicLibrary(handle));
  }

  DynamicLibrary(const DynamicLibrary&) = delete;
  DynamicLibrary& operator=(const DynamicLibrary&) = delete;

  ~DynamicLibrary() {
#if defined(_WIN32)
    if (handle_ != nullptr) {
      FreeLibrary(static_cast<HMODULE>(handle_));
    }
#else
    if (handle_ != nullptr) {
      dlclose(handle_);
    }
#endif
  }

  void* Symbol(const char* name) const {
#if defined(_WIN32)
    void* symbol = reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
    dlerror();
    void* symbol = dlsym(handle_, name);
#endif
    if (symbol == nullptr) {
      throw std::runtime_error(std::string("missing TTS.cpp symbol: ") + name);
    }
    return symbol;
  }

  void* OptionalSymbol(const char* name) const noexcept {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
    dlerror();
    return dlsym(handle_, name);
#endif
  }

 private:
#if defined(_WIN32)
  explicit DynamicLibrary(HMODULE handle) : handle_(handle) {}
  HMODULE handle_;
#else
  explicit DynamicLibrary(void* handle) : handle_(handle) {}
  void* handle_;
#endif
};

struct tts_style_bert_vits2_handle;
struct tts_style_bert_vits2_jp_bert_handle;

struct tts_style_bert_vits2_float_buffer {
  const float* data;
  size_t length;
  uint32_t hidden_size;
  float sample_rate;
};

size_t CheckedElementCount(const std::vector<int64_t>& shape, const std::string& name) {
  size_t count = 1;
  for (int64_t dim : shape) {
    if (dim < 0) {
      throw std::runtime_error(name + " has a dynamic runtime dimension.");
    }
    if (dim != 0 && count > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
      throw std::runtime_error(name + " element count overflows size_t.");
    }
    count *= static_cast<size_t>(dim);
  }
  return count;
}

std::vector<int64_t> TensorShape(Ort::ConstValue value) {
  return value.GetTensorTypeAndShapeInfo().GetShape();
}

std::string ShapeString(const std::vector<int64_t>& shape) {
  std::ostringstream out;
  out << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << shape[i];
  }
  out << "]";
  return out.str();
}

void ExpectTensorElementType(
    Ort::ConstValue value,
    ONNXTensorElementDataType expected_type,
    const std::string& name) {
  const ONNXTensorElementDataType actual_type =
      value.GetTensorTypeAndShapeInfo().GetElementType();
  if (actual_type != expected_type) {
    throw std::runtime_error(
        name + " has element type " + TensorElementTypeName(actual_type) +
        ", expected " + TensorElementTypeName(expected_type) + ".");
  }
}

size_t ExpectTokenMatrix(
    Ort::ConstValue value,
    ONNXTensorElementDataType expected_type,
    const std::string& name) {
  if (value == nullptr) {
    throw std::runtime_error(name + " input is missing.");
  }
  ExpectTensorElementType(value, expected_type, name);
  const std::vector<int64_t> shape = TensorShape(value);
  if (shape.size() != 2 || shape[0] != 1 || shape[1] <= 0) {
    throw std::runtime_error(name + " must have shape [1, tokens], got " + ShapeString(shape) + ".");
  }
  return static_cast<size_t>(shape[1]);
}

std::vector<int32_t> CopyInt64TokenInput(
    Ort::ConstValue value,
    size_t expected_tokens,
    const std::string& name) {
  const size_t tokens = ExpectTokenMatrix(value, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, name);
  if (tokens != expected_tokens) {
    throw std::runtime_error(name + " token count does not match x_tst.");
  }
  const int64_t* source = value.GetTensorData<int64_t>();
  std::vector<int32_t> output(tokens);
  for (size_t i = 0; i < tokens; ++i) {
    if (source[i] < std::numeric_limits<int32_t>::min() ||
        source[i] > std::numeric_limits<int32_t>::max()) {
      throw std::runtime_error(name + " contains a value outside int32 range.");
    }
    output[i] = static_cast<int32_t>(source[i]);
  }
  return output;
}

int32_t ReadInt64ScalarVectorInput(Ort::ConstValue value, const std::string& name) {
  if (value == nullptr) {
    throw std::runtime_error(name + " input is missing.");
  }
  ExpectTensorElementType(value, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, name);
  const std::vector<int64_t> shape = TensorShape(value);
  if (CheckedElementCount(shape, name) != 1) {
    throw std::runtime_error(name + " must contain exactly one scalar value.");
  }
  const int64_t raw_value = value.GetTensorData<int64_t>()[0];
  if (raw_value < std::numeric_limits<int32_t>::min() ||
      raw_value > std::numeric_limits<int32_t>::max()) {
    throw std::runtime_error(name + " is outside int32 range.");
  }
  return static_cast<int32_t>(raw_value);
}

float ReadFloatScalarInput(Ort::ConstValue value, const std::string& name) {
  if (value == nullptr) {
    throw std::runtime_error(name + " input is missing.");
  }
  ExpectTensorElementType(value, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, name);
  const std::vector<int64_t> shape = TensorShape(value);
  if (CheckedElementCount(shape, name) != 1) {
    throw std::runtime_error(name + " must contain exactly one scalar value.");
  }
  return value.GetTensorData<float>()[0];
}

const float* ExpectBertInput(Ort::ConstValue value, size_t tokens) {
  if (value == nullptr) {
    throw std::runtime_error("bert input is missing.");
  }
  ExpectTensorElementType(value, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "bert");
  const std::vector<int64_t> shape = TensorShape(value);
  if (shape.size() != 3 || shape[0] != 1 || shape[1] != 1024 ||
      shape[2] != static_cast<int64_t>(tokens)) {
    throw std::runtime_error("bert must have shape [1, 1024, tokens], got " + ShapeString(shape) + ".");
  }
  return value.GetTensorData<float>();
}

const float* ExpectStyleVecInput(Ort::ConstValue value) {
  if (value == nullptr) {
    throw std::runtime_error("style_vec input is missing.");
  }
  ExpectTensorElementType(value, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "style_vec");
  const std::vector<int64_t> shape = TensorShape(value);
  if (shape.size() != 2 || shape[0] != 1 || shape[1] != 256) {
    throw std::runtime_error("style_vec must have shape [1, 256], got " + ShapeString(shape) + ".");
  }
  return value.GetTensorData<float>();
}

class TtsCppRuntime final {
 public:
  using LastErrorFn = const char* (*)();
  using Uint32Fn = uint32_t (*)();
  using LoadModelFn = int (*)(
      const char* model_path,
      int n_threads,
      int cpu_only,
      tts_style_bert_vits2_handle** out_handle);
  using FreeModelFn = void (*)(tts_style_bert_vits2_handle* handle);
  using LoadJpBertModelFn = int (*)(
      const char* model_path,
      int n_threads,
      int cpu_only,
      tts_style_bert_vits2_jp_bert_handle** out_handle);
  using FreeJpBertModelFn = void (*)(tts_style_bert_vits2_jp_bert_handle* handle);
  using SynthesizeFrontFn = int (*)(
      tts_style_bert_vits2_handle* handle,
      const int32_t* phone_ids,
      const int32_t* tone_ids,
      const int32_t* language_ids,
      size_t tokens,
      const float* bert,
      size_t bert_length,
      int32_t speaker_id,
      int32_t style_id,
      float style_weight,
      float sdp_ratio,
      float length_scale,
      float noise_scale,
      float noise_w_scale,
      tts_style_bert_vits2_float_buffer* out_audio);
  using SynthesizeFrontWithStyleVecFn = int (*)(
      tts_style_bert_vits2_handle* handle,
      const int32_t* phone_ids,
      const int32_t* tone_ids,
      const int32_t* language_ids,
      size_t tokens,
      const float* bert,
      size_t bert_length,
      const float* style_vec,
      size_t style_vec_length,
      int32_t speaker_id,
      float sdp_ratio,
      float length_scale,
      float noise_scale,
      float noise_w_scale,
      tts_style_bert_vits2_float_buffer* out_audio);
  using EncodeJpBertFeaturesFn = int (*)(
      tts_style_bert_vits2_jp_bert_handle* handle,
      const int32_t* input_ids,
      size_t tokens,
      tts_style_bert_vits2_float_buffer* out_features);

  static std::shared_ptr<TtsCppRuntime> LoadAndMaybeOpenModel(const AivisGgmlEpConfig& config) {
    auto library = DynamicLibrary::Load(config.tts_cpp_library_path);
    auto runtime = std::shared_ptr<TtsCppRuntime>(new TtsCppRuntime(std::move(library)));
    runtime->last_error_ = reinterpret_cast<LastErrorFn>(
        runtime->library_->Symbol("tts_style_bert_vits2_last_error"));
    runtime->load_model_ = reinterpret_cast<LoadModelFn>(
        runtime->library_->Symbol("tts_style_bert_vits2_load_model"));
    runtime->free_model_ = reinterpret_cast<FreeModelFn>(
        runtime->library_->Symbol("tts_style_bert_vits2_free_model"));
    runtime->load_jp_bert_model_ = reinterpret_cast<LoadJpBertModelFn>(
        runtime->library_->Symbol("tts_style_bert_vits2_jp_bert_load_model"));
    runtime->free_jp_bert_model_ = reinterpret_cast<FreeJpBertModelFn>(
        runtime->library_->Symbol("tts_style_bert_vits2_jp_bert_free_model"));
    runtime->synthesize_front_ = reinterpret_cast<SynthesizeFrontFn>(
        runtime->library_->Symbol("tts_style_bert_vits2_synthesize_front"));
    runtime->synthesize_front_with_style_vec_ = reinterpret_cast<SynthesizeFrontWithStyleVecFn>(
        runtime->library_->Symbol("tts_style_bert_vits2_synthesize_front_with_style_vec"));
    runtime->encode_jp_bert_features_ = reinterpret_cast<EncodeJpBertFeaturesFn>(
        runtime->library_->Symbol("tts_style_bert_vits2_jp_bert_encode_features"));
    runtime->runtime_abi_version_ = reinterpret_cast<Uint32Fn>(
        runtime->library_->OptionalSymbol("tts_style_bert_vits2_runtime_abi_version"));
    runtime->gguf_schema_version_ = reinterpret_cast<Uint32Fn>(
        runtime->library_->OptionalSymbol("tts_style_bert_vits2_gguf_schema_version"));
    runtime->ValidateOptionalVersionSymbols();

    const int cpu_only = config.backend == "cpu" ? 1 : 0;
    if (!config.gguf_path.empty()) {
      tts_style_bert_vits2_handle* handle = nullptr;
      if (runtime->load_model_(config.gguf_path.c_str(), config.n_threads, cpu_only, &handle) == 0 || handle == nullptr) {
        const char* detail = runtime->last_error_ != nullptr ? runtime->last_error_() : nullptr;
        throw std::runtime_error(
            std::string("TTS.cpp failed to load Style-Bert-VITS2 GGUF") +
            (detail != nullptr && detail[0] ? std::string(": ") + detail : std::string(".")));
      }
      runtime->model_handle_ = handle;
    }
    if (!config.jp_bert_gguf_path.empty()) {
      tts_style_bert_vits2_jp_bert_handle* handle = nullptr;
      if (runtime->load_jp_bert_model_(config.jp_bert_gguf_path.c_str(), config.n_threads, cpu_only, &handle) == 0 ||
          handle == nullptr) {
        const char* detail = runtime->last_error_ != nullptr ? runtime->last_error_() : nullptr;
        throw std::runtime_error(
            std::string("TTS.cpp failed to load Style-Bert-VITS2 JP-BERT GGUF") +
            (detail != nullptr && detail[0] ? std::string(": ") + detail : std::string(".")));
      }
      runtime->jp_bert_handle_ = handle;
    }
    return runtime;
  }

  TtsCppRuntime(const TtsCppRuntime&) = delete;
  TtsCppRuntime& operator=(const TtsCppRuntime&) = delete;

  ~TtsCppRuntime() {
    if (jp_bert_handle_ != nullptr && free_jp_bert_model_ != nullptr) {
      free_jp_bert_model_(jp_bert_handle_);
    }
    if (model_handle_ != nullptr && free_model_ != nullptr) {
      free_model_(model_handle_);
    }
  }

  bool HasSynthesisModel() const noexcept {
    return model_handle_ != nullptr;
  }

  bool HasJpBertModel() const noexcept {
    return jp_bert_handle_ != nullptr;
  }

  std::string ContractSummary() const {
    std::ostringstream out;
    out << "contract=" << kTtsCppRuntimeContract
        << ", runtime_abi=" << OptionalVersionString(runtime_abi_version_)
        << ", gguf_schema=" << OptionalVersionString(gguf_schema_version_);
    return out.str();
  }

  bool SynthesizeFrontWithStyleVec(
      const int32_t* phone_ids,
      const int32_t* tone_ids,
      const int32_t* language_ids,
      size_t tokens,
      const float* bert,
      size_t bert_length,
      const float* style_vec,
      size_t style_vec_length,
      int32_t speaker_id,
      float sdp_ratio,
      float length_scale,
      float noise_scale,
      float noise_w_scale,
      tts_style_bert_vits2_float_buffer& out_audio,
      std::string& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (synthesize_front_with_style_vec_ == nullptr || model_handle_ == nullptr) {
      error = "TTS.cpp Style-Bert-VITS2 direct style_vec synthesis API is not loaded.";
      return false;
    }
    const int ok = synthesize_front_with_style_vec_(
        model_handle_,
        phone_ids,
        tone_ids,
        language_ids,
        tokens,
        bert,
        bert_length,
        style_vec,
        style_vec_length,
        speaker_id,
        sdp_ratio,
        length_scale,
        noise_scale,
        noise_w_scale,
        &out_audio);
    if (ok == 0) {
      const char* detail = last_error_ != nullptr ? last_error_() : nullptr;
      error = detail != nullptr && detail[0] ? detail : "TTS.cpp Style-Bert-VITS2 synthesis failed.";
      return false;
    }
    return true;
  }

  bool EncodeJpBertFeatures(
      const int32_t* input_ids,
      size_t tokens,
      tts_style_bert_vits2_float_buffer& out_features,
      std::string& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (encode_jp_bert_features_ == nullptr || jp_bert_handle_ == nullptr) {
      error = "TTS.cpp Style-Bert-VITS2 JP-BERT API is not loaded.";
      return false;
    }
    const int ok = encode_jp_bert_features_(
        jp_bert_handle_,
        input_ids,
        tokens,
        &out_features);
    if (ok == 0) {
      const char* detail = last_error_ != nullptr ? last_error_() : nullptr;
      error = detail != nullptr && detail[0] ? detail : "TTS.cpp Style-Bert-VITS2 JP-BERT feature extraction failed.";
      return false;
    }
    return true;
  }

 private:
  explicit TtsCppRuntime(std::unique_ptr<DynamicLibrary> library)
      : library_(std::move(library)) {}

  static std::string OptionalVersionString(Uint32Fn version_fn) {
    if (version_fn == nullptr) {
      return "legacy";
    }
    return std::to_string(version_fn());
  }

  void ValidateOptionalVersionSymbols() const {
    if (runtime_abi_version_ != nullptr) {
      const uint32_t actual = runtime_abi_version_();
      if (actual != kExpectedTtsCppRuntimeAbiVersion) {
        std::ostringstream message;
        message << "TTS.cpp Style-Bert-VITS2 runtime ABI version " << actual
                << " != " << kExpectedTtsCppRuntimeAbiVersion << ".";
        throw std::runtime_error(message.str());
      }
    }
    if (gguf_schema_version_ != nullptr) {
      const uint32_t actual = gguf_schema_version_();
      if (actual != kExpectedTtsCppGgufSchemaVersion) {
        std::ostringstream message;
        message << "TTS.cpp Style-Bert-VITS2 GGUF schema version " << actual
                << " != " << kExpectedTtsCppGgufSchemaVersion << ".";
        throw std::runtime_error(message.str());
      }
    }
  }

  std::unique_ptr<DynamicLibrary> library_;
  LastErrorFn last_error_ = nullptr;
  Uint32Fn runtime_abi_version_ = nullptr;
  Uint32Fn gguf_schema_version_ = nullptr;
  LoadModelFn load_model_ = nullptr;
  FreeModelFn free_model_ = nullptr;
  LoadJpBertModelFn load_jp_bert_model_ = nullptr;
  FreeJpBertModelFn free_jp_bert_model_ = nullptr;
  SynthesizeFrontFn synthesize_front_ = nullptr;
  SynthesizeFrontWithStyleVecFn synthesize_front_with_style_vec_ = nullptr;
  EncodeJpBertFeaturesFn encode_jp_bert_features_ = nullptr;
  tts_style_bert_vits2_handle* model_handle_ = nullptr;
  tts_style_bert_vits2_jp_bert_handle* jp_bert_handle_ = nullptr;
  std::mutex mutex_;
};

std::string NormalizeRuntimeRegistryPath(const std::string& raw_path) {
  if (raw_path.empty()) {
    return "";
  }
  std::error_code error;
  std::filesystem::path normalized = std::filesystem::weakly_canonical(raw_path, error);
  if (error) {
    error.clear();
    normalized = std::filesystem::absolute(raw_path, error);
  }
  if (error) {
    normalized = std::filesystem::path(raw_path);
  }
  return normalized.lexically_normal().string();
}

std::string BuildRuntimeRegistryKey(const AivisGgmlEpConfig& config) {
  std::ostringstream out;
  out << kRuntimeRegistryContract
      << "\nbackend=" << config.backend
      << "\ndevice=" << config.device
      << "\nprecision=" << config.precision
      << "\nn_threads=" << config.n_threads
      << "\ntts_cpp_library_path=" << NormalizeRuntimeRegistryPath(config.tts_cpp_library_path)
      << "\ngguf_path=" << NormalizeRuntimeRegistryPath(config.gguf_path)
      << "\njp_bert_gguf_path=" << NormalizeRuntimeRegistryPath(config.jp_bert_gguf_path);
  return out.str();
}

class TtsCppRuntimeRegistry final {
 public:
  static std::shared_ptr<TtsCppRuntime> Acquire(
      const AivisGgmlEpConfig& config,
      bool& reused) {
    reused = false;
    const std::string key = BuildRuntimeRegistryKey(config);
    std::lock_guard<std::mutex> lock(Mutex());
    auto& registry = Registry();
    const auto existing = registry.find(key);
    if (existing != registry.end()) {
      if (std::shared_ptr<TtsCppRuntime> runtime = existing->second.lock()) {
        reused = true;
        return runtime;
      }
      registry.erase(existing);
    }

    std::shared_ptr<TtsCppRuntime> runtime = TtsCppRuntime::LoadAndMaybeOpenModel(config);
    registry.emplace(key, runtime);
    return runtime;
  }

 private:
  static std::mutex& Mutex() {
    static std::mutex mutex;
    return mutex;
  }

  static std::unordered_map<std::string, std::weak_ptr<TtsCppRuntime>>& Registry() {
    static std::unordered_map<std::string, std::weak_ptr<TtsCppRuntime>> registry;
    return registry;
  }
};

enum class AivisGgmlGraphKind {
  Synthesis,
  JpBert,
};

const char* GraphKindName(AivisGgmlGraphKind graph_kind) noexcept {
  switch (graph_kind) {
    case AivisGgmlGraphKind::Synthesis:
      return "synthesis";
    case AivisGgmlGraphKind::JpBert:
      return "jp-bert";
  }
  return "unknown";
}

std::string JsonEscape(const std::string& value) {
  std::ostringstream out;
  for (char raw : value) {
    const unsigned char c = static_cast<unsigned char>(raw);
    switch (c) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\b':
        out << "\\b";
        break;
      case '\f':
        out << "\\f";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        if (c < 0x20) {
          out << "\\u00";
          const char* hex = "0123456789abcdef";
          out << hex[(c >> 4) & 0x0F] << hex[c & 0x0F];
        } else {
          out << raw;
        }
        break;
    }
  }
  return out.str();
}

bool HasParentDirectoryTraversal(const std::filesystem::path& path) {
  return std::any_of(
      path.begin(),
      path.end(),
      [](const std::filesystem::path& component) {
        return component == "..";
      });
}

std::filesystem::path EpContextModelDirectory(const AivisGgmlEpConfig& config) {
  if (config.ort_ep_context_file_path.empty()) {
    if (!config.ort_ep_context_embed_mode) {
      throw std::runtime_error(
          "AivisGgmlExecutionProvider ep.context_file_path is required when "
          "ep.context_enable=1 and ep.context_embed_mode=0.");
    }
    return {};
  }
  std::filesystem::path context_model_path(config.ort_ep_context_file_path);
  std::filesystem::path directory = context_model_path.parent_path();
  if (directory.empty()) {
    directory = ".";
  }
  return directory;
}

std::string PortablePathForEpContext(
    const std::string& raw_path,
    const std::filesystem::path& context_model_directory,
    const char* option_name) {
  if (raw_path.empty()) {
    return "";
  }
  if (context_model_directory.empty()) {
    return std::filesystem::path(raw_path).filename().string();
  }

  std::error_code error;
  const std::filesystem::path absolute_target =
      std::filesystem::weakly_canonical(raw_path, error);
  if (error) {
    throw std::runtime_error(std::string("could not canonicalize ") + option_name + ".");
  }

  error.clear();
  const std::filesystem::path absolute_base =
      std::filesystem::absolute(context_model_directory, error).lexically_normal();
  if (error) {
    throw std::runtime_error("could not resolve ep.context_file_path directory.");
  }

  error.clear();
  std::filesystem::path relative_path =
      std::filesystem::relative(absolute_target, absolute_base, error);
  if (error || relative_path.empty() || relative_path.is_absolute() ||
      HasParentDirectoryTraversal(relative_path)) {
    throw std::runtime_error(
        std::string("AivisGgmlExecutionProvider ") + option_name +
        " must be in the ep.context_file_path directory or one of its subdirectories "
        "when generating a portable EPContext model.");
  }
  return relative_path.generic_string();
}

std::string BuildEpContextPayload(
    const AivisGgmlEpConfig& config,
    AivisGgmlGraphKind graph_kind,
    const OrtGraph* ort_graph,
    size_t graph_index) {
  const std::filesystem::path context_model_directory = EpContextModelDirectory(config);
  Ort::ConstGraph graph{ort_graph};
  std::ostringstream out;
  out << "{";
  out << "\"version\":\"" << kOfficialEpContextVersion << "\"";
  out << ",\"provider_name\":\"" << kEpName << "\"";
  out << ",\"provider_version\":\"" << kVersion << "\"";
  out << ",\"runtime_registry_contract\":\"" << kRuntimeRegistryContract << "\"";
  out << ",\"tts_cpp_runtime_contract\":\"" << kTtsCppRuntimeContract << "\"";
  out << ",\"graph_kind\":\"" << GraphKindName(graph_kind) << "\"";
  out << ",\"graph_name\":\"" << JsonEscape(graph.GetName()) << "\"";
  out << ",\"graph_index\":" << graph_index;
  out << ",\"backend\":\"" << JsonEscape(config.backend) << "\"";
  out << ",\"device\":\"" << JsonEscape(config.device) << "\"";
  out << ",\"precision\":\"" << JsonEscape(config.precision) << "\"";
  out << ",\"n_threads\":" << config.n_threads;
  out << ",\"artifacts\":{";
  out << "\"cache_manifest_path\":\""
      << JsonEscape(PortablePathForEpContext(
             config.cache_manifest_path,
             context_model_directory,
             "cache_manifest_path"))
      << "\"";
  out << ",\"gguf_path\":\""
      << JsonEscape(PortablePathForEpContext(
             config.gguf_path,
             context_model_directory,
             "gguf_path"))
      << "\"";
  out << ",\"jp_bert_gguf_path\":\""
      << JsonEscape(PortablePathForEpContext(
             config.jp_bert_gguf_path,
             context_model_directory,
             "jp_bert_gguf_path"))
      << "\"";
  out << "}}";
  return out.str();
}

std::string EpContextExternalFilename(
    const AivisGgmlEpConfig& config,
    AivisGgmlGraphKind graph_kind,
    size_t graph_index) {
  std::filesystem::path context_model_path(config.ort_ep_context_file_path);
  std::string stem = context_model_path.stem().string();
  if (stem.empty()) {
    stem = "model_ctx";
  }
  std::ostringstream filename;
  filename << stem << "_aivis_ggml_" << GraphKindName(graph_kind)
           << "_" << graph_index << ".json";
  return filename.str();
}

std::string WriteEpContextPayloadFile(
    const AivisGgmlEpConfig& config,
    AivisGgmlGraphKind graph_kind,
    size_t graph_index,
    const std::string& payload) {
  const std::filesystem::path directory = EpContextModelDirectory(config);
  const std::string filename = EpContextExternalFilename(config, graph_kind, graph_index);
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  if (error) {
    throw std::runtime_error("could not create ep.context_file_path directory.");
  }

  const std::filesystem::path output_path = directory / filename;
  std::ofstream file(output_path, std::ios::binary | std::ios::trunc);
  if (!file) {
    throw std::runtime_error("could not create Aivis GGML EPContext payload file.");
  }
  file << payload;
  if (!file) {
    throw std::runtime_error("could not write Aivis GGML EPContext payload file.");
  }
  return filename;
}

OrtStatus* AddInt64OpAttr(
    const OrtApi& api,
    std::vector<OrtOpAttr*>& attributes,
    const char* name,
    int64_t value) {
  OrtOpAttr* attribute = nullptr;
  OrtStatus* status = api.CreateOpAttr(
      name,
      &value,
      1,
      ORT_OP_ATTR_INT,
      &attribute);
  if (status != nullptr) {
    return status;
  }
  attributes.push_back(attribute);
  return nullptr;
}

OrtStatus* AddStringOpAttr(
    const OrtApi& api,
    std::vector<OrtOpAttr*>& attributes,
    const char* name,
    const std::string& value) {
  OrtOpAttr* attribute = nullptr;
  OrtStatus* status = api.CreateOpAttr(
      name,
      value.data(),
      static_cast<int>(value.size()),
      ORT_OP_ATTR_STRING,
      &attribute);
  if (status != nullptr) {
    return status;
  }
  attributes.push_back(attribute);
  return nullptr;
}

void ReleaseOpAttrs(const OrtApi& api, std::vector<OrtOpAttr*>& attributes) noexcept {
  for (OrtOpAttr* attribute : attributes) {
    if (attribute != nullptr) {
      api.ReleaseOpAttr(attribute);
    }
  }
  attributes.clear();
}

OrtStatus* CreateEpContextNode(
    const OrtApi& api,
    const AivisGgmlEpConfig& config,
    const OrtGraph* ort_graph,
    const OrtNode* fused_node,
    AivisGgmlGraphKind graph_kind,
    size_t graph_index,
    OrtNode** ep_context_node) noexcept {
  if (ep_context_node == nullptr) {
    return CreateStatus(api, ORT_INVALID_ARGUMENT, "EPContext node output is null.");
  }
  *ep_context_node = nullptr;

  try {
    const OrtModelEditorApi* model_editor_api = api.GetModelEditorApi();
    if (model_editor_api == nullptr) {
      return CreateStatus(
          api,
          ORT_NOT_IMPLEMENTED,
          "ONNX Runtime Model Editor API is unavailable; cannot create EPContext nodes.");
    }

    Ort::ConstGraph graph{ort_graph};
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::string ep_context_node_name;
    if (fused_node != nullptr) {
      Ort::ConstNode fused{fused_node};
      ep_context_node_name = fused.GetName();
      for (const Ort::ConstValueInfo& input : fused.GetInputs()) {
        input_names.push_back(input.GetName());
      }
      for (const Ort::ConstValueInfo& output : fused.GetOutputs()) {
        output_names.push_back(output.GetName());
      }
    } else {
      for (const Ort::ConstValueInfo& input : graph.GetInputs()) {
        input_names.push_back(input.GetName());
      }
      for (const Ort::ConstValueInfo& output : graph.GetOutputs()) {
        output_names.push_back(output.GetName());
      }
    }

    std::vector<const char*> input_name_ptrs;
    std::vector<const char*> output_name_ptrs;
    input_name_ptrs.reserve(input_names.size());
    output_name_ptrs.reserve(output_names.size());
    for (const std::string& name : input_names) {
      input_name_ptrs.push_back(name.c_str());
    }
    for (const std::string& name : output_names) {
      output_name_ptrs.push_back(name.c_str());
    }

    const std::string payload =
        BuildEpContextPayload(config, graph_kind, ort_graph, graph_index);
    const int64_t embed_mode = config.ort_ep_context_embed_mode ? 1 : 0;
    const std::string ep_cache_context =
        config.ort_ep_context_embed_mode
            ? payload
            : WriteEpContextPayloadFile(config, graph_kind, graph_index, payload);
    if (ep_context_node_name.empty()) {
      const std::string prefix =
          config.ort_ep_context_node_name_prefix.empty()
              ? std::string(kEpName)
              : config.ort_ep_context_node_name_prefix;
      std::ostringstream fallback_name;
      fallback_name << prefix << "_" << GraphKindName(graph_kind)
                    << "_ep_context_" << graph_index;
      ep_context_node_name = fallback_name.str();
    }
    const std::string partition_name =
        std::string(GraphKindName(graph_kind)) + "_" + std::to_string(graph_index);
    const std::string hardware_architecture =
        config.device.empty() ? config.backend : config.backend + ":" + config.device;
    const std::string ep_sdk_version =
        std::string("onnxruntime-ep-aivis-ggml/") + kVersion;

    std::vector<OrtOpAttr*> attributes;
    attributes.reserve(8);
    OrtStatus* status = AddInt64OpAttr(api, attributes, "main_context", 1);
    if (status == nullptr) {
      status = AddStringOpAttr(api, attributes, "ep_cache_context", ep_cache_context);
    }
    if (status == nullptr) {
      status = AddInt64OpAttr(api, attributes, "embed_mode", embed_mode);
    }
    if (status == nullptr) {
      status = AddStringOpAttr(api, attributes, "source", kEpName);
    }
    if (status == nullptr) {
      status = AddStringOpAttr(api, attributes, "ep_sdk_version", ep_sdk_version);
    }
    if (status == nullptr) {
      status = AddStringOpAttr(api, attributes, "partition_name", partition_name);
    }
    if (status == nullptr) {
      status = AddStringOpAttr(api, attributes, "hardware_architecture", hardware_architecture);
    }
    if (status == nullptr) {
      status = AddStringOpAttr(api, attributes, "notes", kOfficialEpContextVersion);
    }
    if (status != nullptr) {
      ReleaseOpAttrs(api, attributes);
      return status;
    }

    status = model_editor_api->CreateNode(
        "EPContext",
        "com.microsoft",
        ep_context_node_name.c_str(),
        input_name_ptrs.data(),
        input_name_ptrs.size(),
        output_name_ptrs.data(),
        output_name_ptrs.size(),
        attributes.data(),
        attributes.size(),
        ep_context_node);
    ReleaseOpAttrs(api, attributes);
    return status;
  } catch (const std::bad_alloc&) {
    return CreateStatus(api, ORT_FAIL, "Out of memory creating Aivis GGML EPContext node.");
  } catch (const std::exception& ex) {
    return CreateStatus(api, ORT_FAIL, ex.what());
  } catch (...) {
    return CreateStatus(api, ORT_FAIL, "Unknown error creating Aivis GGML EPContext node.");
  }
}

struct AivisGgmlNodeComputeInfo final : OrtNodeComputeInfo {
  AivisGgmlNodeComputeInfo(
      const OrtApi& ort_api,
      std::shared_ptr<TtsCppRuntime> runtime,
      AivisGgmlGraphKind graph_kind,
      std::vector<size_t> input_indices,
      size_t primary_output_index)
      : OrtNodeComputeInfo{},
        ort_api_(ort_api),
        runtime_(std::move(runtime)),
        graph_kind_(graph_kind),
        input_indices_(std::move(input_indices)),
        primary_output_index_(primary_output_index) {
    ort_version_supported = ORT_API_VERSION;
    CreateState = CreateStateImpl;
    Compute = ComputeImpl;
    ReleaseState = ReleaseStateImpl;
  }

  static OrtStatus* ORT_API_CALL CreateStateImpl(
      OrtNodeComputeInfo* this_ptr,
      OrtNodeComputeContext* compute_context,
      void** compute_state) noexcept {
    auto* info = static_cast<AivisGgmlNodeComputeInfo*>(this_ptr);
    if (compute_state == nullptr) {
      return CreateStatus(info->ort_api_, ORT_INVALID_ARGUMENT, "Aivis GGML compute state output is null.");
    }

    *compute_state = nullptr;
    try {
      const OrtEpApi* ep_api = info->ort_api_.GetEpApi();
      const char* node_name =
          ep_api != nullptr && compute_context != nullptr
              ? ep_api->NodeComputeContext_NodeName(compute_context)
              : nullptr;
      auto state = std::make_unique<std::string>(node_name != nullptr ? node_name : "");
      *compute_state = state.release();
      return nullptr;
    } catch (const std::bad_alloc&) {
      return CreateStatus(info->ort_api_, ORT_FAIL, "Out of memory creating Aivis GGML compute state.");
    } catch (const std::exception& ex) {
      return CreateStatus(info->ort_api_, ORT_FAIL, ex.what());
    } catch (...) {
      return CreateStatus(info->ort_api_, ORT_FAIL, "Unknown error creating Aivis GGML compute state.");
    }
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(
      OrtNodeComputeInfo* this_ptr,
      void* compute_state,
      OrtKernelContext* kernel_context) noexcept {
    auto* info = static_cast<AivisGgmlNodeComputeInfo*>(this_ptr);
    if (kernel_context == nullptr) {
      return CreateStatus(info->ort_api_, ORT_INVALID_ARGUMENT, "Aivis GGML Compute received a null kernel context.");
    }
    if (info->runtime_ == nullptr) {
      return CreateStatus(
          info->ort_api_,
          ORT_NOT_IMPLEMENTED,
          "Aivis GGML Compute requires an eager-loaded TTS.cpp runtime.");
    }
    if (info->graph_kind_ == AivisGgmlGraphKind::JpBert) {
      return ComputeJpBertImpl(info, compute_state, kernel_context);
    }

    try {
      Ort::KernelContext context(kernel_context);
      const size_t input_count = context.GetInputCount();
      const size_t output_count = context.GetOutputCount();
      if (input_count != kExpectedInputNames.size()) {
        std::ostringstream message;
        message << "Aivis GGML Compute expected " << kExpectedInputNames.size()
                << " inputs, got " << input_count << ".";
        const std::string error_message = message.str();
        return CreateStatus(info->ort_api_, ORT_INVALID_ARGUMENT, error_message.c_str());
      }
      if (output_count == 0) {
        return CreateStatus(info->ort_api_, ORT_INVALID_ARGUMENT, "Aivis GGML Compute expected at least one output.");
      }

      if (info->input_indices_.size() != kExpectedInputNames.size()) {
        return CreateStatus(info->ort_api_, ORT_FAIL, "Aivis GGML synthesis input index map is invalid.");
      }
      Ort::ConstValue x_tst = context.GetInput(info->input_indices_[0]);
      const size_t tokens = ExpectTokenMatrix(x_tst, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, "x_tst");
      std::vector<int32_t> phone_ids = CopyInt64TokenInput(x_tst, tokens, "x_tst");
      const int32_t x_tst_length = ReadInt64ScalarVectorInput(context.GetInput(info->input_indices_[1]), "x_tst_lengths");
      if (x_tst_length < 0 || static_cast<size_t>(x_tst_length) != tokens) {
        throw std::runtime_error("x_tst_lengths does not match x_tst token count.");
      }
      const int32_t speaker_id = ReadInt64ScalarVectorInput(context.GetInput(info->input_indices_[2]), "sid");
      std::vector<int32_t> tone_ids = CopyInt64TokenInput(context.GetInput(info->input_indices_[3]), tokens, "tones");
      std::vector<int32_t> language_ids = CopyInt64TokenInput(context.GetInput(info->input_indices_[4]), tokens, "language");
      const float* bert = ExpectBertInput(context.GetInput(info->input_indices_[5]), tokens);
      const float* style_vec = ExpectStyleVecInput(context.GetInput(info->input_indices_[6]));
      const float length_scale = ReadFloatScalarInput(context.GetInput(info->input_indices_[7]), "length_scale");
      const float sdp_ratio = ReadFloatScalarInput(context.GetInput(info->input_indices_[8]), "sdp_ratio");
      const float noise_scale = ReadFloatScalarInput(context.GetInput(info->input_indices_[9]), "noise_scale");
      const float noise_scale_w = ReadFloatScalarInput(context.GetInput(info->input_indices_[10]), "noise_scale_w");
      if (tokens > std::numeric_limits<size_t>::max() / 1024) {
        throw std::runtime_error("bert element count overflows size_t.");
      }

      tts_style_bert_vits2_float_buffer audio{};
      std::string synthesis_error;
      if (!info->runtime_->SynthesizeFrontWithStyleVec(
              phone_ids.data(),
              tone_ids.data(),
              language_ids.data(),
              tokens,
              bert,
              tokens * 1024,
              style_vec,
              256,
              speaker_id,
              sdp_ratio,
              length_scale,
              noise_scale,
              noise_scale_w,
              audio,
              synthesis_error)) {
        return CreateStatus(info->ort_api_, ORT_FAIL, synthesis_error.c_str());
      }
      if (audio.hidden_size != 1) {
        std::ostringstream message;
        message << "TTS.cpp returned hidden_size=" << audio.hidden_size << ", expected 1.";
        const std::string error_message = message.str();
        return CreateStatus(info->ort_api_, ORT_FAIL, error_message.c_str());
      }
      if (audio.length > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        return CreateStatus(info->ort_api_, ORT_FAIL, "TTS.cpp returned an audio tensor too large for ONNX shape.");
      }

      const std::vector<int64_t> output_shape = {1, 1, static_cast<int64_t>(audio.length)};
      Ort::UnownedValue output = context.GetOutput(info->primary_output_index_, output_shape);
      if (output == nullptr) {
        return CreateStatus(info->ort_api_, ORT_FAIL, "Aivis GGML Compute could not allocate output tensor.");
      }
      float* output_data = output.GetTensorMutableData<float>();
      if (audio.length > 0) {
        if (audio.data == nullptr) {
          return CreateStatus(info->ort_api_, ORT_FAIL, "TTS.cpp returned null audio data.");
        }
        std::copy(audio.data, audio.data + audio.length, output_data);
      }

      const std::vector<int64_t> placeholder_shape = {1};
      for (size_t output_index = 0; output_index < output_count; ++output_index) {
        if (output_index == info->primary_output_index_) {
          continue;
        }
        Ort::UnownedValue placeholder = context.GetOutput(output_index, placeholder_shape);
        if (placeholder != nullptr) {
          placeholder.GetTensorMutableData<float>()[0] = 0.0f;
        }
      }

      const auto* node_name = static_cast<const std::string*>(compute_state);
      std::ostringstream trace;
      trace << "synthesized samples=" << audio.length << " tokens=" << tokens;
      if (node_name != nullptr && !node_name->empty()) {
        trace << " node='" << *node_name << "'";
      }
      TraceMessage(trace.str());
      return nullptr;
    } catch (const Ort::Exception& ex) {
      return CreateStatus(info->ort_api_, ORT_FAIL, ex.what());
    } catch (const std::bad_alloc&) {
      return CreateStatus(info->ort_api_, ORT_FAIL, "Out of memory during Aivis GGML Compute.");
    } catch (const std::exception& ex) {
      return CreateStatus(info->ort_api_, ORT_FAIL, ex.what());
    } catch (...) {
      return CreateStatus(info->ort_api_, ORT_FAIL, "Unknown error during Aivis GGML Compute.");
    }
  }

  static OrtStatus* ComputeJpBertImpl(
      AivisGgmlNodeComputeInfo* info,
      void* compute_state,
      OrtKernelContext* kernel_context) noexcept {
    if (!info->runtime_->HasJpBertModel()) {
      return CreateStatus(
          info->ort_api_,
          ORT_NOT_IMPLEMENTED,
          "Aivis GGML JP-BERT Compute requires an eager-loaded TTS.cpp JP-BERT runtime.");
    }

    try {
      Ort::KernelContext context(kernel_context);
      const size_t input_count = context.GetInputCount();
      const size_t output_count = context.GetOutputCount();
      if (input_count != kExpectedJpBertInputCount) {
        std::ostringstream message;
        message << "Aivis GGML JP-BERT Compute expected " << kExpectedJpBertInputCount
                << " inputs, got " << input_count << ".";
        const std::string error_message = message.str();
        return CreateStatus(info->ort_api_, ORT_INVALID_ARGUMENT, error_message.c_str());
      }
      if (output_count != kExpectedJpBertOutputCount) {
        std::ostringstream message;
        message << "Aivis GGML JP-BERT Compute expected " << kExpectedJpBertOutputCount
                << " output, got " << output_count << ".";
        const std::string error_message = message.str();
        return CreateStatus(info->ort_api_, ORT_INVALID_ARGUMENT, error_message.c_str());
      }

      if (info->input_indices_.size() != kExpectedJpBertInputNames.size()) {
        return CreateStatus(info->ort_api_, ORT_FAIL, "Aivis GGML JP-BERT input index map is invalid.");
      }
      Ort::ConstValue input_ids_value = context.GetInput(info->input_indices_[0]);
      const size_t tokens = ExpectTokenMatrix(input_ids_value, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, "input_ids");
      std::vector<int32_t> input_ids = CopyInt64TokenInput(input_ids_value, tokens, "input_ids");
      std::vector<int32_t> attention_mask = CopyInt64TokenInput(context.GetInput(info->input_indices_[1]), tokens, "attention_mask");
      for (int32_t value : attention_mask) {
        if (value != 0 && value != 1) {
          throw std::runtime_error("attention_mask must contain only 0 or 1 values.");
        }
      }

      tts_style_bert_vits2_float_buffer features{};
      std::string error;
      if (!info->runtime_->EncodeJpBertFeatures(
              input_ids.data(),
              tokens,
              features,
              error)) {
        return CreateStatus(info->ort_api_, ORT_FAIL, error.c_str());
      }
      if (features.hidden_size != 1024) {
        std::ostringstream message;
        message << "TTS.cpp JP-BERT returned hidden_size=" << features.hidden_size << ", expected 1024.";
        const std::string error_message = message.str();
        return CreateStatus(info->ort_api_, ORT_FAIL, error_message.c_str());
      }
      const size_t expected_values = tokens * static_cast<size_t>(features.hidden_size);
      if (features.length != expected_values) {
        return CreateStatus(info->ort_api_, ORT_FAIL, "TTS.cpp JP-BERT feature length does not match metadata.");
      }
      if (tokens > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        return CreateStatus(info->ort_api_, ORT_FAIL, "TTS.cpp JP-BERT returned too many tokens for ONNX shape.");
      }

      const std::vector<int64_t> output_shape = {
          static_cast<int64_t>(tokens),
          static_cast<int64_t>(features.hidden_size),
      };
      Ort::UnownedValue output = context.GetOutput(info->primary_output_index_, output_shape);
      if (output == nullptr) {
        return CreateStatus(info->ort_api_, ORT_FAIL, "Aivis GGML JP-BERT Compute could not allocate output tensor.");
      }
      float* output_data = output.GetTensorMutableData<float>();
      if (features.data == nullptr) {
        return CreateStatus(info->ort_api_, ORT_FAIL, "TTS.cpp JP-BERT returned null feature data.");
      }
      std::copy(features.data, features.data + features.length, output_data);

      const auto* node_name = static_cast<const std::string*>(compute_state);
      std::ostringstream trace;
      trace << "encoded jp-bert tokens=" << tokens;
      if (node_name != nullptr && !node_name->empty()) {
        trace << " node='" << *node_name << "'";
      }
      TraceMessage(trace.str());
      return nullptr;
    } catch (const Ort::Exception& ex) {
      return CreateStatus(info->ort_api_, ORT_FAIL, ex.what());
    } catch (const std::bad_alloc&) {
      return CreateStatus(info->ort_api_, ORT_FAIL, "Out of memory during Aivis GGML JP-BERT Compute.");
    } catch (const std::exception& ex) {
      return CreateStatus(info->ort_api_, ORT_FAIL, ex.what());
    } catch (...) {
      return CreateStatus(info->ort_api_, ORT_FAIL, "Unknown error during Aivis GGML JP-BERT Compute.");
    }
  }

  static void ORT_API_CALL ReleaseStateImpl(
      OrtNodeComputeInfo* /*this_ptr*/,
      void* compute_state) noexcept {
    delete static_cast<std::string*>(compute_state);
  }

  const OrtApi& ort_api_;
  std::shared_ptr<TtsCppRuntime> runtime_;
  AivisGgmlGraphKind graph_kind_;
  std::vector<size_t> input_indices_;
  size_t primary_output_index_;
};

struct GraphSignatureGateResult {
  bool supported = true;
  std::vector<std::string> reasons;

  void Reject(std::string reason) {
    supported = false;
    reasons.push_back(std::move(reason));
  }
};

struct EpContextGateResult {
  bool supported = false;
  AivisGgmlGraphKind graph_kind = AivisGgmlGraphKind::Synthesis;
  std::vector<std::string> reasons;

  void Reject(std::string reason) {
    supported = false;
    reasons.push_back(std::move(reason));
  }

  void Accept(AivisGgmlGraphKind accepted_graph_kind) {
    supported = true;
    graph_kind = accepted_graph_kind;
    reasons.clear();
  }
};

GraphSignatureGateResult MatchStyleBertVits2SynthesisGraph(const OrtGraph* ort_graph) {
  GraphSignatureGateResult result;
  Ort::ConstGraph graph{ort_graph};

  const std::string graph_name = graph.GetName();
  if (!IsAcceptedGraphName(graph_name)) {
    std::ostringstream reason;
    reason << "graph name '" << graph_name << "' is not accepted";
    result.Reject(reason.str());
  }

  const int64_t ir_version = graph.GetOnnxIRVersion();
  if (ir_version != kExpectedIrVersion) {
    std::ostringstream reason;
    reason << "IR version " << ir_version << " != " << kExpectedIrVersion;
    result.Reject(reason.str());
  }

  const std::vector<Ort::OperatorSet> opsets = graph.GetOperatorSets();
  const bool has_default_opset = std::any_of(
      opsets.begin(),
      opsets.end(),
      [](const Ort::OperatorSet& opset) {
        return opset.domain == "" && opset.version == kExpectedOpsetVersion;
      });
  if (!has_default_opset) {
    std::ostringstream reason;
    reason << "default-domain opset " << kExpectedOpsetVersion << " is missing";
    result.Reject(reason.str());
  }

  const std::vector<Ort::ConstValueInfo> inputs = graph.GetInputs();
  if (inputs.size() != kExpectedInputNames.size()) {
    std::ostringstream reason;
    reason << "input count " << inputs.size() << " != " << kExpectedInputNames.size();
    result.Reject(reason.str());
  }

  for (size_t i = 0; i < kExpectedInputNames.size(); ++i) {
    const auto input = std::find_if(
        inputs.begin(),
        inputs.end(),
        [i](const Ort::ConstValueInfo& value_info) {
          return value_info.GetName() == kExpectedInputNames[i];
        });
    if (input == inputs.end()) {
      std::ostringstream reason;
      reason << "input '" << kExpectedInputNames[i] << "' is missing";
      result.Reject(reason.str());
      continue;
    }

    const ONNXTensorElementDataType elem_type = TensorElementType(*input);
    if (elem_type != kExpectedInputTypes[i]) {
      std::ostringstream reason;
      reason << "input '" << kExpectedInputNames[i] << "' type " << TensorElementTypeName(elem_type)
             << " != " << TensorElementTypeName(kExpectedInputTypes[i]);
      result.Reject(reason.str());
    }
  }

  const std::vector<Ort::ConstValueInfo> outputs = graph.GetOutputs();
  if (outputs.size() != kExpectedOutputCount) {
    std::ostringstream reason;
    reason << "output count " << outputs.size() << " != " << kExpectedOutputCount;
    result.Reject(reason.str());
  }
  const auto audio_output = std::find_if(
      outputs.begin(),
      outputs.end(),
      [](const Ort::ConstValueInfo& value_info) {
        return value_info.GetName() == kExpectedFirstOutputName;
      });
  if (audio_output == outputs.end()) {
    std::ostringstream reason;
    reason << "output '" << kExpectedFirstOutputName << "' is missing";
    result.Reject(reason.str());
  } else {
    const ONNXTensorElementDataType elem_type = TensorElementType(*audio_output);
    if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      std::ostringstream reason;
      reason << "output '" << kExpectedFirstOutputName << "' type "
             << TensorElementTypeName(elem_type) << " != FLOAT";
      result.Reject(reason.str());
    }
  }

  const std::vector<Ort::ConstValueInfo> initializers = graph.GetInitializers();
  if (initializers.size() != kExpectedInitializerCount &&
      initializers.size() != kExpectedOptimizedInitializerCount) {
    std::ostringstream reason;
    reason << "initializer count " << initializers.size() << " is not one of "
           << kExpectedInitializerCount << " or " << kExpectedOptimizedInitializerCount;
    result.Reject(reason.str());
  }

  std::unordered_set<std::string> initializer_names;
  initializer_names.reserve(initializers.size());
  for (const Ort::ConstValueInfo& initializer : initializers) {
    initializer_names.insert(initializer.GetName());
  }
  for (const char* expected_name : kRequiredInitializerNames) {
    if (initializer_names.find(expected_name) == initializer_names.end()) {
      std::ostringstream reason;
      reason << "required initializer '" << expected_name << "' is missing";
      result.Reject(reason.str());
    }
  }

  const std::vector<Ort::ConstNode> nodes = graph.GetNodes();
  if (nodes.size() != kExpectedNodeCount &&
      nodes.size() != kExpectedOptimizedNodeCount) {
    std::ostringstream reason;
    reason << "node count " << nodes.size() << " is not one of "
           << kExpectedNodeCount << " or " << kExpectedOptimizedNodeCount;
    result.Reject(reason.str());
  }

  std::unordered_set<std::string> op_types;
  op_types.reserve(nodes.size());
  for (const Ort::ConstNode& node : nodes) {
    op_types.insert(node.GetOperatorType());
  }
  for (const char* required_op_type : kRequiredOpTypes) {
    if (op_types.find(required_op_type) == op_types.end()) {
      std::ostringstream reason;
      reason << "required op type '" << required_op_type << "' is missing";
      result.Reject(reason.str());
    }
  }
  if (!nodes.empty() && nodes.back().GetOperatorType() != "Tanh") {
    std::ostringstream reason;
    reason << "last node op '" << nodes.back().GetOperatorType() << "' != 'Tanh'";
    result.Reject(reason.str());
  }

  return result;
}

GraphSignatureGateResult MatchStyleBertVits2JpBertGraph(const OrtGraph* ort_graph) {
  GraphSignatureGateResult result;
  Ort::ConstGraph graph{ort_graph};

  const std::string graph_name = graph.GetName();
  if (!IsAcceptedGraphName(graph_name)) {
    std::ostringstream reason;
    reason << "graph name '" << graph_name << "' is not accepted";
    result.Reject(reason.str());
  }

  const int64_t ir_version = graph.GetOnnxIRVersion();
  if (ir_version != kExpectedIrVersion) {
    std::ostringstream reason;
    reason << "IR version " << ir_version << " != " << kExpectedIrVersion;
    result.Reject(reason.str());
  }

  const std::vector<Ort::OperatorSet> opsets = graph.GetOperatorSets();
  const bool has_default_opset = std::any_of(
      opsets.begin(),
      opsets.end(),
      [](const Ort::OperatorSet& opset) {
        return opset.domain == "" && opset.version == kExpectedJpBertOpsetVersion;
      });
  if (!has_default_opset) {
    std::ostringstream reason;
    reason << "default-domain opset " << kExpectedJpBertOpsetVersion << " is missing";
    result.Reject(reason.str());
  }

  const std::vector<Ort::ConstValueInfo> inputs = graph.GetInputs();
  if (inputs.size() != kExpectedJpBertInputCount) {
    std::ostringstream reason;
    reason << "input count " << inputs.size() << " != " << kExpectedJpBertInputCount;
    result.Reject(reason.str());
  }

  for (size_t i = 0; i < kExpectedJpBertInputNames.size(); ++i) {
    const auto input = std::find_if(
        inputs.begin(),
        inputs.end(),
        [i](const Ort::ConstValueInfo& value_info) {
          return value_info.GetName() == kExpectedJpBertInputNames[i];
        });
    if (input == inputs.end()) {
      std::ostringstream reason;
      reason << "input '" << kExpectedJpBertInputNames[i] << "' is missing";
      result.Reject(reason.str());
      continue;
    }

    const ONNXTensorElementDataType elem_type = TensorElementType(*input);
    if (elem_type != kExpectedJpBertInputTypes[i]) {
      std::ostringstream reason;
      reason << "input '" << kExpectedJpBertInputNames[i] << "' type " << TensorElementTypeName(elem_type)
             << " != " << TensorElementTypeName(kExpectedJpBertInputTypes[i]);
      result.Reject(reason.str());
    }
  }

  const std::vector<Ort::ConstValueInfo> outputs = graph.GetOutputs();
  if (outputs.size() != kExpectedJpBertOutputCount) {
    std::ostringstream reason;
    reason << "output count " << outputs.size() << " != " << kExpectedJpBertOutputCount;
    result.Reject(reason.str());
  } else {
    const std::string output_name = outputs[0].GetName();
    if (output_name != kExpectedJpBertOutputName) {
      std::ostringstream reason;
      reason << "output[0] name '" << output_name << "' != '" << kExpectedJpBertOutputName << "'";
      result.Reject(reason.str());
    }
    const ONNXTensorElementDataType elem_type = TensorElementType(outputs[0]);
    if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      std::ostringstream reason;
      reason << "output[0] type " << TensorElementTypeName(elem_type) << " != FLOAT";
      result.Reject(reason.str());
    }
  }

  const std::vector<Ort::ConstValueInfo> initializers = graph.GetInitializers();
  if (std::find(
          kAcceptedJpBertInitializerCounts.begin(),
          kAcceptedJpBertInitializerCounts.end(),
          initializers.size()) == kAcceptedJpBertInitializerCounts.end()) {
    std::ostringstream reason;
    reason << "initializer count " << initializers.size() << " is not accepted";
    result.Reject(reason.str());
  }

  const std::vector<Ort::ConstNode> nodes = graph.GetNodes();
  if (std::find(
          kAcceptedJpBertNodeCounts.begin(),
          kAcceptedJpBertNodeCounts.end(),
          nodes.size()) == kAcceptedJpBertNodeCounts.end()) {
    std::ostringstream reason;
    reason << "node count " << nodes.size() << " is not accepted";
    result.Reject(reason.str());
  }

  std::unordered_set<std::string> op_types;
  op_types.reserve(nodes.size());
  for (const Ort::ConstNode& node : nodes) {
    op_types.insert(node.GetOperatorType());
  }
  for (const char* required_op_type : kRequiredJpBertOpTypes) {
    if (op_types.find(required_op_type) == op_types.end()) {
      std::ostringstream reason;
      reason << "required JP-BERT op type '" << required_op_type << "' is missing";
      result.Reject(reason.str());
    }
  }
  if (!nodes.empty() && nodes.back().GetOperatorType() != "Cast") {
    std::ostringstream reason;
    reason << "last node op '" << nodes.back().GetOperatorType() << "' != 'Cast'";
    result.Reject(reason.str());
  }

  return result;
}

bool ReadStringNodeAttribute(
    const Ort::ConstNode& node,
    const std::string& name,
    std::string& value) {
  Ort::ConstOpAttr attribute{nullptr};
  Ort::Status status = node.GetAttributeByName(name, attribute);
  if (!status.IsOK() || attribute == nullptr) {
    return false;
  }
  status = attribute.GetValue<std::string>(value);
  return status.IsOK();
}

bool ReadInt64NodeAttribute(
    const Ort::ConstNode& node,
    const std::string& name,
    int64_t& value) {
  Ort::ConstOpAttr attribute{nullptr};
  Ort::Status status = node.GetAttributeByName(name, attribute);
  if (!status.IsOK() || attribute == nullptr) {
    return false;
  }
  status = attribute.GetValue<int64_t>(value);
  return status.IsOK();
}

std::string JsonUnescape(const std::string& value) {
  std::string output;
  output.reserve(value.size());
  for (size_t i = 0; i < value.size(); ++i) {
    if (value[i] != '\\' || i + 1 >= value.size()) {
      output.push_back(value[i]);
      continue;
    }
    const char escaped = value[++i];
    switch (escaped) {
      case '\\':
        output.push_back('\\');
        break;
      case '"':
        output.push_back('"');
        break;
      case 'b':
        output.push_back('\b');
        break;
      case 'f':
        output.push_back('\f');
        break;
      case 'n':
        output.push_back('\n');
        break;
      case 'r':
        output.push_back('\r');
        break;
      case 't':
        output.push_back('\t');
        break;
      default:
        throw std::runtime_error("unsupported JSON escape sequence in Aivis GGML EPContext payload.");
    }
  }
  return output;
}

bool ExtractJsonStringField(
    const std::string& payload,
    const std::string& key,
    std::string& value) {
  const std::string marker = "\"" + key + "\":\"";
  const size_t value_start = payload.find(marker);
  if (value_start == std::string::npos) {
    return false;
  }
  size_t index = value_start + marker.size();
  std::string raw_value;
  bool escaped = false;
  for (; index < payload.size(); ++index) {
    const char c = payload[index];
    if (escaped) {
      raw_value.push_back('\\');
      raw_value.push_back(c);
      escaped = false;
      continue;
    }
    if (c == '\\') {
      escaped = true;
      continue;
    }
    if (c == '"') {
      value = JsonUnescape(raw_value);
      return true;
    }
    raw_value.push_back(c);
  }
  throw std::runtime_error("unterminated JSON string in Aivis GGML EPContext payload.");
}

bool ExtractJsonIntField(
    const std::string& payload,
    const std::string& key,
    int& value) {
  const std::string marker = "\"" + key + "\":";
  const size_t value_start = payload.find(marker);
  if (value_start == std::string::npos) {
    return false;
  }
  size_t index = value_start + marker.size();
  while (index < payload.size() && std::isspace(static_cast<unsigned char>(payload[index]))) {
    ++index;
  }
  size_t value_end = index;
  while (value_end < payload.size() &&
         (std::isdigit(static_cast<unsigned char>(payload[value_end])) ||
          payload[value_end] == '-')) {
    ++value_end;
  }
  if (value_end == index) {
    return false;
  }
  value = ParseNonNegativeIntOption(payload.substr(index, value_end - index), 0, key.c_str());
  return true;
}

std::filesystem::path EpContextInferenceBaseDirectory(
    const AivisGgmlEpConfig& config,
    const OrtGraph* ort_graph) {
  if (!config.ort_ep_context_file_path.empty()) {
    return EpContextModelDirectory(config);
  }

  Ort::ConstGraph graph{ort_graph};
  const std::filesystem::path model_path = graph.GetModelPath();
  const std::filesystem::path parent_path = model_path.parent_path();
  if (!parent_path.empty()) {
    return parent_path;
  }
  throw std::runtime_error(
      "AivisGgmlExecutionProvider needs ep.context_file_path to resolve "
      "external EPContext payloads or relative GGUF artifacts when the model "
      "is loaded from memory.");
}

std::string ResolveEpContextArtifactPath(
    const std::string& relative_path,
    const std::filesystem::path& base_directory,
    const char* artifact_name) {
  if (relative_path.empty()) {
    return "";
  }
  const std::filesystem::path path(relative_path);
  if (path.is_absolute() || HasParentDirectoryTraversal(path)) {
    throw std::runtime_error(
        std::string("Aivis GGML EPContext artifact path is not portable: ") +
        artifact_name);
  }
  std::error_code error;
  const std::filesystem::path absolute_base =
      std::filesystem::absolute(base_directory, error).lexically_normal();
  if (error) {
    throw std::runtime_error("could not resolve EPContext model directory.");
  }
  return (absolute_base / path).lexically_normal().string();
}

std::string ReadEpContextPayloadText(
    const AivisGgmlEpConfig& config,
    const OrtGraph* ort_graph,
    const Ort::ConstNode& node) {
  std::string ep_cache_context;
  if (!ReadStringNodeAttribute(node, "ep_cache_context", ep_cache_context)) {
    throw std::runtime_error("EPContext ep_cache_context attribute is missing or unreadable.");
  }
  int64_t embed_mode = 1;
  if (!ReadInt64NodeAttribute(node, "embed_mode", embed_mode)) {
    embed_mode = 1;
  }
  if (embed_mode == 1) {
    return ep_cache_context;
  }
  if (embed_mode != 0) {
    throw std::runtime_error("EPContext embed_mode must be 0 or 1.");
  }

  const std::filesystem::path base_directory =
      EpContextInferenceBaseDirectory(config, ort_graph);
  const std::string payload_path = ResolveEpContextArtifactPath(
      ep_cache_context,
      base_directory,
      "ep_cache_context");
  return ReadSmallTextFile(payload_path, 1024 * 1024);
}

struct EpContextPayload {
  AivisGgmlGraphKind graph_kind = AivisGgmlGraphKind::Synthesis;
  std::string backend;
  std::string device;
  std::string precision;
  std::string cache_manifest_path;
  std::string gguf_path;
  std::string jp_bert_gguf_path;
  int n_threads = 0;
};

EpContextPayload ParseEpContextPayload(
    const AivisGgmlEpConfig& config,
    const OrtGraph* ort_graph,
    const Ort::ConstNode& node,
    AivisGgmlGraphKind expected_graph_kind) {
  const std::string payload = ReadEpContextPayloadText(config, ort_graph, node);
  std::string value;
  if (!ExtractJsonStringField(payload, "version", value) ||
      value != kOfficialEpContextVersion) {
    throw std::runtime_error("EPContext payload version is unsupported.");
  }
  if (!ExtractJsonStringField(payload, "provider_name", value) || value != kEpName) {
    throw std::runtime_error("EPContext payload provider_name is unsupported.");
  }
  if (!ExtractJsonStringField(payload, "provider_version", value) || value != kVersion) {
    throw std::runtime_error("EPContext payload provider_version is unsupported.");
  }
  if (!ExtractJsonStringField(payload, "runtime_registry_contract", value) ||
      value != kRuntimeRegistryContract) {
    throw std::runtime_error("EPContext payload runtime_registry_contract is unsupported.");
  }
  if (!ExtractJsonStringField(payload, "tts_cpp_runtime_contract", value) ||
      value != kTtsCppRuntimeContract) {
    throw std::runtime_error("EPContext payload tts_cpp_runtime_contract is unsupported.");
  }
  if (!ExtractJsonStringField(payload, "graph_kind", value)) {
    throw std::runtime_error("EPContext payload graph_kind is missing.");
  }

  EpContextPayload parsed;
  if (value == "synthesis") {
    parsed.graph_kind = AivisGgmlGraphKind::Synthesis;
  } else if (value == "jp-bert") {
    parsed.graph_kind = AivisGgmlGraphKind::JpBert;
  } else {
    throw std::runtime_error("EPContext payload graph_kind is unsupported.");
  }
  if (parsed.graph_kind != expected_graph_kind) {
    throw std::runtime_error("EPContext payload graph_kind does not match partition_name.");
  }

  ExtractJsonStringField(payload, "backend", parsed.backend);
  ExtractJsonStringField(payload, "device", parsed.device);
  ExtractJsonStringField(payload, "precision", parsed.precision);
  ExtractJsonIntField(payload, "n_threads", parsed.n_threads);

  const std::filesystem::path base_directory =
      EpContextInferenceBaseDirectory(config, ort_graph);
  if (ExtractJsonStringField(payload, "cache_manifest_path", value)) {
    parsed.cache_manifest_path =
        ResolveEpContextArtifactPath(value, base_directory, "cache_manifest_path");
  }
  if (ExtractJsonStringField(payload, "gguf_path", value)) {
    parsed.gguf_path = ResolveEpContextArtifactPath(value, base_directory, "gguf_path");
  }
  if (ExtractJsonStringField(payload, "jp_bert_gguf_path", value)) {
    parsed.jp_bert_gguf_path =
        ResolveEpContextArtifactPath(value, base_directory, "jp_bert_gguf_path");
  }
  return parsed;
}

AivisGgmlEpConfig BuildConfigFromEpContextPayload(
    const AivisGgmlEpConfig& base_config,
    const OrtGraph* ort_graph,
    AivisGgmlGraphKind graph_kind) {
  Ort::ConstGraph graph{ort_graph};
  const std::vector<Ort::ConstNode> nodes = graph.GetNodes();
  if (nodes.size() != 1) {
    throw std::runtime_error("EPContext graph must contain exactly one node.");
  }

  const EpContextPayload payload =
      ParseEpContextPayload(base_config, ort_graph, nodes[0], graph_kind);
  AivisGgmlEpConfig runtime_config = base_config;
  runtime_config.eager_load_model = true;
  if (!payload.backend.empty()) {
    runtime_config.backend = payload.backend;
  }
  if (!payload.device.empty() && runtime_config.device.empty()) {
    runtime_config.device = payload.device;
  }
  if (!payload.precision.empty()) {
    runtime_config.precision = payload.precision;
  }
  if (runtime_config.n_threads == 0 && payload.n_threads > 0) {
    runtime_config.n_threads = payload.n_threads;
  }
  if (runtime_config.cache_manifest_path.empty()) {
    runtime_config.cache_manifest_path = payload.cache_manifest_path;
  }
  if (runtime_config.gguf_path.empty()) {
    runtime_config.gguf_path = payload.gguf_path;
  }
  if (runtime_config.jp_bert_gguf_path.empty()) {
    runtime_config.jp_bert_gguf_path = payload.jp_bert_gguf_path;
  }
  if (runtime_config.tts_cpp_library_path.empty()) {
    throw std::runtime_error(
        "EPContext lazy restore requires tts_cpp_library_path because the "
        "portable context payload does not store deployment-specific shared library paths.");
  }
  if (graph_kind == AivisGgmlGraphKind::Synthesis && runtime_config.gguf_path.empty()) {
    throw std::runtime_error("EPContext lazy restore could not resolve gguf_path.");
  }
  if (graph_kind == AivisGgmlGraphKind::JpBert && runtime_config.jp_bert_gguf_path.empty()) {
    throw std::runtime_error("EPContext lazy restore could not resolve jp_bert_gguf_path.");
  }
  if (!PathExists(runtime_config.tts_cpp_library_path)) {
    throw std::runtime_error("EPContext lazy restore tts_cpp_library_path does not exist.");
  }
  if (!runtime_config.gguf_path.empty() && !PathExists(runtime_config.gguf_path)) {
    throw std::runtime_error("EPContext lazy restore gguf_path does not exist.");
  }
  if (!runtime_config.jp_bert_gguf_path.empty() &&
      !PathExists(runtime_config.jp_bert_gguf_path)) {
    throw std::runtime_error("EPContext lazy restore jp_bert_gguf_path does not exist.");
  }
  return runtime_config;
}

bool CanLazyRestoreEpContextRuntime(
    const AivisGgmlEpConfig& config,
    const OrtGraph* ort_graph,
    AivisGgmlGraphKind graph_kind,
    std::string& reason) {
  try {
    (void)BuildConfigFromEpContextPayload(config, ort_graph, graph_kind);
    return true;
  } catch (const std::exception& ex) {
    reason = ex.what();
    return false;
  }
}

EpContextGateResult MatchAivisGgmlEpContextGraph(const OrtGraph* ort_graph) {
  EpContextGateResult result;
  Ort::ConstGraph graph{ort_graph};
  const std::vector<Ort::ConstNode> nodes = graph.GetNodes();
  if (nodes.size() != 1) {
    std::ostringstream reason;
    reason << "EPContext graph node count " << nodes.size() << " != 1";
    result.Reject(reason.str());
    return result;
  }

  const Ort::ConstNode node = nodes[0];
  if (node.GetOperatorType() != "EPContext" || node.GetDomain() != "com.microsoft") {
    std::ostringstream reason;
    reason << "single node is not com.microsoft::EPContext, got "
           << node.GetDomain() << "::" << node.GetOperatorType();
    result.Reject(reason.str());
    return result;
  }

  std::string source;
  if (!ReadStringNodeAttribute(node, "source", source)) {
    result.Reject("EPContext source attribute is missing or unreadable");
    return result;
  }
  if (source != kEpName) {
    std::ostringstream reason;
    reason << "EPContext source '" << source << "' != '" << kEpName << "'";
    result.Reject(reason.str());
    return result;
  }

  std::string partition_name;
  if (!ReadStringNodeAttribute(node, "partition_name", partition_name)) {
    result.Reject("EPContext partition_name attribute is missing or unreadable");
    return result;
  }

  const std::vector<Ort::ConstValueInfo> inputs = graph.GetInputs();
  const std::vector<Ort::ConstValueInfo> outputs = graph.GetOutputs();
  if (partition_name.rfind("synthesis", 0) == 0) {
    if (inputs.size() != kExpectedInputNames.size()) {
      std::ostringstream reason;
      reason << "EPContext synthesis input count " << inputs.size()
             << " != " << kExpectedInputNames.size();
      result.Reject(reason.str());
      return result;
    }
    if (outputs.size() != kExpectedOutputCount) {
      std::ostringstream reason;
      reason << "EPContext synthesis output count " << outputs.size()
             << " != " << kExpectedOutputCount;
      result.Reject(reason.str());
      return result;
    }
    result.Accept(AivisGgmlGraphKind::Synthesis);
    return result;
  }

  if (partition_name.rfind("jp-bert", 0) == 0) {
    if (inputs.size() != kExpectedJpBertInputCount) {
      std::ostringstream reason;
      reason << "EPContext JP-BERT input count " << inputs.size()
             << " != " << kExpectedJpBertInputCount;
      result.Reject(reason.str());
      return result;
    }
    if (outputs.size() != kExpectedJpBertOutputCount) {
      std::ostringstream reason;
      reason << "EPContext JP-BERT output count " << outputs.size()
             << " != " << kExpectedJpBertOutputCount;
      result.Reject(reason.str());
      return result;
    }
    result.Accept(AivisGgmlGraphKind::JpBert);
    return result;
  }

  std::ostringstream reason;
  reason << "EPContext partition_name '" << partition_name << "' is unsupported";
  result.Reject(reason.str());
  return result;
}

std::string DetectCompiledModelGraphKind(const OrtGraph* ort_graph) {
  if (ort_graph == nullptr) {
    return "unsupported";
  }

  const EpContextGateResult ep_context_gate = MatchAivisGgmlEpContextGraph(ort_graph);
  if (ep_context_gate.supported) {
    return GraphKindName(ep_context_gate.graph_kind);
  }

  const GraphSignatureGateResult synthesis_gate =
      MatchStyleBertVits2SynthesisGraph(ort_graph);
  if (synthesis_gate.supported) {
    return GraphKindName(AivisGgmlGraphKind::Synthesis);
  }

  const GraphSignatureGateResult jp_bert_gate =
      MatchStyleBertVits2JpBertGraph(ort_graph);
  if (jp_bert_gate.supported) {
    return GraphKindName(AivisGgmlGraphKind::JpBert);
  }

  return "unsupported";
}

std::string BuildCompiledModelCompatibilityInfo(
    const AivisGgmlEpConfig& config,
    const OrtGraph* ort_graph) {
  std::ostringstream out;
  out << "{";
  out << "\"version\":\"" << kCompiledModelCompatibilityVersion << "\"";
  out << ",\"provider_name\":\"" << kEpName << "\"";
  out << ",\"provider_version\":\"" << kVersion << "\"";
  out << ",\"ort_api_version\":" << ORT_API_VERSION;
  out << ",\"runtime_registry_contract\":\"" << kRuntimeRegistryContract << "\"";
  out << ",\"tts_cpp_runtime_contract\":\"" << kTtsCppRuntimeContract << "\"";
  out << ",\"tts_cpp_runtime_abi_version\":"
      << kExpectedTtsCppRuntimeAbiVersion;
  out << ",\"gguf_schema_version\":" << kExpectedTtsCppGgufSchemaVersion;
  out << ",\"model_signature_contract\":\"" << kSignatureContract << "\"";
  out << ",\"official_ep_context_payload_version\":\""
      << kOfficialEpContextVersion << "\"";
  out << ",\"graph_kind\":\"" << DetectCompiledModelGraphKind(ort_graph) << "\"";
  out << ",\"backend\":\"" << JsonEscape(config.backend) << "\"";
  out << ",\"device\":\"" << JsonEscape(config.device) << "\"";
  out << ",\"precision\":\"" << JsonEscape(config.precision) << "\"";
  out << "}";
  return out.str();
}

bool ExtractExpectedJsonStringField(
    const std::string& payload,
    const std::string& key,
    const char* expected_value) {
  std::string value;
  return ExtractJsonStringField(payload, key, value) && value == expected_value;
}

OrtCompiledModelCompatibility ValidateAivisCompiledModelCompatibilityInfo(
    const char* compatibility_info) noexcept {
  if (compatibility_info == nullptr || compatibility_info[0] == '\0') {
    return OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
  }

  try {
    const std::string payload{compatibility_info};
    std::string value;
    if (!ExtractJsonStringField(payload, "provider_name", value)) {
      return OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    }
    if (value != kEpName) {
      return OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    }

    if (!ExtractExpectedJsonStringField(
            payload,
            "version",
            kCompiledModelCompatibilityVersion) ||
        !ExtractExpectedJsonStringField(payload, "provider_version", kVersion) ||
        !ExtractExpectedJsonStringField(
            payload,
            "runtime_registry_contract",
            kRuntimeRegistryContract) ||
        !ExtractExpectedJsonStringField(
            payload,
            "tts_cpp_runtime_contract",
            kTtsCppRuntimeContract) ||
        !ExtractExpectedJsonStringField(
            payload,
            "model_signature_contract",
            kSignatureContract) ||
        !ExtractExpectedJsonStringField(
            payload,
            "official_ep_context_payload_version",
            kOfficialEpContextVersion)) {
      return OrtCompiledModelCompatibility_EP_UNSUPPORTED;
    }

    int ort_api_version = 0;
    if (!ExtractJsonIntField(payload, "ort_api_version", ort_api_version)) {
      return OrtCompiledModelCompatibility_EP_UNSUPPORTED;
    }
    bool prefer_recompilation = ort_api_version != ORT_API_VERSION;

    int runtime_abi_version = 0;
    if (!ExtractJsonIntField(
            payload,
            "tts_cpp_runtime_abi_version",
            runtime_abi_version) ||
        runtime_abi_version !=
            static_cast<int>(kExpectedTtsCppRuntimeAbiVersion)) {
      return OrtCompiledModelCompatibility_EP_UNSUPPORTED;
    }

    int gguf_schema_version = 0;
    if (!ExtractJsonIntField(
            payload,
            "gguf_schema_version",
            gguf_schema_version) ||
        gguf_schema_version != static_cast<int>(kExpectedTtsCppGgufSchemaVersion)) {
      return OrtCompiledModelCompatibility_EP_UNSUPPORTED;
    }

    if (!ExtractJsonStringField(payload, "graph_kind", value) ||
        (value != GraphKindName(AivisGgmlGraphKind::Synthesis) &&
         value != GraphKindName(AivisGgmlGraphKind::JpBert))) {
      return OrtCompiledModelCompatibility_EP_UNSUPPORTED;
    }

    if (ExtractJsonStringField(payload, "backend", value) &&
        !IsSupportedBackend(value)) {
      return OrtCompiledModelCompatibility_EP_UNSUPPORTED;
    }
    if (ExtractJsonStringField(payload, "precision", value) &&
        !IsSupportedPrecision(value)) {
      return OrtCompiledModelCompatibility_EP_UNSUPPORTED;
    }

    return prefer_recompilation
        ? OrtCompiledModelCompatibility_EP_SUPPORTED_PREFER_RECOMPILATION
        : OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL;
  } catch (...) {
    return OrtCompiledModelCompatibility_EP_UNSUPPORTED;
  }
}

struct AivisGgmlEp final : OrtEp {
  AivisGgmlEp(
      const OrtApi& ort_api,
      const OrtEpApi& ep_api,
      const OrtLogger* logger,
      AivisGgmlEpConfig config,
      std::shared_ptr<TtsCppRuntime> runtime)
      : OrtEp{},
        ort_api_{ort_api},
        ep_api_{ep_api},
        logger_{logger},
        config_{std::move(config)},
        runtime_{std::move(runtime)} {
    ort_version_supported = ORT_API_VERSION;
    GetName = GetNameImpl;
    GetCapability = GetCapabilityImpl;
    Compile = CompileImpl;
    ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
    GetCompiledModelCompatibilityInfo = GetCompiledModelCompatibilityInfoImpl;
  }

  static const char* ORT_API_CALL GetNameImpl(const OrtEp* /*this_ptr*/) noexcept {
    return kEpName;
  }

  static OrtStatus* ORT_API_CALL GetCapabilityImpl(
      OrtEp* this_ptr,
      const OrtGraph* graph,
      OrtEpGraphSupportInfo* graph_support_info) noexcept {
    auto* ep = static_cast<AivisGgmlEp*>(this_ptr);
    if (graph == nullptr) {
      LogMessage(
          ep->ort_api_,
          ep->logger_,
          ORT_LOGGING_LEVEL_WARNING,
          "AivisGgmlExecutionProvider received a null graph and claimed no nodes.");
      return nullptr;
    }

    try {
      const EpContextGateResult ep_context_gate = MatchAivisGgmlEpContextGraph(graph);
      if (ep_context_gate.supported) {
        const bool wants_graph =
            ep_context_gate.graph_kind == AivisGgmlGraphKind::Synthesis
                ? ep->config_.claim_synthesis_graph
                : ep->config_.claim_jp_bert_graph;
        const bool has_runtime =
            ep->runtime_ != nullptr &&
            (ep_context_gate.graph_kind == AivisGgmlGraphKind::Synthesis
                 ? ep->runtime_->HasSynthesisModel()
                 : ep->runtime_->HasJpBertModel());
        std::string lazy_restore_reason;
        const bool can_lazy_restore =
            !has_runtime &&
            CanLazyRestoreEpContextRuntime(
                ep->config_,
                graph,
                ep_context_gate.graph_kind,
                lazy_restore_reason);
        if (wants_graph && (has_runtime || can_lazy_restore)) {
          if (graph_support_info == nullptr) {
            return CreateStatus(
                ep->ort_api_,
                ORT_INVALID_ARGUMENT,
                "AivisGgmlExecutionProvider graph_support_info is null.");
          }
          const std::vector<Ort::ConstNode> nodes = Ort::ConstGraph{graph}.GetNodes();
          std::vector<const OrtNode*> raw_nodes;
          raw_nodes.reserve(nodes.size());
          for (const Ort::ConstNode& node : nodes) {
            raw_nodes.push_back(node);
          }

          OrtNodeFusionOptions fusion_options{};
          fusion_options.ort_version_supported = ORT_API_VERSION;
          fusion_options.drop_constant_initializers = true;
          OrtStatus* status = ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
              graph_support_info,
              raw_nodes.data(),
              raw_nodes.size(),
              &fusion_options);
          if (status != nullptr) {
            return status;
          }
          TraceMessage(
              std::string("claimed Aivis GGML EPContext graph_kind=") +
              GraphKindName(ep_context_gate.graph_kind));
          LogMessage(
              ep->ort_api_,
              ep->logger_,
              ORT_LOGGING_LEVEL_INFO,
              std::string("AivisGgmlExecutionProvider claimed an EPContext graph. graph_kind=") +
                  GraphKindName(ep_context_gate.graph_kind) +
                  ", runtime_ready=" + (has_runtime ? "true" : "false") +
                  ", lazy_restore=" + (can_lazy_restore ? "true" : "false") + ", " +
                  ConfigSummary(ep->config_));
          return nullptr;
        }
        LogMessage(
            ep->ort_api_,
            ep->logger_,
            ORT_LOGGING_LEVEL_INFO,
            std::string("AivisGgmlExecutionProvider matched an EPContext graph, ")
                + "but graph claiming or runtime readiness is disabled. graph_kind=" +
                GraphKindName(ep_context_gate.graph_kind) + ", " +
                "lazy_restore_reason=" + lazy_restore_reason + ", " +
                ConfigSummary(ep->config_));
        return nullptr;
      }

      const GraphSignatureGateResult gate = MatchStyleBertVits2SynthesisGraph(graph);
      if (gate.supported) {
        const std::string runtime_state =
            ep->runtime_ != nullptr ? "runtime_ready=true, " : "runtime_ready=false, ";
        if (ep->config_.claim_synthesis_graph) {
          if (ep->runtime_ == nullptr) {
            LogMessage(
                ep->ort_api_,
                ep->logger_,
                ORT_LOGGING_LEVEL_WARNING,
                "AivisGgmlExecutionProvider matched the Style-Bert-VITS2 synthesis graph, "
                "but claim_synthesis_graph was ignored because TTS.cpp runtime is not loaded.");
            return nullptr;
          }
          if (graph_support_info == nullptr) {
            return CreateStatus(
                ep->ort_api_,
                ORT_INVALID_ARGUMENT,
                "AivisGgmlExecutionProvider graph_support_info is null.");
          }

          const std::vector<Ort::ConstNode> nodes = Ort::ConstGraph{graph}.GetNodes();
          std::vector<const OrtNode*> raw_nodes;
          raw_nodes.reserve(nodes.size());
          for (const Ort::ConstNode& node : nodes) {
            raw_nodes.push_back(node);
          }

          OrtNodeFusionOptions fusion_options{};
          fusion_options.ort_version_supported = ORT_API_VERSION;
          fusion_options.drop_constant_initializers = true;
          OrtStatus* status = ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
              graph_support_info,
              raw_nodes.data(),
              raw_nodes.size(),
              &fusion_options);
          if (status != nullptr) {
            return status;
          }
          TraceMessage("claimed Style-Bert-VITS2 graph nodes=" + std::to_string(raw_nodes.size()));
          LogMessage(
              ep->ort_api_,
              ep->logger_,
              ORT_LOGGING_LEVEL_INFO,
              "AivisGgmlExecutionProvider claimed the Style-Bert-VITS2 synthesis graph. " +
                  runtime_state + ConfigSummary(ep->config_));
          return nullptr;
        }
        LogMessage(
            ep->ort_api_,
            ep->logger_,
            ORT_LOGGING_LEVEL_INFO,
            "AivisGgmlExecutionProvider matched the Style-Bert-VITS2 synthesis graph, "
            "but graph claiming is disabled until compile/compute is implemented. " +
                runtime_state +
                ConfigSummary(ep->config_));
      } else {
        const GraphSignatureGateResult jp_bert_gate = MatchStyleBertVits2JpBertGraph(graph);
        if (jp_bert_gate.supported) {
          const std::string runtime_state =
              ep->runtime_ != nullptr && ep->runtime_->HasJpBertModel()
                  ? "jp_bert_runtime_ready=true, "
                  : "jp_bert_runtime_ready=false, ";
          if (ep->config_.claim_jp_bert_graph) {
            if (ep->runtime_ == nullptr || !ep->runtime_->HasJpBertModel()) {
              LogMessage(
                  ep->ort_api_,
                  ep->logger_,
                  ORT_LOGGING_LEVEL_WARNING,
                  "AivisGgmlExecutionProvider matched the Style-Bert-VITS2 JP-BERT graph, "
                  "but claim_jp_bert_graph was ignored because TTS.cpp JP-BERT runtime is not loaded.");
              return nullptr;
            }
            if (graph_support_info == nullptr) {
              return CreateStatus(
                  ep->ort_api_,
                  ORT_INVALID_ARGUMENT,
                  "AivisGgmlExecutionProvider graph_support_info is null.");
            }

            const std::vector<Ort::ConstNode> nodes = Ort::ConstGraph{graph}.GetNodes();
            std::vector<const OrtNode*> raw_nodes;
            raw_nodes.reserve(nodes.size());
            for (const Ort::ConstNode& node : nodes) {
              raw_nodes.push_back(node);
            }

            OrtNodeFusionOptions fusion_options{};
            fusion_options.ort_version_supported = ORT_API_VERSION;
            fusion_options.drop_constant_initializers = true;
            OrtStatus* status = ep->ep_api_.EpGraphSupportInfo_AddNodesToFuse(
                graph_support_info,
                raw_nodes.data(),
                raw_nodes.size(),
                &fusion_options);
            if (status != nullptr) {
              return status;
            }
            TraceMessage("claimed Style-Bert-VITS2 JP-BERT graph nodes=" + std::to_string(raw_nodes.size()));
            LogMessage(
                ep->ort_api_,
                ep->logger_,
                ORT_LOGGING_LEVEL_INFO,
                "AivisGgmlExecutionProvider claimed the Style-Bert-VITS2 JP-BERT graph. " +
                    runtime_state + ConfigSummary(ep->config_));
            return nullptr;
          }
          LogMessage(
              ep->ort_api_,
              ep->logger_,
              ORT_LOGGING_LEVEL_INFO,
              "AivisGgmlExecutionProvider matched the Style-Bert-VITS2 JP-BERT graph, "
              "but claim_jp_bert_graph is disabled. " +
                  runtime_state +
                  ConfigSummary(ep->config_));
          return nullptr;
        }
        TraceMessage(
            "rejected graph: synthesis={" + JoinReasons(gate.reasons) +
            "}; jp-bert={" + JoinReasons(jp_bert_gate.reasons) +
            "}; ep-context={" + JoinReasons(ep_context_gate.reasons) + "}");
        LogMessage(
            ep->ort_api_,
            ep->logger_,
            ORT_LOGGING_LEVEL_VERBOSE,
            "AivisGgmlExecutionProvider rejected graph signatures and claimed no nodes: "
            "synthesis={" + JoinReasons(gate.reasons) + "}; jp-bert={" +
                JoinReasons(jp_bert_gate.reasons) + "}; ep-context={" +
                JoinReasons(ep_context_gate.reasons) + "}");
      }
    } catch (const Ort::Exception& ex) {
      LogMessage(
          ep->ort_api_,
          ep->logger_,
          ORT_LOGGING_LEVEL_WARNING,
          std::string("AivisGgmlExecutionProvider could not inspect graph and claimed no nodes: ") + ex.what());
    } catch (const std::exception& ex) {
      LogMessage(
          ep->ort_api_,
          ep->logger_,
          ORT_LOGGING_LEVEL_WARNING,
          std::string("AivisGgmlExecutionProvider graph inspection failed and claimed no nodes: ") + ex.what());
    } catch (...) {
      LogMessage(
          ep->ort_api_,
          ep->logger_,
          ORT_LOGGING_LEVEL_WARNING,
          "AivisGgmlExecutionProvider graph inspection failed with an unknown error and claimed no nodes.");
    }
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL CompileImpl(
      OrtEp* this_ptr,
      const OrtGraph** graphs,
      const OrtNode** fused_nodes,
      size_t count,
      OrtNodeComputeInfo** node_compute_infos,
      OrtNode** ep_context_nodes) noexcept {
    auto* ep = static_cast<AivisGgmlEp*>(this_ptr);
    for (size_t i = 0; i < count; ++i) {
      if (node_compute_infos != nullptr) {
        node_compute_infos[i] = nullptr;
      }
      if (ep_context_nodes != nullptr) {
        ep_context_nodes[i] = nullptr;
      }
    }
    if (!ep->config_.claim_synthesis_graph && !ep->config_.claim_jp_bert_graph) {
      return CreateStatus(
          ep->ort_api_,
          ORT_EP_FAIL,
          "AivisGgmlExecutionProvider Compile was called while graph claiming is disabled.");
    }
    if (node_compute_infos == nullptr) {
      return CreateStatus(
          ep->ort_api_,
          ORT_INVALID_ARGUMENT,
          "AivisGgmlExecutionProvider Compile received a null node_compute_infos array.");
    }
    if (ep->config_.ort_ep_context_enable && ep_context_nodes == nullptr) {
      return CreateStatus(
          ep->ort_api_,
          ORT_INVALID_ARGUMENT,
          "AivisGgmlExecutionProvider Compile received a null ep_context_nodes array.");
    }

    try {
      for (size_t i = 0; i < count; ++i) {
        (void) fused_nodes;
        if (graphs == nullptr || graphs[i] == nullptr) {
          throw std::runtime_error("AivisGgmlExecutionProvider Compile received a null graph.");
        }
        AivisGgmlGraphKind graph_kind = AivisGgmlGraphKind::Synthesis;
        const GraphSignatureGateResult synthesis_gate = MatchStyleBertVits2SynthesisGraph(graphs[i]);
        const GraphSignatureGateResult jp_bert_gate = MatchStyleBertVits2JpBertGraph(graphs[i]);
        const EpContextGateResult ep_context_gate = MatchAivisGgmlEpContextGraph(graphs[i]);
        std::vector<size_t> input_indices;
        size_t primary_output_index = 0;
        std::shared_ptr<TtsCppRuntime> runtime_for_graph = ep->runtime_;
        if (synthesis_gate.supported) {
          if (!ep->config_.claim_synthesis_graph) {
            throw std::runtime_error("AivisGgmlExecutionProvider Compile received a synthesis graph while claim_synthesis_graph is disabled.");
          }
          if (ep->runtime_ == nullptr || !ep->runtime_->HasSynthesisModel()) {
            throw std::runtime_error("AivisGgmlExecutionProvider Compile requires a loaded synthesis runtime.");
          }
          graph_kind = AivisGgmlGraphKind::Synthesis;
          input_indices = BuildInputIndices(graphs[i], kExpectedInputNames);
          primary_output_index = BuildOutputIndex(graphs[i], kExpectedFirstOutputName);
        } else if (jp_bert_gate.supported) {
          if (!ep->config_.claim_jp_bert_graph) {
            throw std::runtime_error("AivisGgmlExecutionProvider Compile received a JP-BERT graph while claim_jp_bert_graph is disabled.");
          }
          if (ep->runtime_ == nullptr || !ep->runtime_->HasJpBertModel()) {
            throw std::runtime_error("AivisGgmlExecutionProvider Compile requires a loaded JP-BERT runtime.");
          }
          graph_kind = AivisGgmlGraphKind::JpBert;
          input_indices = BuildInputIndices(graphs[i], kExpectedJpBertInputNames);
          primary_output_index = BuildOutputIndex(graphs[i], kExpectedJpBertOutputName);
        } else if (ep_context_gate.supported) {
          if (ep->config_.ort_ep_context_enable) {
            throw std::runtime_error("AivisGgmlExecutionProvider cannot generate an EPContext node from an EPContext graph.");
          }
          graph_kind = ep_context_gate.graph_kind;
          if (graph_kind == AivisGgmlGraphKind::Synthesis) {
            if (!ep->config_.claim_synthesis_graph) {
              throw std::runtime_error("AivisGgmlExecutionProvider Compile received a synthesis EPContext graph while claim_synthesis_graph is disabled.");
            }
            if (runtime_for_graph == nullptr || !runtime_for_graph->HasSynthesisModel()) {
              const AivisGgmlEpConfig runtime_config =
                  BuildConfigFromEpContextPayload(ep->config_, graphs[i], graph_kind);
              bool reused_runtime = false;
              runtime_for_graph = TtsCppRuntimeRegistry::Acquire(runtime_config, reused_runtime);
              if (ep->runtime_ == nullptr) {
                ep->runtime_ = runtime_for_graph;
              }
              TraceMessage(
                  std::string(reused_runtime ? "reused" : "lazy-loaded") +
                  " TTS.cpp runtime from synthesis EPContext payload");
            }
            input_indices = BuildInputIndices(graphs[i], kExpectedInputNames);
            primary_output_index = BuildOutputIndex(graphs[i], kExpectedFirstOutputName);
          } else {
            if (!ep->config_.claim_jp_bert_graph) {
              throw std::runtime_error("AivisGgmlExecutionProvider Compile received a JP-BERT EPContext graph while claim_jp_bert_graph is disabled.");
            }
            if (runtime_for_graph == nullptr || !runtime_for_graph->HasJpBertModel()) {
              const AivisGgmlEpConfig runtime_config =
                  BuildConfigFromEpContextPayload(ep->config_, graphs[i], graph_kind);
              bool reused_runtime = false;
              runtime_for_graph = TtsCppRuntimeRegistry::Acquire(runtime_config, reused_runtime);
              if (ep->runtime_ == nullptr) {
                ep->runtime_ = runtime_for_graph;
              }
              TraceMessage(
                  std::string(reused_runtime ? "reused" : "lazy-loaded") +
                  " TTS.cpp runtime from JP-BERT EPContext payload");
            }
            input_indices = BuildInputIndices(graphs[i], kExpectedJpBertInputNames);
            primary_output_index = BuildOutputIndex(graphs[i], kExpectedJpBertOutputName);
          }
        } else {
          throw std::runtime_error(
              "AivisGgmlExecutionProvider Compile received an unsupported graph signature: synthesis={" +
              JoinReasons(synthesis_gate.reasons) + "}; jp-bert={" +
              JoinReasons(jp_bert_gate.reasons) + "}; ep-context={" +
              JoinReasons(ep_context_gate.reasons) + "}");
        }
        if (ep->config_.ort_ep_context_enable) {
          OrtStatus* status = CreateEpContextNode(
              ep->ort_api_,
              ep->config_,
              graphs[i],
              fused_nodes != nullptr ? fused_nodes[i] : nullptr,
              graph_kind,
              i,
              &ep_context_nodes[i]);
          if (status != nullptr) {
            for (size_t cleanup_index = 0; cleanup_index < count; ++cleanup_index) {
              delete static_cast<AivisGgmlNodeComputeInfo*>(node_compute_infos[cleanup_index]);
              node_compute_infos[cleanup_index] = nullptr;
              if (ep_context_nodes[cleanup_index] != nullptr) {
                ep->ort_api_.ReleaseNode(ep_context_nodes[cleanup_index]);
                ep_context_nodes[cleanup_index] = nullptr;
              }
            }
            return status;
          }
          TraceMessage(
              std::string("created EPContext node graph_kind=") +
              GraphKindName(graph_kind) +
              " index=" + std::to_string(i));
          node_compute_infos[i] = new AivisGgmlNodeComputeInfo(
              ep->ort_api_,
              runtime_for_graph,
              graph_kind,
              std::move(input_indices),
              primary_output_index);
          continue;
        }
        node_compute_infos[i] = new AivisGgmlNodeComputeInfo(
            ep->ort_api_,
            runtime_for_graph,
            graph_kind,
            std::move(input_indices),
            primary_output_index);
      }
      TraceMessage("compiled fused graph count=" + std::to_string(count));
      return nullptr;
    } catch (const std::bad_alloc&) {
      for (size_t i = 0; i < count; ++i) {
        delete static_cast<AivisGgmlNodeComputeInfo*>(node_compute_infos[i]);
        node_compute_infos[i] = nullptr;
        if (ep_context_nodes != nullptr && ep_context_nodes[i] != nullptr) {
          ep->ort_api_.ReleaseNode(ep_context_nodes[i]);
          ep_context_nodes[i] = nullptr;
        }
      }
      return CreateStatus(ep->ort_api_, ORT_FAIL, "Out of memory compiling Aivis GGML graph.");
    } catch (const Ort::Exception& ex) {
      for (size_t i = 0; i < count; ++i) {
        delete static_cast<AivisGgmlNodeComputeInfo*>(node_compute_infos[i]);
        node_compute_infos[i] = nullptr;
        if (ep_context_nodes != nullptr && ep_context_nodes[i] != nullptr) {
          ep->ort_api_.ReleaseNode(ep_context_nodes[i]);
          ep_context_nodes[i] = nullptr;
        }
      }
      return CreateStatus(ep->ort_api_, ORT_FAIL, ex.what());
    } catch (const std::exception& ex) {
      for (size_t i = 0; i < count; ++i) {
        delete static_cast<AivisGgmlNodeComputeInfo*>(node_compute_infos[i]);
        node_compute_infos[i] = nullptr;
        if (ep_context_nodes != nullptr && ep_context_nodes[i] != nullptr) {
          ep->ort_api_.ReleaseNode(ep_context_nodes[i]);
          ep_context_nodes[i] = nullptr;
        }
      }
      return CreateStatus(ep->ort_api_, ORT_FAIL, ex.what());
    } catch (...) {
      for (size_t i = 0; i < count; ++i) {
        delete static_cast<AivisGgmlNodeComputeInfo*>(node_compute_infos[i]);
        node_compute_infos[i] = nullptr;
        if (ep_context_nodes != nullptr && ep_context_nodes[i] != nullptr) {
          ep->ort_api_.ReleaseNode(ep_context_nodes[i]);
          ep_context_nodes[i] = nullptr;
        }
      }
      return CreateStatus(ep->ort_api_, ORT_FAIL, "Unknown error compiling Aivis GGML graph.");
    }
  }

  static void ORT_API_CALL ReleaseNodeComputeInfosImpl(
      OrtEp* /*this_ptr*/,
      OrtNodeComputeInfo** node_compute_infos,
      size_t num_node_compute_infos) noexcept {
    if (node_compute_infos == nullptr) {
      return;
    }
    for (size_t i = 0; i < num_node_compute_infos; ++i) {
      delete static_cast<AivisGgmlNodeComputeInfo*>(node_compute_infos[i]);
      node_compute_infos[i] = nullptr;
    }
  }

  static const char* ORT_API_CALL GetCompiledModelCompatibilityInfoImpl(
      OrtEp* this_ptr,
      const OrtGraph* graph) noexcept {
    if (this_ptr == nullptr) {
      return nullptr;
    }
    auto* ep = static_cast<AivisGgmlEp*>(this_ptr);
    try {
      ep->compatibility_info_ =
          BuildCompiledModelCompatibilityInfo(ep->config_, graph);
    } catch (...) {
      ep->compatibility_info_ =
          BuildCompiledModelCompatibilityInfo(ep->config_, nullptr);
    }
    return ep->compatibility_info_.c_str();
  }

  const OrtApi& ort_api_;
  const OrtEpApi& ep_api_;
  const OrtLogger* logger_;
  AivisGgmlEpConfig config_;
  std::shared_ptr<TtsCppRuntime> runtime_;
  std::string compatibility_info_;
};

struct AivisGgmlEpFactory final : OrtEpFactory {
  AivisGgmlEpFactory(
      const OrtApi& ort_api,
      const OrtEpApi& ep_api,
      const OrtLogger* default_logger,
      std::string registration_name)
      : OrtEpFactory{},
        ort_api_{ort_api},
        ep_api_{ep_api},
        default_logger_{default_logger},
        registration_name_{std::move(registration_name)} {
    ort_version_supported = ORT_API_VERSION;
    GetName = GetNameImpl;
    GetVendor = GetVendorImpl;
    GetSupportedDevices = GetSupportedDevicesImpl;
    CreateEp = CreateEpImpl;
    ReleaseEp = ReleaseEpImpl;
    GetVendorId = GetVendorIdImpl;
    GetVersion = GetVersionImpl;
    ValidateCompiledModelCompatibilityInfo = ValidateCompiledModelCompatibilityInfoImpl;
    CreateAllocator = CreateAllocatorImpl;
    ReleaseAllocator = ReleaseAllocatorImpl;
    CreateDataTransfer = CreateDataTransferImpl;
    IsStreamAware = IsStreamAwareImpl;
    CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;
    GetHardwareDeviceIncompatibilityDetails = GetHardwareDeviceIncompatibilityDetailsImpl;
    CreateExternalResourceImporterForDevice = CreateExternalResourceImporterForDeviceImpl;
    GetNumCustomOpDomains = GetNumCustomOpDomainsImpl;
    GetCustomOpDomains = GetCustomOpDomainsImpl;
    InitGraphicsInterop = InitGraphicsInteropImpl;
    DeinitGraphicsInterop = DeinitGraphicsInteropImpl;
  }

  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return kEpName;
  }

  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return kVendor;
  }

  static uint32_t ORT_API_CALL GetVendorIdImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return 0;
  }

  static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return kVersion;
  }

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(
      OrtEpFactory* this_ptr,
      const OrtHardwareDevice* const* devices,
      size_t num_devices,
      OrtEpDevice** ep_devices,
      size_t max_ep_devices,
      size_t* p_num_ep_devices) noexcept {
    auto* factory = static_cast<AivisGgmlEpFactory*>(this_ptr);
    if (p_num_ep_devices == nullptr) {
      return CreateStatus(factory->ort_api_, ORT_INVALID_ARGUMENT, "num_ep_devices output is null.");
    }

    *p_num_ep_devices = 0;
    if (ep_devices == nullptr || max_ep_devices == 0) {
      return nullptr;
    }

    for (size_t i = 0; i < num_devices; ++i) {
      const OrtHardwareDevice* device = devices[i];
      if (device == nullptr ||
          factory->ort_api_.HardwareDevice_Type(device) != OrtHardwareDeviceType_CPU) {
        continue;
      }

      OrtKeyValuePairs* ep_metadata = nullptr;
      OrtKeyValuePairs* ep_options = nullptr;
      factory->ort_api_.CreateKeyValuePairs(&ep_metadata);
      factory->ort_api_.CreateKeyValuePairs(&ep_options);

      factory->ort_api_.AddKeyValuePair(ep_metadata, "aivis.stage", kStage);
      factory->ort_api_.AddKeyValuePair(ep_metadata, "aivis.runtime_registry_contract", kRuntimeRegistryContract);
      factory->ort_api_.AddKeyValuePair(ep_metadata, "aivis.tts_cpp_runtime_contract", kTtsCppRuntimeContract);
      factory->ort_api_.AddKeyValuePair(ep_metadata, "aivis.official_ep_context", "generation_supported");
      factory->ort_api_.AddKeyValuePair(ep_metadata, "aivis.official_ep_context_inference", "lazy_artifact_restore_tts_library_required");
      factory->ort_api_.AddKeyValuePair(ep_metadata, "registration_name", factory->registration_name_.c_str());
      factory->ort_api_.AddKeyValuePair(ep_metadata, "aivis.supported_backends", "vulkan,metal,cpu");
      factory->ort_api_.AddKeyValuePair(ep_metadata, "aivis.supported_precision", "accurate,fast");
      factory->ort_api_.AddKeyValuePair(ep_options, "backend", kDefaultBackend);
      factory->ort_api_.AddKeyValuePair(ep_options, "precision", kDefaultPrecision);
      factory->ort_api_.AddKeyValuePair(ep_options, "eager_load_model", "0");
      factory->ort_api_.AddKeyValuePair(ep_options, "claim_synthesis_graph", "0");
      factory->ort_api_.AddKeyValuePair(ep_options, "claim_jp_bert_graph", "0");
      factory->ort_api_.AddKeyValuePair(ep_options, "n_threads", "0");

      OrtEpDevice* ep_device = nullptr;
      OrtStatus* status = factory->ep_api_.CreateEpDevice(
          this_ptr,
          device,
          ep_metadata,
          ep_options,
          &ep_device);

      factory->ort_api_.ReleaseKeyValuePairs(ep_metadata);
      factory->ort_api_.ReleaseKeyValuePairs(ep_options);

      if (status != nullptr) {
        return status;
      }

      ep_devices[0] = ep_device;
      *p_num_ep_devices = 1;
      return nullptr;
    }

    return nullptr;
  }

  static OrtStatus* ORT_API_CALL CreateEpImpl(
      OrtEpFactory* this_ptr,
      const OrtHardwareDevice* const* /*devices*/,
      const OrtKeyValuePairs* const* /*ep_metadata_pairs*/,
      size_t num_devices,
      const OrtSessionOptions* session_options,
      const OrtLogger* logger,
      OrtEp** ep) noexcept {
    auto* factory = static_cast<AivisGgmlEpFactory*>(this_ptr);
    if (ep == nullptr) {
      return CreateStatus(factory->ort_api_, ORT_INVALID_ARGUMENT, "EP output is null.");
    }

    *ep = nullptr;
    if (num_devices != 1) {
      return CreateStatus(
          factory->ort_api_,
          ORT_INVALID_ARGUMENT,
          "AivisGgmlExecutionProvider bootstrap EP expects exactly one device.");
    }

    try {
      const OrtLogger* ep_logger = logger != nullptr ? logger : factory->default_logger_;
      AivisGgmlEpConfig config = ReadEpConfig(session_options);
      OrtStatus* status = ValidateEpConfig(factory->ort_api_, config);
      if (status != nullptr) {
        return status;
      }
      std::shared_ptr<TtsCppRuntime> runtime;
      if (config.eager_load_model) {
        bool reused_runtime = false;
        runtime = TtsCppRuntimeRegistry::Acquire(config, reused_runtime);
        TraceMessage(
            std::string(reused_runtime ? "reused" : "eager-loaded") +
            " TTS.cpp GGUF runtime");
        LogMessage(
            factory->ort_api_,
            ep_logger,
            ORT_LOGGING_LEVEL_INFO,
            std::string("AivisGgmlExecutionProvider ") +
                (reused_runtime ? "reused" : "eagerly loaded") +
                " configured TTS.cpp GGUF runtime. " +
                runtime->ContractSummary() + ".");
      }
      LogMessage(
          factory->ort_api_,
          ep_logger,
          ORT_LOGGING_LEVEL_INFO,
          "Creating AivisGgmlExecutionProvider with " + ConfigSummary(config) + ".");
      TraceMessage("CreateEp " + ConfigSummary(config));
      *ep = new AivisGgmlEp(
          factory->ort_api_,
          factory->ep_api_,
          ep_logger,
          std::move(config),
          std::move(runtime));
      return nullptr;
    } catch (const std::bad_alloc&) {
      return CreateStatus(factory->ort_api_, ORT_FAIL, "Out of memory creating Aivis GGML EP.");
    } catch (const std::exception& ex) {
      return CreateStatus(factory->ort_api_, ORT_FAIL, ex.what());
    } catch (...) {
      return CreateStatus(factory->ort_api_, ORT_FAIL, "Unknown error creating Aivis GGML EP.");
    }
  }

  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept {
    delete static_cast<AivisGgmlEp*>(ep);
  }

  static OrtStatus* ORT_API_CALL ValidateCompiledModelCompatibilityInfoImpl(
      OrtEpFactory* this_ptr,
      const OrtHardwareDevice* const* /*devices*/,
      size_t /*num_devices*/,
      const char* compatibility_info,
      OrtCompiledModelCompatibility* model_compatibility) noexcept {
    auto* factory = static_cast<AivisGgmlEpFactory*>(this_ptr);
    if (model_compatibility == nullptr) {
      return CreateStatus(
          factory->ort_api_,
          ORT_INVALID_ARGUMENT,
          "Compiled model compatibility output is null.");
    }

    *model_compatibility =
        ValidateAivisCompiledModelCompatibilityInfo(compatibility_info);
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(
      OrtEpFactory* /*this_ptr*/,
      const OrtMemoryInfo* /*memory_info*/,
      const OrtKeyValuePairs* /*allocator_options*/,
      OrtAllocator** allocator) noexcept {
    if (allocator != nullptr) {
      *allocator = nullptr;
    }
    return nullptr;
  }

  static void ORT_API_CALL ReleaseAllocatorImpl(
      OrtEpFactory* /*this_ptr*/,
      OrtAllocator* /*allocator*/) noexcept {}

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(
      OrtEpFactory* /*this_ptr*/,
      OrtDataTransferImpl** data_transfer) noexcept {
    if (data_transfer != nullptr) {
      *data_transfer = nullptr;
    }
    return nullptr;
  }

  static bool ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
    return false;
  }

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(
      OrtEpFactory* this_ptr,
      const OrtMemoryDevice* /*memory_device*/,
      const OrtKeyValuePairs* /*stream_options*/,
      OrtSyncStreamImpl** stream) noexcept {
    auto* factory = static_cast<AivisGgmlEpFactory*>(this_ptr);
    if (stream != nullptr) {
      *stream = nullptr;
    }
    return CreateStatus(factory->ort_api_, ORT_NOT_IMPLEMENTED, "Aivis GGML EP has no stream support yet.");
  }

  static OrtStatus* ORT_API_CALL GetHardwareDeviceIncompatibilityDetailsImpl(
      OrtEpFactory* /*this_ptr*/,
      const OrtHardwareDevice* /*hw*/,
      OrtDeviceEpIncompatibilityDetails* /*details*/) noexcept {
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL CreateExternalResourceImporterForDeviceImpl(
      OrtEpFactory* this_ptr,
      const OrtEpDevice* /*ep_device*/,
      OrtExternalResourceImporterImpl** out_importer) noexcept {
    auto* factory = static_cast<AivisGgmlEpFactory*>(this_ptr);
    if (out_importer != nullptr) {
      *out_importer = nullptr;
    }
    return CreateStatus(
        factory->ort_api_,
        ORT_NOT_IMPLEMENTED,
        "Aivis GGML EP has no external resource importer yet.");
  }

  static OrtStatus* ORT_API_CALL GetNumCustomOpDomainsImpl(
      OrtEpFactory* this_ptr,
      size_t* num_domains) noexcept {
    auto* factory = static_cast<AivisGgmlEpFactory*>(this_ptr);
    if (num_domains == nullptr) {
      return CreateStatus(factory->ort_api_, ORT_INVALID_ARGUMENT, "num_domains output is null.");
    }
    *num_domains = 0;
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL GetCustomOpDomainsImpl(
      OrtEpFactory* this_ptr,
      OrtCustomOpDomain** /*domains*/,
      size_t num_domains) noexcept {
    auto* factory = static_cast<AivisGgmlEpFactory*>(this_ptr);
    if (num_domains != 0) {
      return CreateStatus(
          factory->ort_api_,
          ORT_INVALID_ARGUMENT,
          "Aivis GGML EP bootstrap has no custom op domains.");
    }
    return nullptr;
  }

  static OrtStatus* ORT_API_CALL InitGraphicsInteropImpl(
      OrtEpFactory* this_ptr,
      const OrtEpDevice* /*ep_device*/,
      const OrtGraphicsInteropConfig* /*config*/) noexcept {
    auto* factory = static_cast<AivisGgmlEpFactory*>(this_ptr);
    return CreateStatus(
        factory->ort_api_,
        ORT_NOT_IMPLEMENTED,
        "Aivis GGML EP has no graphics interop support yet.");
  }

  static OrtStatus* ORT_API_CALL DeinitGraphicsInteropImpl(
      OrtEpFactory* this_ptr,
      const OrtEpDevice* /*ep_device*/) noexcept {
    auto* factory = static_cast<AivisGgmlEpFactory*>(this_ptr);
    return CreateStatus(
        factory->ort_api_,
        ORT_NOT_IMPLEMENTED,
        "Aivis GGML EP has no graphics interop support yet.");
  }

  const OrtApi& ort_api_;
  const OrtEpApi& ep_api_;
  const OrtLogger* default_logger_;
  std::string registration_name_;
};

}  // namespace

extern "C" {

AIVIS_GGML_EP_EXPORT OrtStatus* CreateEpFactories(
    const char* registration_name,
    const OrtApiBase* ort_api_base,
    const OrtLogger* default_logger,
    OrtEpFactory** factories,
    size_t max_factories,
    size_t* num_factories) {
  const OrtApi* ort_api = ort_api_base != nullptr ? ort_api_base->GetApi(ORT_API_VERSION) : nullptr;
  if (num_factories != nullptr) {
    *num_factories = 0;
  }
  if (ort_api == nullptr) {
    return nullptr;
  }
  if (factories == nullptr || num_factories == nullptr) {
    return CreateStatus(*ort_api, ORT_INVALID_ARGUMENT, "Factory output is null.");
  }
  if (max_factories < 1) {
    return CreateStatus(*ort_api, ORT_INVALID_ARGUMENT, "Not enough space to return EP factory.");
  }

  const OrtEpApi* ep_api = ort_api->GetEpApi();
  if (ep_api == nullptr) {
    return CreateStatus(*ort_api, ORT_FAIL, "ONNX Runtime did not provide OrtEpApi.");
  }
  Ort::InitApi(ort_api);

  try {
    auto factory = std::make_unique<AivisGgmlEpFactory>(
        *ort_api,
        *ep_api,
        default_logger,
        registration_name != nullptr ? registration_name : "");
    factories[0] = factory.release();
    *num_factories = 1;
    return nullptr;
  } catch (const std::bad_alloc&) {
    return CreateStatus(*ort_api, ORT_FAIL, "Out of memory creating Aivis GGML EP factory.");
  } catch (const std::exception& ex) {
    return CreateStatus(*ort_api, ORT_FAIL, ex.what());
  } catch (...) {
    return CreateStatus(*ort_api, ORT_FAIL, "Unknown error creating Aivis GGML EP factory.");
  }
}

AIVIS_GGML_EP_EXPORT OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<AivisGgmlEpFactory*>(factory);
  return nullptr;
}

}  // extern "C"
