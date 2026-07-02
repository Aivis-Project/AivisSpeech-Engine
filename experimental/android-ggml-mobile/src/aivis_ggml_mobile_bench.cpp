#include "tts_style_bert_vits2_c_api.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

namespace {

constexpr char kBundleMagic[] = "AIVISMB1";
constexpr uint32_t kBundleVersion = 1;

struct CaseInput {
    uint32_t role = 0;
    std::string label;
    std::string text;
    uint32_t tokens = 0;
    int32_t speaker_id = 0;
    uint32_t input_sample_rate = 0;
    float sdp_ratio = 0.0f;
    float length_scale = 1.0f;
    float noise_scale = 0.6f;
    float noise_w_scale = 0.8f;
    std::vector<int32_t> phone_ids;
    std::vector<int32_t> tone_ids;
    std::vector<int32_t> language_ids;
    std::vector<float> bert;
    std::vector<float> style_vec;
};

struct Record {
    std::string label;
    std::string text;
    int run_index = 0;
    double elapsed_seconds = 0.0;
    double output_duration_seconds = 0.0;
    size_t output_samples = 0;
    double rtf = 0.0;
    float peak_abs = 0.0f;
};

struct Args {
    std::string model_path;
    std::string bundle_path;
    std::string output_json;
    std::string audio_dir;
    std::string backend = "vulkan";
    std::string precision = "fast";
    int runs = 3;
    int warmup_runs = 1;
    int n_threads = 0;
    bool cpu_only = false;
};

void setenv_if_requested(const char * name, const std::string & value) {
    if (!value.empty()) {
        setenv(name, value.c_str(), 0);
    }
}

template<typename T>
T read_scalar(std::ifstream & input, const char * name) {
    T value{};
    input.read(reinterpret_cast<char *>(&value), sizeof(T));
    if (!input) {
        throw std::runtime_error(std::string("Failed to read ") + name);
    }
    return value;
}

void read_exact(std::ifstream & input, void * data, size_t bytes, const char * name) {
    if (bytes == 0) {
        return;
    }
    input.read(reinterpret_cast<char *>(data), static_cast<std::streamsize>(bytes));
    if (!input) {
        throw std::runtime_error(std::string("Failed to read ") + name);
    }
}

std::string read_string(std::ifstream & input, uint32_t length, const char * name) {
    std::string value(length, '\0');
    read_exact(input, value.data(), value.size(), name);
    return value;
}

std::vector<CaseInput> read_bundle(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open bundle: " + path);
    }

    char magic[8]{};
    read_exact(input, magic, sizeof(magic), "bundle magic");
    if (std::memcmp(magic, kBundleMagic, sizeof(magic)) != 0) {
        throw std::runtime_error("Unsupported bundle magic.");
    }

    const uint32_t version = read_scalar<uint32_t>(input, "bundle version");
    if (version != kBundleVersion) {
        throw std::runtime_error("Unsupported bundle version.");
    }
    const uint32_t case_count = read_scalar<uint32_t>(input, "case count");

    std::vector<CaseInput> cases;
    cases.reserve(case_count);
    for (uint32_t i = 0; i < case_count; ++i) {
        CaseInput item;
        item.role = read_scalar<uint32_t>(input, "role");
        const uint32_t label_len = read_scalar<uint32_t>(input, "label length");
        const uint32_t text_len = read_scalar<uint32_t>(input, "text length");
        item.tokens = read_scalar<uint32_t>(input, "tokens");
        item.speaker_id = read_scalar<int32_t>(input, "speaker id");
        item.input_sample_rate = read_scalar<uint32_t>(input, "input sample rate");
        item.sdp_ratio = read_scalar<float>(input, "sdp ratio");
        item.length_scale = read_scalar<float>(input, "length scale");
        item.noise_scale = read_scalar<float>(input, "noise scale");
        item.noise_w_scale = read_scalar<float>(input, "noise w scale");
        item.label = read_string(input, label_len, "label");
        item.text = read_string(input, text_len, "text");

        if (item.tokens == 0) {
            throw std::runtime_error("Case has zero tokens: " + item.label);
        }
        if (item.tokens > std::numeric_limits<uint32_t>::max() / 1024U) {
            throw std::runtime_error("Case token count overflows BERT size: " + item.label);
        }

        item.phone_ids.resize(item.tokens);
        item.tone_ids.resize(item.tokens);
        item.language_ids.resize(item.tokens);
        item.bert.resize(static_cast<size_t>(item.tokens) * 1024U);
        item.style_vec.resize(256);
        read_exact(input, item.phone_ids.data(), item.phone_ids.size() * sizeof(int32_t), "phone ids");
        read_exact(input, item.tone_ids.data(), item.tone_ids.size() * sizeof(int32_t), "tone ids");
        read_exact(input, item.language_ids.data(), item.language_ids.size() * sizeof(int32_t), "language ids");
        read_exact(input, item.bert.data(), item.bert.size() * sizeof(float), "bert");
        read_exact(input, item.style_vec.data(), item.style_vec.size() * sizeof(float), "style vec");
        cases.push_back(std::move(item));
    }

    return cases;
}

std::string json_escape(const std::string & value) {
    std::ostringstream out;
    for (unsigned char c : value) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
                } else {
                    out << c;
                }
        }
    }
    return out.str();
}

float peak_abs(const float * data, size_t length) {
    float peak = 0.0f;
    for (size_t i = 0; i < length; ++i) {
        peak = std::max(peak, std::fabs(data[i]));
    }
    return peak;
}

bool ensure_dir(const std::string & path) {
    if (path.empty()) {
        return true;
    }
    if (mkdir(path.c_str(), 0755) == 0 || errno == EEXIST) {
        return true;
    }
    return false;
}

void write_u16(std::ofstream & output, uint16_t value) {
    output.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void write_u32(std::ofstream & output, uint32_t value) {
    output.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void write_wav(const std::string & path, const float * samples, size_t sample_count, uint32_t sample_rate) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Failed to write WAV: " + path);
    }
    const uint16_t channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint16_t block_align = channels * bits_per_sample / 8;
    const uint32_t byte_rate = sample_rate * block_align;
    const uint32_t data_bytes = static_cast<uint32_t>(sample_count * block_align);
    output.write("RIFF", 4);
    write_u32(output, 36U + data_bytes);
    output.write("WAVE", 4);
    output.write("fmt ", 4);
    write_u32(output, 16);
    write_u16(output, 1);
    write_u16(output, channels);
    write_u32(output, sample_rate);
    write_u32(output, byte_rate);
    write_u16(output, block_align);
    write_u16(output, bits_per_sample);
    output.write("data", 4);
    write_u32(output, data_bytes);
    for (size_t i = 0; i < sample_count; ++i) {
        const float clamped = std::max(-1.0f, std::min(1.0f, samples[i]));
        const int16_t pcm = static_cast<int16_t>(std::lrintf(clamped * 32767.0f));
        output.write(reinterpret_cast<const char *>(&pcm), sizeof(pcm));
    }
}

Record run_case(tts_style_bert_vits2_handle * handle, const CaseInput & item, int run_index) {
    tts_style_bert_vits2_float_buffer audio{};
    const auto started_at = std::chrono::steady_clock::now();
    const int ok = tts_style_bert_vits2_synthesize_front_with_style_vec(
        handle,
        item.phone_ids.data(),
        item.tone_ids.data(),
        item.language_ids.data(),
        item.tokens,
        item.bert.data(),
        item.bert.size(),
        item.style_vec.data(),
        item.style_vec.size(),
        item.speaker_id,
        item.sdp_ratio,
        item.length_scale,
        item.noise_scale,
        item.noise_w_scale,
        &audio);
    const auto finished_at = std::chrono::steady_clock::now();
    if (!ok) {
        const char * error = tts_style_bert_vits2_last_error();
        throw std::runtime_error(std::string("Synthesis failed for ") + item.label + ": " + (error ? error : "<unknown>"));
    }
    if (!audio.data || audio.length == 0 || audio.sample_rate <= 0.0f) {
        throw std::runtime_error("Synthesis returned empty audio for " + item.label);
    }

    Record record;
    record.label = item.label;
    record.text = item.text;
    record.run_index = run_index;
    record.elapsed_seconds = std::chrono::duration<double>(finished_at - started_at).count();
    record.output_samples = audio.length;
    record.output_duration_seconds = static_cast<double>(audio.length) / static_cast<double>(audio.sample_rate);
    record.rtf = record.elapsed_seconds / record.output_duration_seconds;
    record.peak_abs = peak_abs(audio.data, audio.length);
    return record;
}

void print_usage(const char * argv0) {
    std::cerr
        << "Usage: " << argv0 << " --model model.gguf --bundle input.aivis_mobile_bundle [options]\n"
        << "Options:\n"
        << "  --backend vulkan|cpu      TTS_BACKEND value (default: vulkan)\n"
        << "  --precision fast|accurate STYLE_BERT_VITS2_VULKAN_PRECISION value (default: fast)\n"
        << "  --runs N                 measured runs per text (default: 3)\n"
        << "  --warmup-runs N          warmup runs before each text (default: 1)\n"
        << "  --n-threads N            TTS.cpp n_threads (default: auto)\n"
        << "  --cpu-only               pass cpu_only=1 to TTS.cpp\n"
        << "  --output-json PATH       write JSON results\n"
        << "  --audio-dir PATH         write first-run WAV previews\n";
}

Args parse_args(int argc, char ** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto require_value = [&](const char * name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };
        if (key == "--model") {
            args.model_path = require_value("--model");
        } else if (key == "--bundle") {
            args.bundle_path = require_value("--bundle");
        } else if (key == "--backend") {
            args.backend = require_value("--backend");
        } else if (key == "--precision") {
            args.precision = require_value("--precision");
        } else if (key == "--runs") {
            args.runs = std::stoi(require_value("--runs"));
        } else if (key == "--warmup-runs") {
            args.warmup_runs = std::stoi(require_value("--warmup-runs"));
        } else if (key == "--n-threads") {
            args.n_threads = std::stoi(require_value("--n-threads"));
        } else if (key == "--output-json") {
            args.output_json = require_value("--output-json");
        } else if (key == "--audio-dir") {
            args.audio_dir = require_value("--audio-dir");
        } else if (key == "--cpu-only") {
            args.cpu_only = true;
        } else if (key == "--help" || key == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + key);
        }
    }
    if (args.model_path.empty() || args.bundle_path.empty()) {
        print_usage(argv[0]);
        throw std::runtime_error("--model and --bundle are required.");
    }
    if (args.runs <= 0 || args.warmup_runs < 0) {
        throw std::runtime_error("--runs must be > 0 and --warmup-runs must be >= 0.");
    }
    if (args.n_threads <= 0) {
        const unsigned int detected_threads = std::thread::hardware_concurrency();
        args.n_threads = detected_threads > 0 ? static_cast<int>(detected_threads) : 4;
    }
    return args;
}

void write_json(const Args & args,
                const std::vector<CaseInput> & measured,
                const std::vector<Record> & records,
                const std::string & path) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Failed to write JSON: " + path);
    }
    output << std::fixed << std::setprecision(9);
    output << "{\n";
    output << "  \"profile\": {\n";
    output << "    \"backend\": \"" << json_escape(args.backend) << "\",\n";
    output << "    \"precision\": \"" << json_escape(args.precision) << "\",\n";
    output << "    \"runs\": " << args.runs << ",\n";
    output << "    \"warmup_runs\": " << args.warmup_runs << ",\n";
    output << "    \"cpu_only\": " << (args.cpu_only ? "true" : "false") << ",\n";
    output << "    \"model_path\": \"" << json_escape(args.model_path) << "\",\n";
    output << "    \"bundle_path\": \"" << json_escape(args.bundle_path) << "\"\n";
    output << "  },\n";

    output << "  \"summary\": [\n";
    for (size_t i = 0; i < measured.size(); ++i) {
        const auto & item = measured[i];
        std::vector<const Record *> group;
        for (const auto & record : records) {
            if (record.label == item.label) {
                group.push_back(&record);
            }
        }
        double sum = 0.0;
        double min_rtf = std::numeric_limits<double>::infinity();
        double max_rtf = 0.0;
        double duration_sum = 0.0;
        size_t last_samples = 0;
        for (const Record * record : group) {
            sum += record->rtf;
            min_rtf = std::min(min_rtf, record->rtf);
            max_rtf = std::max(max_rtf, record->rtf);
            duration_sum += record->output_duration_seconds;
            last_samples = record->output_samples;
        }
        const double mean = group.empty() ? 0.0 : sum / static_cast<double>(group.size());
        const double duration_mean = group.empty() ? 0.0 : duration_sum / static_cast<double>(group.size());
        output << "    {\n";
        output << "      \"text_label\": \"" << json_escape(item.label) << "\",\n";
        output << "      \"text\": \"" << json_escape(item.text) << "\",\n";
        output << "      \"tokens\": " << item.tokens << ",\n";
        output << "      \"rtf_mean\": " << mean << ",\n";
        output << "      \"rtf_min\": " << (std::isfinite(min_rtf) ? min_rtf : 0.0) << ",\n";
        output << "      \"rtf_max\": " << max_rtf << ",\n";
        output << "      \"output_duration_seconds_mean\": " << duration_mean << ",\n";
        output << "      \"output_samples_last\": " << last_samples << "\n";
        output << "    }" << (i + 1 == measured.size() ? "\n" : ",\n");
    }
    output << "  ],\n";

    output << "  \"records\": [\n";
    for (size_t i = 0; i < records.size(); ++i) {
        const auto & record = records[i];
        output << "    {\n";
        output << "      \"text_label\": \"" << json_escape(record.label) << "\",\n";
        output << "      \"text\": \"" << json_escape(record.text) << "\",\n";
        output << "      \"run_index\": " << record.run_index << ",\n";
        output << "      \"elapsed_seconds\": " << record.elapsed_seconds << ",\n";
        output << "      \"output_duration_seconds\": " << record.output_duration_seconds << ",\n";
        output << "      \"output_samples\": " << record.output_samples << ",\n";
        output << "      \"rtf\": " << record.rtf << ",\n";
        output << "      \"peak_abs\": " << record.peak_abs << "\n";
        output << "    }" << (i + 1 == records.size() ? "\n" : ",\n");
    }
    output << "  ]\n";
    output << "}\n";
}

} // namespace

int main(int argc, char ** argv) {
    try {
        Args args = parse_args(argc, argv);
        setenv_if_requested("TTS_BACKEND", args.backend);
        setenv_if_requested("STYLE_BERT_VITS2_VULKAN_PRECISION", args.precision);
        if (args.backend == "vulkan") {
            setenv("TTS_BACKEND_STRICT", "1", 0);
        }

        std::vector<CaseInput> cases = read_bundle(args.bundle_path);
        std::vector<CaseInput> warmups;
        std::vector<CaseInput> measured;
        for (const auto & item : cases) {
            if (item.role == 0) {
                warmups.push_back(item);
            } else {
                measured.push_back(item);
            }
        }
        if (measured.empty()) {
            throw std::runtime_error("Bundle does not contain measured cases.");
        }

        tts_style_bert_vits2_handle * handle = nullptr;
        if (!tts_style_bert_vits2_load_model(args.model_path.c_str(), args.n_threads, args.cpu_only ? 1 : 0, &handle)) {
            const char * error = tts_style_bert_vits2_last_error();
            throw std::runtime_error(std::string("Failed to load model: ") + (error ? error : "<unknown>"));
        }

        if (!args.audio_dir.empty() && !ensure_dir(args.audio_dir)) {
            throw std::runtime_error("Failed to create audio dir: " + args.audio_dir);
        }

        std::vector<Record> records;
        for (size_t i = 0; i < measured.size(); ++i) {
            const CaseInput & warmup = warmups.empty() ? measured[i] : warmups[std::min(i, warmups.size() - 1)];
            for (int w = 0; w < args.warmup_runs; ++w) {
                (void) run_case(handle, warmup, -1);
            }

            for (int run = 0; run < args.runs; ++run) {
                tts_style_bert_vits2_float_buffer audio{};
                const auto started_at = std::chrono::steady_clock::now();
                const int ok = tts_style_bert_vits2_synthesize_front_with_style_vec(
                    handle,
                    measured[i].phone_ids.data(),
                    measured[i].tone_ids.data(),
                    measured[i].language_ids.data(),
                    measured[i].tokens,
                    measured[i].bert.data(),
                    measured[i].bert.size(),
                    measured[i].style_vec.data(),
                    measured[i].style_vec.size(),
                    measured[i].speaker_id,
                    measured[i].sdp_ratio,
                    measured[i].length_scale,
                    measured[i].noise_scale,
                    measured[i].noise_w_scale,
                    &audio);
                const auto finished_at = std::chrono::steady_clock::now();
                if (!ok) {
                    const char * error = tts_style_bert_vits2_last_error();
                    throw std::runtime_error(std::string("Synthesis failed for ") + measured[i].label + ": " + (error ? error : "<unknown>"));
                }
                Record record;
                record.label = measured[i].label;
                record.text = measured[i].text;
                record.run_index = run;
                record.elapsed_seconds = std::chrono::duration<double>(finished_at - started_at).count();
                record.output_samples = audio.length;
                record.output_duration_seconds = static_cast<double>(audio.length) / static_cast<double>(audio.sample_rate);
                record.rtf = record.elapsed_seconds / record.output_duration_seconds;
                record.peak_abs = peak_abs(audio.data, audio.length);
                std::cout << record.label
                          << " run=" << run
                          << " elapsed=" << record.elapsed_seconds
                          << " duration=" << record.output_duration_seconds
                          << " rtf=" << record.rtf
                          << " samples=" << record.output_samples
                          << "\n";

                if (run == 0 && !args.audio_dir.empty()) {
                    write_wav(args.audio_dir + "/" + measured[i].label + ".wav",
                              audio.data,
                              audio.length,
                              static_cast<uint32_t>(audio.sample_rate));
                }
                records.push_back(record);
            }
        }

        if (!args.output_json.empty()) {
            write_json(args, measured, records, args.output_json);
        }
        tts_style_bert_vits2_free_model(handle);
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
