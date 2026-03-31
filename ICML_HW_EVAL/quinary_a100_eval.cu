#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace bench {

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err__ = (call);                                                         \
        if (err__ != cudaSuccess) {                                                         \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)                      \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";              \
            std::exit(1);                                                                   \
        }                                                                                   \
    } while (0)

#define CUBLAS_CHECK(call)                                                                  \
    do {                                                                                    \
        cublasStatus_t st__ = (call);                                                       \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                                \
            std::cerr << "cuBLAS error: status=" << static_cast<int>(st__)                \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";              \
            std::exit(1);                                                                   \
        }                                                                                   \
    } while (0)

struct Options {
    int n = 1024;
    int iters = 300;
    int warmup = 30;
    double zero_prob = 0.40;
    double sample_ms = 5.0;
    uint32_t seed_a = 1234;
    uint32_t seed_b = 4321;
    bool run_fp16 = false;
    bool run_bf16 = true;
    bool run_int8_tc = true;
    bool run_dense_cuda_int8 = true;
    bool run_opcode_adddbl = true;
    bool run_opcode_shift = true;
    bool verify_each = false;
    bool append_csv = false;
    std::vector<int> sweep_n;
    std::vector<double> sweep_zero_prob;
    std::string csv_path;
};

struct OpcodeCSR {
    std::vector<int> row_ptr_pos2;
    std::vector<int> row_ptr_pos1;
    std::vector<int> row_ptr_neg1;
    std::vector<int> row_ptr_neg2;
    std::vector<int> idx_pos2;
    std::vector<int> idx_pos1;
    std::vector<int> idx_neg1;
    std::vector<int> idx_neg2;
};

struct DeviceOpcodeCSR {
    int* row_ptr_pos2 = nullptr;
    int* row_ptr_pos1 = nullptr;
    int* row_ptr_neg1 = nullptr;
    int* row_ptr_neg2 = nullptr;
    int* idx_pos2 = nullptr;
    int* idx_pos1 = nullptr;
    int* idx_neg1 = nullptr;
    int* idx_neg2 = nullptr;
};

struct DeviceInfo {
    std::string gpu_name;
    int cc_major = 0;
    int cc_minor = 0;
};

struct BenchmarkResult {
    std::string gpu_name;
    int cc_major = 0;
    int cc_minor = 0;
    std::string method;
    int n = 0;
    double zero_prob = 0.0;
    double zero_ratio = 0.0;
    int iters = 0;
    int warmup = 0;
    double sample_ms = 0.0;
    uint32_t seed_a = 0;
    uint32_t seed_b = 0;
    int verify_performed = 0;
    double gpu_ms_total = 0.0;
    double gpu_ms_per_iter = 0.0;
    double wall_ms_total = 0.0;
    double wall_ms_per_iter = 0.0;
    double throughput_inf_s = 0.0;
    double avg_power_w = -1.0;
    double energy_j_total = -1.0;
    double energy_j_per_iter = -1.0;
};

struct PowerSample {
    std::chrono::steady_clock::time_point t;
    double power_w = 0.0;
};

class NvmlEnergySampler {
public:
    NvmlEnergySampler() = default;
    ~NvmlEnergySampler() { shutdown(); }

    bool init_from_current_cuda_device() {
        if (initialized_) return ok_;
        if (nvmlInit_v2() != NVML_SUCCESS) {
            ok_ = false;
            return false;
        }
        char pci_bus_id[32] = {0};
        int cuda_device = 0;
        CUDA_CHECK(cudaGetDevice(&cuda_device));
        CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), cuda_device));
        if (nvmlDeviceGetHandleByPciBusId_v2(pci_bus_id, &device_) != NVML_SUCCESS) {
            nvmlShutdown();
            ok_ = false;
            return false;
        }
        initialized_ = true;
        ok_ = true;
        return true;
    }

    bool ok() const { return ok_; }

    void start(double interval_ms) {
        if (!ok_) return;
        interval_ms_ = std::max(1.0, interval_ms);
        samples_.clear();
        running_.store(true);
        worker_ = std::thread([this]() { run(); });
    }

    double stop_and_integrate_joules() {
        if (!ok_) return -1.0;
        running_.store(false);
        if (worker_.joinable()) worker_.join();
        if (samples_.size() < 2) return 0.0;
        double joules = 0.0;
        for (size_t i = 1; i < samples_.size(); ++i) {
            const double dt = std::chrono::duration<double>(samples_[i].t - samples_[i - 1].t).count();
            const double pw = 0.5 * (samples_[i - 1].power_w + samples_[i].power_w);
            joules += pw * dt;
        }
        return joules;
    }

    void shutdown() {
        running_.store(false);
        if (worker_.joinable()) worker_.join();
        if (initialized_) {
            nvmlShutdown();
            initialized_ = false;
        }
        ok_ = false;
    }

private:
    void run() {
        sample_once();
        while (running_.load()) {
            std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(interval_ms_));
            sample_once();
        }
        sample_once();
    }

    void sample_once() {
        unsigned int power_mw = 0;
        if (nvmlDeviceGetPowerUsage(device_, &power_mw) == NVML_SUCCESS) {
            samples_.push_back({std::chrono::steady_clock::now(), static_cast<double>(power_mw) / 1000.0});
        }
    }

    bool initialized_ = false;
    bool ok_ = false;
    double interval_ms_ = 5.0;
    std::atomic<bool> running_{false};
    nvmlDevice_t device_ = nullptr;
    std::thread worker_;
    std::vector<PowerSample> samples_;
};

static std::string trim_copy(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

static std::string lower_ascii(std::string s) {
    for (char& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return s;
}

static std::vector<std::string> split_csv_tokens(const std::string& csv) {
    std::vector<std::string> out;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        tok = trim_copy(tok);
        if (!tok.empty()) out.push_back(tok);
    }
    return out;
}

static std::vector<int> parse_int_list(const std::string& csv) {
    std::vector<int> out;
    for (const auto& tok : split_csv_tokens(csv)) out.push_back(std::stoi(tok));
    return out;
}

static std::vector<double> parse_double_list(const std::string& csv) {
    std::vector<double> out;
    for (const auto& tok : split_csv_tokens(csv)) out.push_back(std::stod(tok));
    return out;
}

static void apply_modes_csv(Options& opt, const std::string& csv) {
    opt.run_fp16 = false;
    opt.run_bf16 = false;
    opt.run_int8_tc = false;
    opt.run_dense_cuda_int8 = false;
    opt.run_opcode_adddbl = false;
    opt.run_opcode_shift = false;
    for (auto tok : split_csv_tokens(csv)) {
        tok = lower_ascii(tok);
        if (tok == "fp16") opt.run_fp16 = true;
        else if (tok == "bf16") opt.run_bf16 = true;
        else if (tok == "int8" || tok == "int8_tc") opt.run_int8_tc = true;
        else if (tok == "dense_cuda" || tok == "dense_cuda_int8") opt.run_dense_cuda_int8 = true;
        else if (tok == "opcode" || tok == "opcode_adddbl") opt.run_opcode_adddbl = true;
        else if (tok == "opcode_shift") opt.run_opcode_shift = true;
        else {
            std::cerr << "Unknown mode in --modes: " << tok << "\n";
            std::exit(2);
        }
    }
}

static void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "  --n <int>                         Matrix size when not sweeping (default: 1024)\n"
        << "  --iters <int>                     Timed iterations (default: 300)\n"
        << "  --warmup <int>                    Warmup iterations (default: 30)\n"
        << "  --zero-prob <float>               Zero probability for A_code (default: 0.40)\n"
        << "  --sample-ms <float>               NVML polling interval in ms (default: 5.0)\n"
        << "  --seed-a <int>                    RNG seed for A_code (default: 1234)\n"
        << "  --seed-b <int>                    RNG seed for B activations (default: 4321)\n"
        << "  --modes <csv>                     Comma list from {fp16,bf16,int8,dense_cuda_int8,opcode_adddbl,opcode_shift}\n"
        << "  --sweep-n <csv>                   Sweep N values, e.g. 512,1024,2048\n"
        << "  --sweep-zero-prob <csv>           Sweep zero_prob values, e.g. 0.2,0.4,0.6,0.8\n"
        << "  --csv <path>                      Write/append benchmark rows as CSV\n"
        << "  --append-csv                      Append to existing CSV instead of overwriting\n"
        << "  --verify-each                     Run correctness checks at every sweep point\n"
        << "  --no-fp16                         Disable FP16 Tensor Core dense baseline\n"
        << "  --no-bf16                         Disable BF16 Tensor Core dense baseline\n"
        << "  --no-int8                         Disable INT8 Tensor Core dense baseline\n"
        << "  --no-dense-cuda                   Disable INT8 CUDA-core dense baseline\n"
        << "  --no-opcode-adddbl                Disable opcode add-doubling kernel\n"
        << "  --no-opcode-shift                 Disable opcode literal-shift kernel\n"
        << "  --help                            Show this message\n";
}

static Options parse_options(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                std::exit(2);
            }
        };
        if (arg == "--n") {
            require_value("--n");
            opt.n = std::stoi(argv[++i]);
        } else if (arg == "--iters") {
            require_value("--iters");
            opt.iters = std::stoi(argv[++i]);
        } else if (arg == "--warmup") {
            require_value("--warmup");
            opt.warmup = std::stoi(argv[++i]);
        } else if (arg == "--zero-prob") {
            require_value("--zero-prob");
            opt.zero_prob = std::stod(argv[++i]);
        } else if (arg == "--sample-ms") {
            require_value("--sample-ms");
            opt.sample_ms = std::stod(argv[++i]);
        } else if (arg == "--seed-a") {
            require_value("--seed-a");
            opt.seed_a = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--seed-b") {
            require_value("--seed-b");
            opt.seed_b = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--modes") {
            require_value("--modes");
            apply_modes_csv(opt, argv[++i]);
        } else if (arg == "--sweep-n") {
            require_value("--sweep-n");
            opt.sweep_n = parse_int_list(argv[++i]);
        } else if (arg == "--sweep-zero-prob") {
            require_value("--sweep-zero-prob");
            opt.sweep_zero_prob = parse_double_list(argv[++i]);
        } else if (arg == "--csv") {
            require_value("--csv");
            opt.csv_path = argv[++i];
        } else if (arg == "--append-csv") {
            opt.append_csv = true;
        } else if (arg == "--verify-each") {
            opt.verify_each = true;
        } else if (arg == "--no-fp16") {
            opt.run_fp16 = false;
        } else if (arg == "--no-bf16") {
            opt.run_bf16 = false;
        } else if (arg == "--no-int8") {
            opt.run_int8_tc = false;
        } else if (arg == "--no-dense-cuda") {
            opt.run_dense_cuda_int8 = false;
        } else if (arg == "--no-opcode-adddbl") {
            opt.run_opcode_adddbl = false;
        } else if (arg == "--no-opcode-shift") {
            opt.run_opcode_shift = false;
        } else if (arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            std::exit(2);
        }
    }

    if (opt.n <= 0 || opt.iters <= 0 || opt.warmup < 0) {
        std::cerr << "n/iters must be positive and warmup must be non-negative.\n";
        std::exit(2);
    }
    if (opt.zero_prob < 0.0 || opt.zero_prob >= 1.0) {
        std::cerr << "zero_prob must be in [0,1).\n";
        std::exit(2);
    }
    if (opt.sweep_n.empty()) opt.sweep_n.push_back(opt.n);
    if (opt.sweep_zero_prob.empty()) opt.sweep_zero_prob.push_back(opt.zero_prob);
    if (!opt.run_fp16 && !opt.run_bf16 && !opt.run_int8_tc && !opt.run_dense_cuda_int8 &&
        !opt.run_opcode_adddbl && !opt.run_opcode_shift) {
        std::cerr << "At least one method must remain enabled.\n";
        std::exit(2);
    }
    return opt;
}

static DeviceInfo get_device_info() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    DeviceInfo info;
    info.gpu_name = prop.name;
    info.cc_major = prop.major;
    info.cc_minor = prop.minor;
    return info;
}

static void print_device_info(const DeviceInfo& info) {
    std::cout << "GPU             : " << info.gpu_name << "\n";
    std::cout << "Compute cap.    : " << info.cc_major << "." << info.cc_minor << "\n";
    if (!(info.cc_major == 8 && info.cc_minor == 0)) {
        std::cout << "[warn] This code is tuned for A100/SM80. Running on a different GPU is allowed, but results may differ.\n";
    }
}

static void init_quinary_codes(std::vector<int8_t>& a_code, double zero_prob, uint32_t seed) {
    std::mt19937 rng(seed);
    const int8_t vals[5] = {-2, -1, 0, 1, 2};
    const double nz = (1.0 - zero_prob) * 0.25;
    const double probs[5] = {nz, nz, zero_prob, nz, nz};
    std::discrete_distribution<int> dist(probs, probs + 5);
    for (auto& v : a_code) v = vals[dist(rng)];
}

static void init_activation_int8(std::vector<int8_t>& b, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-127, 127);
    for (auto& v : b) v = static_cast<int8_t>(dist(rng));
}

static void convert_to_fp16_weights(const std::vector<int8_t>& a_code, std::vector<__half>& out) {
    out.resize(a_code.size());
    for (size_t i = 0; i < a_code.size(); ++i) out[i] = __float2half_rn(0.5f * static_cast<float>(a_code[i]));
}

static void convert_to_fp16_activations(const std::vector<int8_t>& b, std::vector<__half>& out) {
    out.resize(b.size());
    for (size_t i = 0; i < b.size(); ++i) out[i] = __float2half_rn(static_cast<float>(b[i]));
}

static void convert_to_bf16_weights(const std::vector<int8_t>& a_code, std::vector<__nv_bfloat16>& out) {
    out.resize(a_code.size());
    for (size_t i = 0; i < a_code.size(); ++i) out[i] = __float2bfloat16(0.5f * static_cast<float>(a_code[i]));
}

static void convert_to_bf16_activations(const std::vector<int8_t>& b, std::vector<__nv_bfloat16>& out) {
    out.resize(b.size());
    for (size_t i = 0; i < b.size(); ++i) out[i] = __float2bfloat16(static_cast<float>(b[i]));
}

static OpcodeCSR build_opcode_csr(const std::vector<int8_t>& a_code, int n) {
    OpcodeCSR csr;
    csr.row_ptr_pos2.assign(n + 1, 0);
    csr.row_ptr_pos1.assign(n + 1, 0);
    csr.row_ptr_neg1.assign(n + 1, 0);
    csr.row_ptr_neg2.assign(n + 1, 0);

    for (int i = 0; i < n; ++i) {
        const int base = i * n;
        for (int k = 0; k < n; ++k) {
            switch (a_code[base + k]) {
                case 2:  ++csr.row_ptr_pos2[i + 1]; break;
                case 1:  ++csr.row_ptr_pos1[i + 1]; break;
                case -1: ++csr.row_ptr_neg1[i + 1]; break;
                case -2: ++csr.row_ptr_neg2[i + 1]; break;
                default: break;
            }
        }
    }

    std::partial_sum(csr.row_ptr_pos2.begin(), csr.row_ptr_pos2.end(), csr.row_ptr_pos2.begin());
    std::partial_sum(csr.row_ptr_pos1.begin(), csr.row_ptr_pos1.end(), csr.row_ptr_pos1.begin());
    std::partial_sum(csr.row_ptr_neg1.begin(), csr.row_ptr_neg1.end(), csr.row_ptr_neg1.begin());
    std::partial_sum(csr.row_ptr_neg2.begin(), csr.row_ptr_neg2.end(), csr.row_ptr_neg2.begin());

    csr.idx_pos2.resize(csr.row_ptr_pos2.back());
    csr.idx_pos1.resize(csr.row_ptr_pos1.back());
    csr.idx_neg1.resize(csr.row_ptr_neg1.back());
    csr.idx_neg2.resize(csr.row_ptr_neg2.back());

    auto cur_pos2 = csr.row_ptr_pos2;
    auto cur_pos1 = csr.row_ptr_pos1;
    auto cur_neg1 = csr.row_ptr_neg1;
    auto cur_neg2 = csr.row_ptr_neg2;

    for (int i = 0; i < n; ++i) {
        const int base = i * n;
        for (int k = 0; k < n; ++k) {
            switch (a_code[base + k]) {
                case 2:  csr.idx_pos2[cur_pos2[i]++] = k; break;
                case 1:  csr.idx_pos1[cur_pos1[i]++] = k; break;
                case -1: csr.idx_neg1[cur_neg1[i]++] = k; break;
                case -2: csr.idx_neg2[cur_neg2[i]++] = k; break;
                default: break;
            }
        }
    }
    return csr;
}

static double compute_zero_ratio(const OpcodeCSR& csr, int n) {
    const size_t total = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t nonzero = csr.idx_pos2.size() + csr.idx_pos1.size() + csr.idx_neg1.size() + csr.idx_neg2.size();
    return static_cast<double>(total - nonzero) / static_cast<double>(total);
}

static void copy_vector_to_device(const std::vector<int>& src, int** dst) {
    if (src.empty()) {
        *dst = nullptr;
        return;
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(dst), src.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(*dst, src.data(), src.size() * sizeof(int), cudaMemcpyHostToDevice));
}

static DeviceOpcodeCSR upload_opcode_csr(const OpcodeCSR& csr) {
    DeviceOpcodeCSR d;
    copy_vector_to_device(csr.row_ptr_pos2, &d.row_ptr_pos2);
    copy_vector_to_device(csr.row_ptr_pos1, &d.row_ptr_pos1);
    copy_vector_to_device(csr.row_ptr_neg1, &d.row_ptr_neg1);
    copy_vector_to_device(csr.row_ptr_neg2, &d.row_ptr_neg2);
    copy_vector_to_device(csr.idx_pos2, &d.idx_pos2);
    copy_vector_to_device(csr.idx_pos1, &d.idx_pos1);
    copy_vector_to_device(csr.idx_neg1, &d.idx_neg1);
    copy_vector_to_device(csr.idx_neg2, &d.idx_neg2);
    return d;
}

static void free_device_opcode_csr(DeviceOpcodeCSR& d) {
    if (d.row_ptr_pos2) CUDA_CHECK(cudaFree(d.row_ptr_pos2));
    if (d.row_ptr_pos1) CUDA_CHECK(cudaFree(d.row_ptr_pos1));
    if (d.row_ptr_neg1) CUDA_CHECK(cudaFree(d.row_ptr_neg1));
    if (d.row_ptr_neg2) CUDA_CHECK(cudaFree(d.row_ptr_neg2));
    if (d.idx_pos2) CUDA_CHECK(cudaFree(d.idx_pos2));
    if (d.idx_pos1) CUDA_CHECK(cudaFree(d.idx_pos1));
    if (d.idx_neg1) CUDA_CHECK(cudaFree(d.idx_neg1));
    if (d.idx_neg2) CUDA_CHECK(cudaFree(d.idx_neg2));
    d = {};
}

static int32_t reference_entry_int(const std::vector<int8_t>& a_code, const std::vector<int8_t>& b, int row, int col, int n) {
    int32_t acc = 0;
    const int base = row * n;
    for (int k = 0; k < n; ++k) acc += static_cast<int32_t>(a_code[base + k]) * static_cast<int32_t>(b[k * n + col]);
    return acc;
}

static float reference_entry_real(const std::vector<int8_t>& a_code, const std::vector<int8_t>& b, int row, int col, int n) {
    float acc = 0.0f;
    const int base = row * n;
    for (int k = 0; k < n; ++k) acc += 0.5f * static_cast<float>(a_code[base + k]) * static_cast<float>(b[k * n + col]);
    return acc;
}

static bool spot_check_exact_i32(const std::vector<int8_t>& a_code,
                                 const std::vector<int8_t>& b,
                                 const std::vector<int32_t>& c,
                                 int n,
                                 int checks,
                                 uint32_t seed,
                                 std::string& msg) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, n - 1);
    for (int t = 0; t < checks; ++t) {
        const int i = dist(rng);
        const int j = dist(rng);
        const int32_t ref = reference_entry_int(a_code, b, i, j, n);
        const int32_t got = c[i * n + j];
        if (ref != got) {
            msg = "Mismatch at (" + std::to_string(i) + "," + std::to_string(j) + "): ref=" +
                  std::to_string(ref) + ", got=" + std::to_string(got);
            return false;
        }
    }
    msg = "Spot checks passed";
    return true;
}

static bool spot_check_float(const std::vector<int8_t>& a_code,
                             const std::vector<int8_t>& b,
                             const std::vector<float>& c,
                             int n,
                             int checks,
                             uint32_t seed,
                             float tol,
                             std::string& msg) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, n - 1);
    float max_abs_err = 0.0f;
    for (int t = 0; t < checks; ++t) {
        const int i = dist(rng);
        const int j = dist(rng);
        const float ref = reference_entry_real(a_code, b, i, j, n);
        const float got = c[i * n + j];
        const float err = std::abs(ref - got);
        max_abs_err = std::max(max_abs_err, err);
        if (err > tol) {
            msg = "Mismatch at (" + std::to_string(i) + "," + std::to_string(j) + "): ref=" +
                  std::to_string(ref) + ", got=" + std::to_string(got) + ", abs_err=" + std::to_string(err);
            return false;
        }
    }
    msg = "Spot checks passed, max_abs_err=" + std::to_string(max_abs_err);
    return true;
}

constexpr int OPCODE_BLOCK_X = 32;
constexpr int OPCODE_BLOCK_Y = 8;

__device__ __forceinline__ int32_t shl1_i32_literal(int32_t x) {
    return static_cast<int32_t>(static_cast<uint32_t>(x) << 1);
}

template <bool USE_SHIFT>
__global__ void quinary_opcode_kernel(const int* __restrict__ row_ptr_pos2,
                                      const int* __restrict__ row_ptr_pos1,
                                      const int* __restrict__ row_ptr_neg1,
                                      const int* __restrict__ row_ptr_neg2,
                                      const int* __restrict__ idx_pos2,
                                      const int* __restrict__ idx_pos1,
                                      const int* __restrict__ idx_neg1,
                                      const int* __restrict__ idx_neg2,
                                      const int8_t* __restrict__ b,
                                      int32_t* __restrict__ c2,
                                      int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;

    int32_t acc = 0;
    const int8_t* bcol = b + col;

    for (int p = row_ptr_pos2[row]; p < row_ptr_pos2[row + 1]; ++p) {
        const int32_t v = static_cast<int32_t>(bcol[idx_pos2[p] * n]);
        acc += USE_SHIFT ? shl1_i32_literal(v) : (v + v);
    }
    for (int p = row_ptr_pos1[row]; p < row_ptr_pos1[row + 1]; ++p) {
        acc += static_cast<int32_t>(bcol[idx_pos1[p] * n]);
    }
    for (int p = row_ptr_neg1[row]; p < row_ptr_neg1[row + 1]; ++p) {
        acc -= static_cast<int32_t>(bcol[idx_neg1[p] * n]);
    }
    for (int p = row_ptr_neg2[row]; p < row_ptr_neg2[row + 1]; ++p) {
        const int32_t v = static_cast<int32_t>(bcol[idx_neg2[p] * n]);
        acc -= USE_SHIFT ? shl1_i32_literal(v) : (v + v);
    }
    c2[row * n + col] = acc;
}

static void launch_opcode_cuda(const DeviceOpcodeCSR& d_csr,
                               const int8_t* d_b,
                               int32_t* d_c2,
                               int n,
                               bool use_shift) {
    const dim3 block(OPCODE_BLOCK_X, OPCODE_BLOCK_Y);
    const dim3 grid((n + OPCODE_BLOCK_X - 1) / OPCODE_BLOCK_X,
                    (n + OPCODE_BLOCK_Y - 1) / OPCODE_BLOCK_Y);
    if (use_shift) {
        quinary_opcode_kernel<true><<<grid, block>>>(d_csr.row_ptr_pos2, d_csr.row_ptr_pos1, d_csr.row_ptr_neg1,
                                                     d_csr.row_ptr_neg2, d_csr.idx_pos2, d_csr.idx_pos1,
                                                     d_csr.idx_neg1, d_csr.idx_neg2, d_b, d_c2, n);
    } else {
        quinary_opcode_kernel<false><<<grid, block>>>(d_csr.row_ptr_pos2, d_csr.row_ptr_pos1, d_csr.row_ptr_neg1,
                                                      d_csr.row_ptr_neg2, d_csr.idx_pos2, d_csr.idx_pos1,
                                                      d_csr.idx_neg1, d_csr.idx_neg2, d_b, d_c2, n);
    }
    CUDA_CHECK(cudaGetLastError());
}

constexpr int DENSE_TILE_M = 16;
constexpr int DENSE_TILE_N = 16;
constexpr int DENSE_TILE_K = 32;

__global__ void dense_int8_cuda_core_kernel(const int8_t* __restrict__ a,
                                            const int8_t* __restrict__ b,
                                            int32_t* __restrict__ c,
                                            int n) {
    __shared__ int8_t As[DENSE_TILE_M][DENSE_TILE_K];
    __shared__ int8_t Bs[DENSE_TILE_K][DENSE_TILE_N];

    const int row = blockIdx.y * DENSE_TILE_M + threadIdx.y;
    const int col = blockIdx.x * DENSE_TILE_N + threadIdx.x;
    int32_t acc = 0;

    for (int kb = 0; kb < n; kb += DENSE_TILE_K) {
        for (int kk_load = threadIdx.x; kk_load < DENSE_TILE_K; kk_load += DENSE_TILE_N) {
            As[threadIdx.y][kk_load] = (row < n && (kb + kk_load) < n) ? a[row * n + (kb + kk_load)] : static_cast<int8_t>(0);
        }
        for (int kk_load = threadIdx.y; kk_load < DENSE_TILE_K; kk_load += DENSE_TILE_M) {
            Bs[kk_load][threadIdx.x] = (col < n && (kb + kk_load) < n) ? b[(kb + kk_load) * n + col] : static_cast<int8_t>(0);
        }
        __syncthreads();
        if (row < n && col < n) {
            #pragma unroll
            for (int kk = 0; kk < DENSE_TILE_K; ++kk) {
                acc += static_cast<int32_t>(As[threadIdx.y][kk]) * static_cast<int32_t>(Bs[kk][threadIdx.x]);
            }
        }
        __syncthreads();
    }
    if (row < n && col < n) c[row * n + col] = acc;
}

static void dense_int8_cuda_core(const int8_t* d_a_code, const int8_t* d_b, int32_t* d_c2, int n) {
    const dim3 block(DENSE_TILE_N, DENSE_TILE_M);
    const dim3 grid((n + DENSE_TILE_N - 1) / DENSE_TILE_N, (n + DENSE_TILE_M - 1) / DENSE_TILE_M);
    dense_int8_cuda_core_kernel<<<grid, block>>>(d_a_code, d_b, d_c2, n);
    CUDA_CHECK(cudaGetLastError());
}

static void dense_int8_tensorcore(cublasHandle_t handle,
                                  const int8_t* d_a_code,
                                  const int8_t* d_b,
                                  int32_t* d_c2,
                                  int n) {
    const int32_t alpha = 1;
    const int32_t beta = 0;
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              n,
                              n,
                              n,
                              &alpha,
                              d_b,
                              CUDA_R_8I,
                              n,
                              d_a_code,
                              CUDA_R_8I,
                              n,
                              &beta,
                              d_c2,
                              CUDA_R_32I,
                              n,
                              CUBLAS_COMPUTE_32I,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

static void dense_fp16_tensorcore(cublasHandle_t handle,
                                  const __half* d_a,
                                  const __half* d_b,
                                  float* d_c,
                                  int n) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              n,
                              n,
                              n,
                              &alpha,
                              d_b,
                              CUDA_R_16F,
                              n,
                              d_a,
                              CUDA_R_16F,
                              n,
                              &beta,
                              d_c,
                              CUDA_R_32F,
                              n,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

static void dense_bf16_tensorcore(cublasHandle_t handle,
                                  const __nv_bfloat16* d_a,
                                  const __nv_bfloat16* d_b,
                                  float* d_c,
                                  int n) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              n,
                              n,
                              n,
                              &alpha,
                              d_b,
                              CUDA_R_16BF,
                              n,
                              d_a,
                              CUDA_R_16BF,
                              n,
                              &beta,
                              d_c,
                              CUDA_R_32F,
                              n,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

template <typename LaunchFn>
static BenchmarkResult benchmark_method(const DeviceInfo& dev,
                                        const std::string& method,
                                        int n,
                                        double zero_prob,
                                        double zero_ratio,
                                        int iters,
                                        int warmup,
                                        double sample_ms,
                                        uint32_t seed_a,
                                        uint32_t seed_b,
                                        int verify_performed,
                                        LaunchFn&& launch) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    NvmlEnergySampler sampler;
    sampler.init_from_current_cuda_device();

    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    const auto wall_t0 = std::chrono::steady_clock::now();
    if (sampler.ok()) sampler.start(sample_ms);
    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    const auto wall_t1 = std::chrono::steady_clock::now();
    const double energy_j = sampler.ok() ? sampler.stop_and_integrate_joules() : -1.0;

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    const double wall_ms = std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count();

    BenchmarkResult r;
    r.gpu_name = dev.gpu_name;
    r.cc_major = dev.cc_major;
    r.cc_minor = dev.cc_minor;
    r.method = method;
    r.n = n;
    r.zero_prob = zero_prob;
    r.zero_ratio = zero_ratio;
    r.iters = iters;
    r.warmup = warmup;
    r.sample_ms = sample_ms;
    r.seed_a = seed_a;
    r.seed_b = seed_b;
    r.verify_performed = verify_performed;
    r.gpu_ms_total = gpu_ms;
    r.gpu_ms_per_iter = gpu_ms / static_cast<double>(iters);
    r.wall_ms_total = wall_ms;
    r.wall_ms_per_iter = wall_ms / static_cast<double>(iters);
    r.throughput_inf_s = 1000.0 / r.wall_ms_per_iter;
    r.energy_j_total = energy_j;
    r.energy_j_per_iter = (energy_j >= 0.0 ? energy_j / static_cast<double>(iters) : -1.0);
    r.avg_power_w = (energy_j >= 0.0 && wall_ms > 0.0 ? energy_j / (wall_ms / 1000.0) : -1.0);
    return r;
}

static void print_result(const BenchmarkResult& r) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  [" << r.method << "]\n";
    std::cout << "    gpu_ms/iter   : " << r.gpu_ms_per_iter << "\n";
    std::cout << "    wall_ms/iter  : " << r.wall_ms_per_iter << "\n";
    std::cout << "    inf/s         : " << r.throughput_inf_s << "\n";
    if (r.avg_power_w >= 0.0) {
        std::cout << "    avg_power_w   : " << r.avg_power_w << "\n";
        std::cout << "    energy_j/iter : " << r.energy_j_per_iter << "\n";
    } else {
        std::cout << "    avg_power_w   : NA\n";
        std::cout << "    energy_j/iter : NA\n";
    }
}

static std::string csv_escape(const std::string& s) {
    if (s.find_first_of(",\"\n") == std::string::npos) return s;
    std::string out = "\"";
    for (char ch : s) {
        if (ch == '\"') out += "\"\"";
        else out += ch;
    }
    out += "\"";
    return out;
}

static void write_csv_rows(const std::string& path,
                           const std::vector<BenchmarkResult>& rows,
                           bool append) {
    if (path.empty() || rows.empty()) return;
    const bool exists = static_cast<bool>(std::ifstream(path).good());
    std::ofstream ofs(path, append ? std::ios::app : std::ios::out);
    if (!ofs) {
        std::cerr << "Failed to open CSV path: " << path << "\n";
        std::exit(1);
    }
    if (!append || !exists) {
        ofs << "gpu_name,cc_major,cc_minor,method,n,zero_prob,zero_ratio,iters,warmup,sample_ms,seed_a,seed_b,verify_performed,"
               "gpu_ms_total,gpu_ms_per_iter,wall_ms_total,wall_ms_per_iter,throughput_inf_s,avg_power_w,energy_j_total,energy_j_per_iter\n";
    }
    ofs << std::fixed << std::setprecision(8);
    for (const auto& r : rows) {
        ofs << csv_escape(r.gpu_name) << ','
            << r.cc_major << ',' << r.cc_minor << ','
            << csv_escape(r.method) << ','
            << r.n << ',' << r.zero_prob << ',' << r.zero_ratio << ','
            << r.iters << ',' << r.warmup << ',' << r.sample_ms << ','
            << r.seed_a << ',' << r.seed_b << ',' << r.verify_performed << ','
            << r.gpu_ms_total << ',' << r.gpu_ms_per_iter << ','
            << r.wall_ms_total << ',' << r.wall_ms_per_iter << ','
            << r.throughput_inf_s << ',';
        if (r.avg_power_w >= 0.0) ofs << r.avg_power_w; else ofs << "NA";
        ofs << ',';
        if (r.energy_j_total >= 0.0) ofs << r.energy_j_total; else ofs << "NA";
        ofs << ',';
        if (r.energy_j_per_iter >= 0.0) ofs << r.energy_j_per_iter; else ofs << "NA";
        ofs << '\n';
    }
}

static int run_one_point(const Options& opt,
                         const DeviceInfo& dev,
                         int n,
                         double zero_prob,
                         bool do_verify,
                         bool csv_append) {
    std::cout << "\n=== Benchmark point: N=" << n << ", zero_prob=" << zero_prob << " ===\n";

    std::vector<int8_t> a_code(static_cast<size_t>(n) * static_cast<size_t>(n));
    std::vector<int8_t> b_i8(static_cast<size_t>(n) * static_cast<size_t>(n));
    init_quinary_codes(a_code, zero_prob, opt.seed_a);
    init_activation_int8(b_i8, opt.seed_b);
    const OpcodeCSR csr = build_opcode_csr(a_code, n);
    const double zero_ratio = compute_zero_ratio(csr, n);
    std::cout << "  zero_ratio      : " << std::fixed << std::setprecision(4) << zero_ratio << "\n";

    int8_t* d_a_i8 = nullptr;
    int8_t* d_b_i8 = nullptr;
    int32_t* d_c_i32 = nullptr;
    int32_t* d_c_tmp = nullptr;
    float* d_c_fp = nullptr;
    __half* d_a_fp16 = nullptr;
    __half* d_b_fp16 = nullptr;
    __nv_bfloat16* d_a_bf16 = nullptr;
    __nv_bfloat16* d_b_bf16 = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a_i8), a_code.size() * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b_i8), b_i8.size() * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c_i32), a_code.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c_tmp), a_code.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_a_i8, a_code.data(), a_code.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_i8, b_i8.data(), b_i8.size() * sizeof(int8_t), cudaMemcpyHostToDevice));

    std::vector<__half> a_fp16_h;
    std::vector<__half> b_fp16_h;
    if (opt.run_fp16) {
        convert_to_fp16_weights(a_code, a_fp16_h);
        convert_to_fp16_activations(b_i8, b_fp16_h);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a_fp16), a_fp16_h.size() * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b_fp16), b_fp16_h.size() * sizeof(__half)));
        CUDA_CHECK(cudaMemcpy(d_a_fp16, a_fp16_h.data(), a_fp16_h.size() * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b_fp16, b_fp16_h.data(), b_fp16_h.size() * sizeof(__half), cudaMemcpyHostToDevice));
    }

    std::vector<__nv_bfloat16> a_bf16_h;
    std::vector<__nv_bfloat16> b_bf16_h;
    if (opt.run_bf16) {
        convert_to_bf16_weights(a_code, a_bf16_h);
        convert_to_bf16_activations(b_i8, b_bf16_h);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a_bf16), a_bf16_h.size() * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b_bf16), b_bf16_h.size() * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemcpy(d_a_bf16, a_bf16_h.data(), a_bf16_h.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b_bf16, b_bf16_h.data(), b_bf16_h.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    }

    if (opt.run_fp16 || opt.run_bf16) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c_fp), a_code.size() * sizeof(float)));
    }

    DeviceOpcodeCSR d_csr = upload_opcode_csr(csr);
    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));

    CUDA_CHECK(cudaFuncSetCacheConfig(quinary_opcode_kernel<false>, cudaFuncCachePreferL1));
    CUDA_CHECK(cudaFuncSetCacheConfig(quinary_opcode_kernel<true>, cudaFuncCachePreferL1));
    CUDA_CHECK(cudaFuncSetCacheConfig(dense_int8_cuda_core_kernel, cudaFuncCachePreferShared));

    if (do_verify) {
        std::cout << "[verify] correctness spot-checks\n";
        std::vector<int32_t> c_host(a_code.size());
        std::vector<float> c_fp_host(a_code.size());
        std::string msg;

        if (opt.run_int8_tc) {
            dense_int8_tensorcore(handle, d_a_i8, d_b_i8, d_c_i32, n);
            CUDA_CHECK(cudaMemcpy(c_host.data(), d_c_i32, c_host.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
            const bool ok = spot_check_exact_i32(a_code, b_i8, c_host, n, 128, 777, msg);
            std::cout << "  INT8 Tensor Core     : " << (ok ? "PASS" : "FAIL") << " | " << msg << "\n";
            if (!ok) return 1;
        }
        if (opt.run_dense_cuda_int8) {
            dense_int8_cuda_core(d_a_i8, d_b_i8, d_c_tmp, n);
            CUDA_CHECK(cudaMemcpy(c_host.data(), d_c_tmp, c_host.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
            const bool ok = spot_check_exact_i32(a_code, b_i8, c_host, n, 128, 778, msg);
            std::cout << "  INT8 CUDA Core       : " << (ok ? "PASS" : "FAIL") << " | " << msg << "\n";
            if (!ok) return 1;
        }
        if (opt.run_opcode_adddbl) {
            launch_opcode_cuda(d_csr, d_b_i8, d_c_tmp, n, false);
            CUDA_CHECK(cudaMemcpy(c_host.data(), d_c_tmp, c_host.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
            const bool ok = spot_check_exact_i32(a_code, b_i8, c_host, n, 128, 779, msg);
            std::cout << "  Opcode adddbl        : " << (ok ? "PASS" : "FAIL") << " | " << msg << "\n";
            if (!ok) return 1;
        }
        if (opt.run_opcode_shift) {
            launch_opcode_cuda(d_csr, d_b_i8, d_c_tmp, n, true);
            CUDA_CHECK(cudaMemcpy(c_host.data(), d_c_tmp, c_host.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
            const bool ok = spot_check_exact_i32(a_code, b_i8, c_host, n, 128, 780, msg);
            std::cout << "  Opcode shift         : " << (ok ? "PASS" : "FAIL") << " | " << msg << "\n";
            if (!ok) return 1;
        }
        if (opt.run_fp16) {
            dense_fp16_tensorcore(handle, d_a_fp16, d_b_fp16, d_c_fp, n);
            CUDA_CHECK(cudaMemcpy(c_fp_host.data(), d_c_fp, c_fp_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
            const bool ok = spot_check_float(a_code, b_i8, c_fp_host, n, 64, 781, 32.0f, msg);
            std::cout << "  FP16 Tensor Core     : " << (ok ? "PASS" : "FAIL") << " | " << msg << "\n";
            if (!ok) return 1;
        }
        if (opt.run_bf16) {
            dense_bf16_tensorcore(handle, d_a_bf16, d_b_bf16, d_c_fp, n);
            CUDA_CHECK(cudaMemcpy(c_fp_host.data(), d_c_fp, c_fp_host.size() * sizeof(float), cudaMemcpyDeviceToHost));
            const bool ok = spot_check_float(a_code, b_i8, c_fp_host, n, 64, 782, 96.0f, msg);
            std::cout << "  BF16 Tensor Core     : " << (ok ? "PASS" : "FAIL") << " | " << msg << "\n";
            if (!ok) return 1;
        }
    }

    std::vector<BenchmarkResult> results;
    if (opt.run_fp16) {
        results.push_back(benchmark_method(dev, "Dense FP16 Tensor Core", n, zero_prob, zero_ratio,
                                           opt.iters, opt.warmup, opt.sample_ms, opt.seed_a, opt.seed_b,
                                           do_verify ? 1 : 0, [&]() {
                                               dense_fp16_tensorcore(handle, d_a_fp16, d_b_fp16, d_c_fp, n);
                                           }));
    }
    if (opt.run_bf16) {
        results.push_back(benchmark_method(dev, "Dense BF16 Tensor Core", n, zero_prob, zero_ratio,
                                           opt.iters, opt.warmup, opt.sample_ms, opt.seed_a, opt.seed_b,
                                           do_verify ? 1 : 0, [&]() {
                                               dense_bf16_tensorcore(handle, d_a_bf16, d_b_bf16, d_c_fp, n);
                                           }));
    }
    if (opt.run_int8_tc) {
        results.push_back(benchmark_method(dev, "Dense INT8 Tensor Core", n, zero_prob, zero_ratio,
                                           opt.iters, opt.warmup, opt.sample_ms, opt.seed_a, opt.seed_b,
                                           do_verify ? 1 : 0, [&]() {
                                               dense_int8_tensorcore(handle, d_a_i8, d_b_i8, d_c_i32, n);
                                           }));
    }
    if (opt.run_dense_cuda_int8) {
        results.push_back(benchmark_method(dev, "Dense INT8 CUDA Core", n, zero_prob, zero_ratio,
                                           opt.iters, opt.warmup, opt.sample_ms, opt.seed_a, opt.seed_b,
                                           do_verify ? 1 : 0, [&]() {
                                               dense_int8_cuda_core(d_a_i8, d_b_i8, d_c_tmp, n);
                                           }));
    }
    if (opt.run_opcode_adddbl) {
        results.push_back(benchmark_method(dev, "Row-Opcode CUDA SM80 (adddbl)", n, zero_prob, zero_ratio,
                                           opt.iters, opt.warmup, opt.sample_ms, opt.seed_a, opt.seed_b,
                                           do_verify ? 1 : 0, [&]() {
                                               launch_opcode_cuda(d_csr, d_b_i8, d_c_tmp, n, false);
                                           }));
    }
    if (opt.run_opcode_shift) {
        results.push_back(benchmark_method(dev, "Row-Opcode CUDA SM80 (shift)", n, zero_prob, zero_ratio,
                                           opt.iters, opt.warmup, opt.sample_ms, opt.seed_a, opt.seed_b,
                                           do_verify ? 1 : 0, [&]() {
                                               launch_opcode_cuda(d_csr, d_b_i8, d_c_tmp, n, true);
                                           }));
    }

    for (const auto& r : results) print_result(r);
    write_csv_rows(opt.csv_path, results, csv_append);

    CUBLAS_CHECK(cublasDestroy(handle));
    free_device_opcode_csr(d_csr);
    if (d_a_fp16) CUDA_CHECK(cudaFree(d_a_fp16));
    if (d_b_fp16) CUDA_CHECK(cudaFree(d_b_fp16));
    if (d_a_bf16) CUDA_CHECK(cudaFree(d_a_bf16));
    if (d_b_bf16) CUDA_CHECK(cudaFree(d_b_bf16));
    if (d_c_fp) CUDA_CHECK(cudaFree(d_c_fp));
    if (d_a_i8) CUDA_CHECK(cudaFree(d_a_i8));
    if (d_b_i8) CUDA_CHECK(cudaFree(d_b_i8));
    if (d_c_i32) CUDA_CHECK(cudaFree(d_c_i32));
    if (d_c_tmp) CUDA_CHECK(cudaFree(d_c_tmp));
    return 0;
}

}  // namespace bench

int main(int argc, char** argv) {
    using namespace bench;
    const Options opt = parse_options(argc, argv);
    const DeviceInfo dev = get_device_info();
    print_device_info(dev);

    bool first_point = true;
    bool append_mode = opt.append_csv;
    for (int n : opt.sweep_n) {
        for (double zero_prob : opt.sweep_zero_prob) {
            const bool do_verify = opt.verify_each || first_point;
            const int rc = run_one_point(opt, dev, n, zero_prob, do_verify, append_mode);
            if (rc != 0) return rc;
            first_point = false;
            append_mode = true;
        }
    }

    if (!opt.csv_path.empty()) {
        std::cout << "\nCSV written to: " << opt.csv_path << "\n";
    }
    return 0;
}
