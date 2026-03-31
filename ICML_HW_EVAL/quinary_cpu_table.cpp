#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

namespace bench {

constexpr int SIMD_WIDTH = 8;  // AVX2 int32 lanes
constexpr int DEFAULT_N = 1024;
constexpr int DEFAULT_TILE_J = 64;
constexpr int DEFAULT_ITERS = 50;
constexpr int DEFAULT_WARMUP = 5;
constexpr uint32_t DEFAULT_SEED_A = 1234;
constexpr uint32_t DEFAULT_SEED_B = 4321;
constexpr double DEFAULT_ZERO_PROB = 0.40;

struct Config {
    int n = DEFAULT_N;
    int tile_j = DEFAULT_TILE_J;
    int iters = DEFAULT_ITERS;
    int warmup = DEFAULT_WARMUP;
    int threads = 0;            // 0 => keep current OMP setting
    int verify_n = 128;
    uint32_t seed_a = DEFAULT_SEED_A;
    uint32_t seed_b = DEFAULT_SEED_B;
    double zero_prob = DEFAULT_ZERO_PROB;
    bool run_ref = true;
    bool run_dense = true;
    bool run_opcode = true;
    bool quiet = false;
    string csv_path;
};

struct RowOps {
    vector<int> idx_pos2; // code =  2 -> +2x
    vector<int> idx_pos1; // code =  1 -> +x
    vector<int> idx_neg1; // code = -1 -> -x
    vector<int> idx_neg2; // code = -2 -> -2x
};

struct OpcodeMatrix {
    vector<RowOps> rows;
};

struct OpcodeStats {
    size_t pos2 = 0;
    size_t pos1 = 0;
    size_t neg1 = 0;
    size_t neg2 = 0;
    size_t nonzero = 0;
    size_t zero = 0;
    double zero_ratio = 0.0;
};

struct BenchResult {
    string method;
    double avg_ms = numeric_limits<double>::quiet_NaN();
    double avg_mj = numeric_limits<double>::quiet_NaN();
    double gops = numeric_limits<double>::quiet_NaN();
    double speedup_vs_dense = numeric_limits<double>::quiet_NaN();
};

struct CpuInfo {
    string model_name = "unknown";
};

string trim_copy(string s) {
    const auto is_not_space = [](unsigned char ch) { return !std::isspace(ch); };
    s.erase(s.begin(), find_if(s.begin(), s.end(), is_not_space));
    s.erase(find_if(s.rbegin(), s.rend(), is_not_space).base(), s.end());
    return s;
}

string read_first_line(const string& path) {
    ifstream file(path);
    string line;
    if (file.is_open() && getline(file, line)) {
        return trim_copy(line);
    }
    return "";
}

string find_powercap_energy_uj_path() {
    namespace fs = std::filesystem;
    const vector<string> preferred = {
        "/sys/class/powercap/intel-rapl:0/energy_uj",
        "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
        "/sys/class/powercap/amd-rapl:0/energy_uj",
        "/sys/class/powercap/amd-rapl/amd-rapl:0/energy_uj",
    };
    for (const auto& p : preferred) {
        if (fs::exists(p)) return p;
    }

    const fs::path root("/sys/class/powercap");
    if (!fs::exists(root) || !fs::is_directory(root)) {
        return "";
    }

    vector<fs::path> stack = {root};
    while (!stack.empty()) {
        fs::path dir = stack.back();
        stack.pop_back();
        for (const auto& entry : fs::directory_iterator(dir)) {
            std::error_code ec;
            if (entry.is_directory(ec)) {
                stack.push_back(entry.path());
                continue;
            }
            if (!entry.is_regular_file(ec)) continue;
            if (entry.path().filename() == "energy_uj") {
                return entry.path().string();
            }
        }
    }
    return "";
}

const string& energy_uj_path() {
    static const string path = find_powercap_energy_uj_path();
    return path;
}

long long get_energy_uj() {
    const string& path = energy_uj_path();
    if (path.empty()) {
        return -1;
    }
    const string line = read_first_line(path);
    if (line.empty()) {
        return -1;
    }
    try {
        return stoll(line);
    } catch (...) {
        return -1;
    }
}

template <typename T>
T* aligned_alloc_t(size_t count, size_t alignment = 32) {
    size_t bytes = count * sizeof(T);
    size_t padded = ((bytes + alignment - 1) / alignment) * alignment;
#if defined(_MSC_VER)
    return static_cast<T*>(_aligned_malloc(padded, alignment));
#else
    return static_cast<T*>(std::aligned_alloc(alignment, padded));
#endif
}

template <typename T>
void aligned_free_t(T* ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

inline __m256i load8_i8_to_i32(const int8_t* ptr) {
    __m128i v8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr));
    return _mm256_cvtepi8_epi32(v8);
}

CpuInfo get_cpu_info() {
    CpuInfo info;
    ifstream file("/proc/cpuinfo");
    string line;
    while (getline(file, line)) {
        const string key = "model name\t: ";
        if (line.rfind(key, 0) == 0) {
            info.model_name = line.substr(key.size());
            break;
        }
    }
    return info;
}

void init_quinary_codes(int8_t* a_code, size_t count, double zero_prob, uint32_t seed) {
    if (zero_prob < 0.0 || zero_prob > 1.0) {
        throw invalid_argument("zero_prob must be in [0, 1]");
    }
    mt19937 rng(seed);
    const int8_t vals[5] = {-2, -1, 0, 1, 2};
    const double nz = (1.0 - zero_prob) / 4.0;
    const double probs[5] = {nz, nz, zero_prob, nz, nz};
    discrete_distribution<int> dist(probs, probs + 5);
    for (size_t i = 0; i < count; ++i) {
        a_code[i] = vals[dist(rng)];
    }
}

void init_activation_int8(int8_t* b, size_t count, uint32_t seed) {
    mt19937 rng(seed);
    uniform_int_distribution<int> dist(-127, 127);
    for (size_t i = 0; i < count; ++i) {
        b[i] = static_cast<int8_t>(dist(rng));
    }
}

OpcodeMatrix build_opcode_matrix(const int8_t* a_code, int n) {
    OpcodeMatrix mat;
    mat.rows.resize(n);
    for (int i = 0; i < n; ++i) {
        RowOps& row = mat.rows[i];
        row.idx_pos2.reserve(n / 5);
        row.idx_pos1.reserve(n / 5);
        row.idx_neg1.reserve(n / 5);
        row.idx_neg2.reserve(n / 5);

        const int base = i * n;
        for (int k = 0; k < n; ++k) {
            const int8_t code = a_code[base + k];
            switch (code) {
                case 2:  row.idx_pos2.push_back(k); break;
                case 1:  row.idx_pos1.push_back(k); break;
                case -1: row.idx_neg1.push_back(k); break;
                case -2: row.idx_neg2.push_back(k); break;
                default: break;
            }
        }
    }
    return mat;
}

OpcodeStats compute_opcode_stats(const OpcodeMatrix& opmat, int n) {
    OpcodeStats s;
    for (const auto& r : opmat.rows) {
        s.pos2 += r.idx_pos2.size();
        s.pos1 += r.idx_pos1.size();
        s.neg1 += r.idx_neg1.size();
        s.neg2 += r.idx_neg2.size();
    }
    const size_t total = static_cast<size_t>(n) * static_cast<size_t>(n);
    s.nonzero = s.pos2 + s.pos1 + s.neg1 + s.neg2;
    s.zero = total - s.nonzero;
    s.zero_ratio = total ? static_cast<double>(s.zero) / static_cast<double>(total) : 0.0;
    return s;
}

void print_opcode_stats(const OpcodeStats& s) {
    cout << "Opcode stats:\n";
    cout << "  +2 count : " << s.pos2 << "\n";
    cout << "  +1 count : " << s.pos1 << "\n";
    cout << "  -1 count : " << s.neg1 << "\n";
    cout << "  -2 count : " << s.neg2 << "\n";
    cout << "  zero     : " << s.zero << "\n";
    cout << "  sparsity : " << fixed << setprecision(2)
         << (100.0 * s.zero_ratio) << "%\n";
}

void gemm_quinary_mul_ref(const int8_t* a_code, const int8_t* b, int32_t* c2, int n) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int32_t acc2 = 0;
            for (int k = 0; k < n; ++k) {
                const int32_t code = static_cast<int32_t>(a_code[i * n + k]);
                const int32_t x = static_cast<int32_t>(b[k * n + j]);
                acc2 += code * x;
            }
            c2[i * n + j] = acc2;
        }
    }
}

void gemm_quinary_opcode_ref(const OpcodeMatrix& opmat, const int8_t* b, int32_t* c2, int n) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int32_t acc2 = 0;
            const RowOps& row = opmat.rows[i];
            for (int k : row.idx_pos2) acc2 += 2 * static_cast<int32_t>(b[k * n + j]);
            for (int k : row.idx_pos1) acc2 += static_cast<int32_t>(b[k * n + j]);
            for (int k : row.idx_neg1) acc2 -= static_cast<int32_t>(b[k * n + j]);
            for (int k : row.idx_neg2) acc2 -= 2 * static_cast<int32_t>(b[k * n + j]);
            c2[i * n + j] = acc2;
        }
    }
}

void gemm_quinary_mul_avx2(const int8_t* a_code, const int8_t* b, int32_t* c2, int n, int tile_j) {
#pragma omp parallel for
    for (int ii = 0; ii < n; ++ii) {
        for (int j = 0; j < n; j += tile_j) {
            const int j_end = min(j + tile_j, n);
            int jj = j;
            for (; jj + SIMD_WIDTH <= j_end; jj += SIMD_WIDTH) {
                __m256i vacc = _mm256_setzero_si256();
                for (int k = 0; k < n; ++k) {
                    const int32_t code = static_cast<int32_t>(a_code[ii * n + k]);
                    const __m256i vcode = _mm256_set1_epi32(code);
                    const __m256i vx = load8_i8_to_i32(&b[k * n + jj]);
                    vacc = _mm256_add_epi32(vacc, _mm256_mullo_epi32(vcode, vx));
                }
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&c2[ii * n + jj]), vacc);
            }
            for (; jj < j_end; ++jj) {
                int32_t acc2 = 0;
                for (int k = 0; k < n; ++k) {
                    acc2 += static_cast<int32_t>(a_code[ii * n + k]) * static_cast<int32_t>(b[k * n + jj]);
                }
                c2[ii * n + jj] = acc2;
            }
        }
    }
}

void gemm_quinary_opcode_avx2(const OpcodeMatrix& opmat, const int8_t* b, int32_t* c2, int n, int tile_j) {
#pragma omp parallel for
    for (int ii = 0; ii < n; ++ii) {
        const RowOps& row = opmat.rows[ii];
        for (int j = 0; j < n; j += tile_j) {
            const int j_end = min(j + tile_j, n);
            int jj = j;
            for (; jj + SIMD_WIDTH <= j_end; jj += SIMD_WIDTH) {
                __m256i vacc = _mm256_setzero_si256();

                for (int k : row.idx_pos2) {
                    const __m256i vx = load8_i8_to_i32(&b[k * n + jj]);
                    const __m256i v2x = _mm256_add_epi32(vx, vx);
                    vacc = _mm256_add_epi32(vacc, v2x);
                }
                for (int k : row.idx_pos1) {
                    const __m256i vx = load8_i8_to_i32(&b[k * n + jj]);
                    vacc = _mm256_add_epi32(vacc, vx);
                }
                for (int k : row.idx_neg1) {
                    const __m256i vx = load8_i8_to_i32(&b[k * n + jj]);
                    vacc = _mm256_sub_epi32(vacc, vx);
                }
                for (int k : row.idx_neg2) {
                    const __m256i vx = load8_i8_to_i32(&b[k * n + jj]);
                    const __m256i v2x = _mm256_add_epi32(vx, vx);
                    vacc = _mm256_sub_epi32(vacc, v2x);
                }

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&c2[ii * n + jj]), vacc);
            }
            for (; jj < j_end; ++jj) {
                int32_t acc2 = 0;
                for (int k : row.idx_pos2) acc2 += 2 * static_cast<int32_t>(b[k * n + jj]);
                for (int k : row.idx_pos1) acc2 += static_cast<int32_t>(b[k * n + jj]);
                for (int k : row.idx_neg1) acc2 -= static_cast<int32_t>(b[k * n + jj]);
                for (int k : row.idx_neg2) acc2 -= 2 * static_cast<int32_t>(b[k * n + jj]);
                c2[ii * n + jj] = acc2;
            }
        }
    }
}

bool compare_exact(const int32_t* a, const int32_t* b, size_t count, string& msg) {
    for (size_t i = 0; i < count; ++i) {
        if (a[i] != b[i]) {
            msg = "Mismatch at index " + to_string(i) + ": lhs=" + to_string(a[i]) + ", rhs=" + to_string(b[i]);
            return false;
        }
    }
    msg = "Exact match";
    return true;
}

double estimate_dense_int32_ops(int n) {
    return 2.0 * static_cast<double>(n) * n * n;
}

template <typename Fn>
BenchResult run_bench(const string& name, Fn fn, int32_t* out, int iters, int n) {
    for (int i = 0; i < iters; ++i) {
        fn(out);
    }
    long long e_start = get_energy_uj();
    auto t_start = chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        fn(out);
    }
    auto t_end = chrono::high_resolution_clock::now();
    long long e_end = get_energy_uj();

    const double total_ms = chrono::duration<double, milli>(t_end - t_start).count();
    BenchResult r;
    r.method = name;
    r.avg_ms = total_ms / static_cast<double>(iters);
    if (e_start >= 0 && e_end >= 0 && e_end >= e_start) {
        r.avg_mj = static_cast<double>(e_end - e_start) / 1000.0 / static_cast<double>(iters);
    }
    const double ops = estimate_dense_int32_ops(n);
    r.gops = ops / (r.avg_ms * 1.0e6);
    return r;
}

void append_csv_header_if_needed(const string& path) {
    if (path.empty()) return;
    ifstream in(path);
    if (in.good()) return;
    ofstream out(path, ios::out);
    out << "cpu_model,energy_path,n,zero_prob,zero_ratio,threads,iters,warmup,method,avg_ms,avg_mj,gops,speedup_vs_dense,pos2,pos1,neg1,neg2,nonzero,zero\n";
}

void append_csv_row(const string& path, const CpuInfo& cpu, const Config& cfg, const OpcodeStats& s, const BenchResult& r) {
    if (path.empty()) return;
    ofstream out(path, ios::app);
    out << '"' << cpu.model_name << '"' << ','
        << '"' << energy_uj_path() << '"' << ','
        << cfg.n << ','
        << cfg.zero_prob << ','
        << s.zero_ratio << ','
        << omp_get_max_threads() << ','
        << cfg.iters << ','
        << cfg.warmup << ','
        << r.method << ','
        << r.avg_ms << ',';
    if (std::isnan(r.avg_mj)) out << "NA"; else out << r.avg_mj;
    out << ',' << r.gops << ',';
    if (std::isnan(r.speedup_vs_dense)) out << "NA"; else out << r.speedup_vs_dense;
    out << ',' << s.pos2 << ',' << s.pos1 << ',' << s.neg1 << ',' << s.neg2 << ','
        << s.nonzero << ',' << s.zero << '\n';
}

void print_usage(const char* argv0) {
    cout << "Usage: " << argv0 << " [options]\n"
         << "  --n <int>             Matrix size N (default 1024)\n"
         << "  --iters <int>         Benchmark iterations (default 50)\n"
         << "  --warmup <int>        Warmup iterations (default 5)\n"
         << "  --threads <int>       OMP thread count\n"
         << "  --tile-j <int>        J-tile size (default 64)\n"
         << "  --zero-prob <float>   Probability of zero code in A (default 0.40)\n"
         << "  --seed-a <int>        Seed for A codes\n"
         << "  --seed-b <int>        Seed for B activations\n"
         << "  --verify-n <int>      Correctness size (default 128, <=0 disables)\n"
         << "  --csv <path>          Append results to CSV\n"
         << "  --no-ref              Skip scalar reference check\n"
         << "  --no-dense            Skip dense AVX2 benchmark\n"
         << "  --no-opcode           Skip opcode AVX2 benchmark\n"
         << "  --quiet               Reduce console output\n"
         << "  --help                Show this message\n";
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const string arg = argv[i];
        auto need_value = [&](const string& name) -> string {
            if (i + 1 >= argc) throw invalid_argument("Missing value for " + name);
            return argv[++i];
        };
        if (arg == "--n") cfg.n = stoi(need_value(arg));
        else if (arg == "--iters") cfg.iters = stoi(need_value(arg));
        else if (arg == "--warmup") cfg.warmup = stoi(need_value(arg));
        else if (arg == "--threads") cfg.threads = stoi(need_value(arg));
        else if (arg == "--tile-j") cfg.tile_j = stoi(need_value(arg));
        else if (arg == "--zero-prob") cfg.zero_prob = stod(need_value(arg));
        else if (arg == "--seed-a") cfg.seed_a = static_cast<uint32_t>(stoul(need_value(arg)));
        else if (arg == "--seed-b") cfg.seed_b = static_cast<uint32_t>(stoul(need_value(arg)));
        else if (arg == "--verify-n") cfg.verify_n = stoi(need_value(arg));
        else if (arg == "--csv") cfg.csv_path = need_value(arg);
        else if (arg == "--no-ref") cfg.run_ref = false;
        else if (arg == "--no-dense") cfg.run_dense = false;
        else if (arg == "--no-opcode") cfg.run_opcode = false;
        else if (arg == "--quiet") cfg.quiet = true;
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw invalid_argument("Unknown option: " + arg);
        }
    }
    if (cfg.n <= 0 || cfg.iters <= 0 || cfg.warmup < 0 || cfg.tile_j <= 0) {
        throw invalid_argument("n, iters, and tile-j must be > 0; warmup must be >= 0");
    }
    return cfg;
}

bool run_correctness_check(const Config& cfg) {
    if (!cfg.run_ref || cfg.verify_n <= 0) return true;
    const int n = min(cfg.verify_n, cfg.n);
    const size_t elems = static_cast<size_t>(n) * static_cast<size_t>(n);

    int8_t* a_code = aligned_alloc_t<int8_t>(elems, 32);
    int8_t* b = aligned_alloc_t<int8_t>(elems, 32);
    int32_t* c_ref_mul = aligned_alloc_t<int32_t>(elems, 32);
    int32_t* c_ref_op = aligned_alloc_t<int32_t>(elems, 32);
    int32_t* c_avx_mul = aligned_alloc_t<int32_t>(elems, 32);
    int32_t* c_avx_op = aligned_alloc_t<int32_t>(elems, 32);

    if (!a_code || !b || !c_ref_mul || !c_ref_op || !c_avx_mul || !c_avx_op) {
        cerr << "Correctness allocation failed\n";
        return false;
    }

    init_quinary_codes(a_code, elems, cfg.zero_prob, cfg.seed_a);
    init_activation_int8(b, elems, cfg.seed_b);
    OpcodeMatrix opmat = build_opcode_matrix(a_code, n);

    gemm_quinary_mul_ref(a_code, b, c_ref_mul, n);
    gemm_quinary_opcode_ref(opmat, b, c_ref_op, n);
    gemm_quinary_mul_avx2(a_code, b, c_avx_mul, n, cfg.tile_j);
    gemm_quinary_opcode_avx2(opmat, b, c_avx_op, n, cfg.tile_j);

    string msg;
    bool ok = compare_exact(c_ref_mul, c_ref_op, elems, msg);
    cout << "REF MUL vs OPCODE: " << (ok ? "PASS" : "FAIL") << " | " << msg << "\n";
    if (ok) {
        ok = compare_exact(c_ref_mul, c_avx_mul, elems, msg);
        cout << "AVX2 MUL    vs REF: " << (ok ? "PASS" : "FAIL") << " | " << msg << "\n";
    }
    if (ok) {
        ok = compare_exact(c_ref_op, c_avx_op, elems, msg);
        cout << "AVX2 OPCODE vs REF: " << (ok ? "PASS" : "FAIL") << " | " << msg << "\n";
    }
    if (ok) {
        ok = compare_exact(c_avx_mul, c_avx_op, elems, msg);
        cout << "AVX2 MUL vs OPCODE: " << (ok ? "PASS" : "FAIL") << " | " << msg << "\n";
    }

    aligned_free_t(a_code);
    aligned_free_t(b);
    aligned_free_t(c_ref_mul);
    aligned_free_t(c_ref_op);
    aligned_free_t(c_avx_mul);
    aligned_free_t(c_avx_op);
    return ok;
}

int run(const Config& cfg) {
    if (cfg.threads > 0) omp_set_num_threads(cfg.threads);

    if (!cfg.quiet) {
        cout << "=== Quinary CPU Benchmark (AVX2 Dense vs Opcode) ===\n";
        cout << "N          : " << cfg.n << "\n";
        cout << "ITER       : " << cfg.iters << "\n";
        cout << "WARMUP     : " << cfg.warmup << "\n";
        cout << "Threads    : " << omp_get_max_threads() << "\n";
        cout << "zero_prob  : " << cfg.zero_prob << "\n";
        cout << "tile_j     : " << cfg.tile_j << "\n";
        cout << "verify_n   : " << cfg.verify_n << "\n\n";
    }

    if (!run_correctness_check(cfg)) {
        return 1;
    }
    if (!cfg.quiet && cfg.run_ref && cfg.verify_n > 0) cout << '\n';

    const size_t elems = static_cast<size_t>(cfg.n) * static_cast<size_t>(cfg.n);
    int8_t* a_code = aligned_alloc_t<int8_t>(elems, 32);
    int8_t* b = aligned_alloc_t<int8_t>(elems, 32);
    int32_t* c_dense = aligned_alloc_t<int32_t>(elems, 32);
    int32_t* c_opcode = aligned_alloc_t<int32_t>(elems, 32);
    if (!a_code || !b || !c_dense || !c_opcode) {
        cerr << "Allocation failed\n";
        return 1;
    }

    init_quinary_codes(a_code, elems, cfg.zero_prob, cfg.seed_a);
    init_activation_int8(b, elems, cfg.seed_b);
    OpcodeMatrix opmat = build_opcode_matrix(a_code, cfg.n);
    OpcodeStats stats = compute_opcode_stats(opmat, cfg.n);
    CpuInfo cpu = get_cpu_info();

    if (!cfg.quiet) {
        cout << "CPU        : " << cpu.model_name << "\n";
        const string& epath = energy_uj_path();
        cout << "Energy path: " << (epath.empty() ? string("not found") : epath) << "\n";
        print_opcode_stats(stats);
        cout << '\n';
    }

    vector<BenchResult> results;
    if (cfg.run_dense) {
        BenchResult dense = run_bench("Dense MUL AVX2", [&](int32_t* out) {
            gemm_quinary_mul_avx2(a_code, b, out, cfg.n, cfg.tile_j);
        }, c_dense, cfg.iters, cfg.n);
        results.push_back(dense);
    }
    if (cfg.run_opcode) {
        BenchResult opcode = run_bench("Row-Opcode AVX2", [&](int32_t* out) {
            gemm_quinary_opcode_avx2(opmat, b, out, cfg.n, cfg.tile_j);
        }, c_opcode, cfg.iters, cfg.n);
        results.push_back(opcode);
    }

    double dense_ms = numeric_limits<double>::quiet_NaN();
    for (const auto& r : results) {
        if (r.method == "Dense MUL AVX2") {
            dense_ms = r.avg_ms;
            break;
        }
    }
    for (auto& r : results) {
        if (!std::isnan(dense_ms)) r.speedup_vs_dense = dense_ms / r.avg_ms;
    }

    if (!cfg.quiet) {
        cout << left << setw(20) << "Method"
             << right << setw(12) << "avg_ms"
             << setw(12) << "avg_mJ"
             << setw(12) << "GOPS"
             << setw(16) << "speedup_vs_dense"
             << "\n";
        cout << string(72, '-') << "\n";
        for (const auto& r : results) {
            cout << left << setw(20) << r.method
                 << right << setw(12) << fixed << setprecision(3) << r.avg_ms;
            if (std::isnan(r.avg_mj)) cout << setw(12) << "NA";
            else cout << setw(12) << fixed << setprecision(3) << r.avg_mj;
            cout << setw(12) << fixed << setprecision(2) << r.gops
                 << setw(16) << fixed << setprecision(3) << r.speedup_vs_dense
                 << "\n";
        }
        cout << "\n";
    }

    append_csv_header_if_needed(cfg.csv_path);
    for (const auto& r : results) {
        append_csv_row(cfg.csv_path, cpu, cfg, stats, r);
    }

    aligned_free_t(a_code);
    aligned_free_t(b);
    aligned_free_t(c_dense);
    aligned_free_t(c_opcode);
    return 0;
}

} // namespace bench

int main(int argc, char** argv) {
    try {
        const bench::Config cfg = bench::parse_args(argc, argv);
        return bench::run(cfg);
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n\n";
        bench::print_usage(argv[0]);
        return 1;
    }
}
