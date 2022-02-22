#pragma once
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#include <CL/opencl.h>

#include <bit>
#include <cassert>
#include <climits>
#include <iterator>
#include <limits>
#include <span>
#include <unordered_map>
#include <vector>

namespace Detail {
enum class ArithmeticCategory {
    SignedIntegral,
    UnsignedIntegral,
    FloatingPoint,
    Invalid,
};

template<typename T>
constexpr auto ArithmeticCategoryV = [] {
    using enum ArithmeticCategory;
    if constexpr (std::is_integral_v<T>) {
        if constexpr (std::is_signed_v<T>) {
            return SignedIntegral;
        }
        return UnsignedIntegral;
    }
    if constexpr (std::is_floating_point_v<T> and std::numeric_limits<T>::is_iec559) {
        return FloatingPoint;
    }
    return Invalid;
}();

template<size_t BitWidth, ArithmeticCategory AC> struct CLTypeRepr     { using type = void; };
template<> struct CLTypeRepr<8 , ArithmeticCategory::SignedIntegral>   { using type = cl_char; };
template<> struct CLTypeRepr<8 , ArithmeticCategory::UnsignedIntegral> { using type = cl_uchar; };
template<> struct CLTypeRepr<16, ArithmeticCategory::SignedIntegral>   { using type = cl_short; };
template<> struct CLTypeRepr<16, ArithmeticCategory::UnsignedIntegral> { using type = cl_ushort; };
template<> struct CLTypeRepr<32, ArithmeticCategory::SignedIntegral>   { using type = cl_int; };
template<> struct CLTypeRepr<32, ArithmeticCategory::UnsignedIntegral> { using type = cl_uint; };
template<> struct CLTypeRepr<64, ArithmeticCategory::SignedIntegral>   { using type = cl_long; };
template<> struct CLTypeRepr<64, ArithmeticCategory::UnsignedIntegral> { using type = cl_ulong; };
template<> struct CLTypeRepr<16, ArithmeticCategory::FloatingPoint>    { using type = cl_half; };
template<> struct CLTypeRepr<32, ArithmeticCategory::FloatingPoint>    { using type = cl_float; };
template<> struct CLTypeRepr<64, ArithmeticCategory::FloatingPoint>    { using type = cl_double; };

using CLChar   = CLTypeRepr<8 , ArithmeticCategory::SignedIntegral>;
using CLUChar  = CLTypeRepr<8 , ArithmeticCategory::UnsignedIntegral>;
using CLShort  = CLTypeRepr<16, ArithmeticCategory::SignedIntegral>;
using CLUShort = CLTypeRepr<16, ArithmeticCategory::UnsignedIntegral>;
using CLInt    = CLTypeRepr<32, ArithmeticCategory::SignedIntegral>;
using CLUInt   = CLTypeRepr<32, ArithmeticCategory::UnsignedIntegral>;
using CLLong   = CLTypeRepr<64, ArithmeticCategory::SignedIntegral>;
using CLULong  = CLTypeRepr<64, ArithmeticCategory::UnsignedIntegral>;
using CLHalf   = CLTypeRepr<16, ArithmeticCategory::FloatingPoint>;
using CLFloat  = CLTypeRepr<32, ArithmeticCategory::FloatingPoint>;
using CLDouble = CLTypeRepr<64, ArithmeticCategory::FloatingPoint>;

template<typename T>
using CLType = CLTypeRepr<sizeof(T) * CHAR_BIT, ArithmeticCategoryV<T>>;

template<typename T>
using CLTypeT = typename CLType<T>::type;

template<typename T>
struct IsCLType: std::conditional_t<std::is_same_v<CLTypeT<T>, void>, std::false_type, std::true_type> {};

template<typename T>
constexpr auto IsCLTypeV = IsCLType<T>::value;

template<typename T> constexpr const char* getKernelTypeStrImpl();
template<> inline constexpr const char* getKernelTypeStrImpl<CLChar>()   { return "char"; }
template<> inline constexpr const char* getKernelTypeStrImpl<CLUChar>()  { return "uchar"; }
template<> inline constexpr const char* getKernelTypeStrImpl<CLShort>()  { return "short"; }
template<> inline constexpr const char* getKernelTypeStrImpl<CLUShort>() { return "ushort"; }
template<> inline constexpr const char* getKernelTypeStrImpl<CLInt>()    { return "int"; }
template<> inline constexpr const char* getKernelTypeStrImpl<CLUInt>()   { return "uint"; }
template<> inline constexpr const char* getKernelTypeStrImpl<CLLong>()   { return "long"; }
template<> inline constexpr const char* getKernelTypeStrImpl<CLULong>()  { return "ulong"; }
template<> inline constexpr const char* getKernelTypeStrImpl<CLHalf>()   { return "half"; }
template<> inline constexpr const char* getKernelTypeStrImpl<CLFloat>()  { return "float"; }
template<> inline constexpr const char* getKernelTypeStrImpl<CLDouble>() { return "double"; }

template<typename T>
constexpr const char* getKernelTypeStr() { return getKernelTypeStrImpl<CLType<T>>(); }

template<typename T> constexpr const char* getKernelExtensionStrImpl() { return ""; }
template<> inline constexpr const char* getKernelExtensionStrImpl<CLHalf>() {
    return "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
}
template<> inline constexpr const char* getKernelExtensionStrImpl<CLDouble>() {
    return "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
}

template<typename T>
constexpr const char* getKernelExtensionStr() {
    return getKernelExtensionStrImpl<CLType<T>>();
}

struct Kernel {
    cl_kernel slow = nullptr;
    cl_kernel fast = nullptr;
    cl_kernel start = nullptr;
    size_t local_size = 0;
    size_t num_thread_elems = 0;
    size_t max_subseq_cnt = 0;

    explicit operator bool() const {
        return slow and fast and start;
    }
};

class BSContext {
    cl_context m_ctx = nullptr;
    cl_device_id m_dev = nullptr;
    cl_command_queue m_q = nullptr;
    std::unordered_map<const char*, Kernel> m_kers;

public:
    BSContext();
    BSContext(const BSContext&) = delete;
    BSContext(BSContext&&) = delete;
    BSContext& operator=(const BSContext&) = delete;
    BSContext& operator=(BSContext&&) = delete;
    ~BSContext();

    cl_context context() const {
        return m_ctx;
    }

    cl_device_id device() const {
        return m_dev;
    }

    cl_command_queue queue() const {
        return m_q;
    }

    template<typename T>
    const Kernel& kernel();

private:
    template<typename T>
    Kernel buildKernel();
};

inline BSContext::BSContext() {
    cl_int ctx_err;
    m_ctx = clCreateContextFromType(nullptr, CL_DEVICE_TYPE_GPU, nullptr, nullptr, &ctx_err);
    if (ctx_err == CL_INVALID_PLATFORM) {
        throw std::runtime_error{"No OpenCL platforms found"};
    }
    if (ctx_err == CL_DEVICE_NOT_FOUND) {
        throw std::runtime_error{"No OpenCL devices found"};
    }

    clGetContextInfo(m_ctx, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &m_dev, nullptr);

    m_q = clCreateCommandQueue(m_ctx, m_dev, 0, nullptr);
}

inline BSContext::~BSContext() {
    clFinish(m_q);
    for (auto& [_, ker]: m_kers) {
        clReleaseKernel(ker.slow);
        clReleaseKernel(ker.fast);
    }
    clReleaseCommandQueue(m_q);
    clReleaseContext(m_ctx);
}

inline BSContext& getBSContext() {
    static BSContext ctx;
    return ctx;
}

template<typename T>
constexpr bool BSTypeSupportedImpl(BSContext&) {
    return true;
}

template<>
inline bool BSTypeSupportedImpl<CLHalf>(BSContext& ctx) {
    cl_device_fp_config fp_config;
    clGetDeviceInfo(ctx.device(), CL_DEVICE_HALF_FP_CONFIG, sizeof(fp_config), &fp_config, nullptr);
    return fp_config;
}

template<>
inline bool BSTypeSupportedImpl<CLDouble>(BSContext& ctx) {
    cl_device_fp_config fp_config;
    clGetDeviceInfo(ctx.device(), CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp_config), &fp_config, nullptr);
    return fp_config;
}

template<typename T>
bool BSTypeSupported(BSContext& ctx) {
    return BSTypeSupportedImpl<CLType<T>>(ctx);
}

template<typename T>
const Kernel& BSContext::kernel() {
    auto& ker = m_kers[getKernelTypeStr<T>()];
    if (!ker) {
        ker = buildKernel<T>();
    }
    return ker;
}

template<typename T>
auto minVStr() {
    return std::to_string(std::numeric_limits<T>::min());
}

template<>
inline auto minVStr<float>() {
    return "(-HUGE_VALF)";
}

template<>
inline auto minVStr<double>() {
    return "(-HUGE_VAL)";
}

template<typename T>
auto maxVStr() {
    return std::to_string(std::numeric_limits<T>::max());
}

template<>
inline auto maxVStr<float>() {
    return "HUGE_VALF";
}

template<>
inline auto maxVStr<double>() {
    return "HUGE_VAL";
}

inline std::string getKernelSourceStr() {
return R"(
__kernel void bitonicSort(
   __global KERNEL_TYPE* data, uint cnt,
   uint seq_cnt, uint subseq_cnt
) {
    size_t i = get_global_id(0);
    uint po2cnt = 1 << (32 - clz(cnt - 1));
    uint mask = (2 * i) | (2 * seq_cnt - 1) | (~(po2cnt - 1));
    bool b_ascending = !(popcount(mask) & 1);
    size_t block_base = (i & ~(subseq_cnt - 1)) * 2;
    size_t block_offt = i & (subseq_cnt - 1);
    size_t sml_idx = block_base + block_offt;
    size_t big_idx = sml_idx + subseq_cnt;
    KERNEL_TYPE data_sml = data[sml_idx];
    KERNEL_TYPE data_big = data[big_idx];
    if (b_ascending == (data_big < data_sml)) {
        KERNEL_TYPE temp = data_sml;
        data[sml_idx] = data_big;
        data[big_idx] = temp;
    }
}

#define LOCAL_MEM_SIZE  (LOCAL_MEM_SIZE_BYTES / sizeof(KERNEL_TYPE))
#define THREAD_ELEMENTS (LOCAL_MEM_SIZE / 2 / LOCAL_SIZE)

void loadCache(__global const KERNEL_TYPE* data, size_t cnt, __local KERNEL_TYPE* cache, uint seq_cnt) {
    for (uint k = 0; k < THREAD_ELEMENTS; k++) {
        size_t cidx = 2 * (k * LOCAL_SIZE + get_local_id(0));
        size_t base_didx = 2 * THREAD_ELEMENTS * (get_global_id(0) - get_local_id(0));
        size_t didx = base_didx + cidx;

        uint po2cnt = 1 << (32 - clz(cnt - 1));
        uint mask = ~(po2cnt - 1) | didx | (2 * seq_cnt - 1);
        bool b_ascending = !(popcount(mask) & 1);

        KERNEL_TYPE pad_value = b_ascending ? MAX_VALUE: MIN_VALUE;
        cache[cidx    ] = (didx     < cnt) ? data[didx    ]: pad_value;
        cache[cidx + 1] = (didx + 1 < cnt) ? data[didx + 1]: pad_value;
    }
}

void storeCache(__global KERNEL_TYPE* data, size_t cnt, __local const KERNEL_TYPE* cache) {
    for (uint k = 0; k < THREAD_ELEMENTS; k++) {
        size_t cidx = 2 * (k * LOCAL_SIZE + get_local_id(0));
        size_t base_didx = 2 * THREAD_ELEMENTS * (get_global_id(0) - get_local_id(0));
        size_t didx = base_didx + cidx;
        if (didx     < cnt) {
            data[didx    ] = cache[cidx];
        }
        if (didx + 1 < cnt) {
            data[didx + 1] = cache[cidx + 1];
        }
    }
}

__attribute__((reqd_work_group_size(LOCAL_SIZE, 1, 1)))
__kernel void bitonicSortFast(
   __global KERNEL_TYPE* data, uint cnt,
   uint seq_cnt, uint subseq_cnt
) {
    __local KERNEL_TYPE cache[LOCAL_MEM_SIZE];
    loadCache(data, cnt, cache, seq_cnt);
    barrier(CLK_LOCAL_MEM_FENCE);

    size_t didx = 2 * THREAD_ELEMENTS * get_global_id(0);
    uint po2cnt = 1 << (32 - clz(cnt - 1));
    uint mask = ~(po2cnt - 1) | didx | (2 * seq_cnt - 1);
    bool b_ascending = !(popcount(mask) & 1);

    for (; subseq_cnt; subseq_cnt /= 2) {
        for (uint k = 0; k < THREAD_ELEMENTS; k++) {
            size_t i = k * LOCAL_SIZE + get_local_id(0);
            size_t block_base = (i & ~(subseq_cnt - 1)) * 2;
            size_t block_offt = i & (subseq_cnt - 1);
            size_t sml_idx = block_base + block_offt;
            size_t big_idx = sml_idx + subseq_cnt;

            KERNEL_TYPE cache_sml = cache[sml_idx];
            KERNEL_TYPE cache_big = cache[big_idx];
            if (b_ascending == (cache_big < cache_sml)) {
                KERNEL_TYPE temp = cache_sml;
                cache[sml_idx] = cache_big;
                cache[big_idx] = temp;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    storeCache(data, cnt, cache);
};

__attribute__((reqd_work_group_size(LOCAL_SIZE, 1, 1)))
__kernel void bitonicSortStart(
   __global KERNEL_TYPE* data, uint cnt
) {
    __local KERNEL_TYPE cache[LOCAL_MEM_SIZE];
    loadCache(data, cnt, cache, LOCAL_MEM_SIZE / 2);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint seq_cnt = 1; seq_cnt < LOCAL_MEM_SIZE; seq_cnt *= 2) {
        for (uint subseq_cnt = seq_cnt; subseq_cnt; subseq_cnt /= 2) {
            for (uint k = 0; k < THREAD_ELEMENTS; k++) {
                size_t ti = k * LOCAL_SIZE + get_local_id(0);
                size_t block_base = (ti & ~(subseq_cnt - 1)) * 2;
                size_t block_offt = ti & (subseq_cnt - 1);
                size_t sml_idx = block_base + block_offt;
                size_t big_idx = block_base + block_offt + subseq_cnt;

                size_t didx = 2 * THREAD_ELEMENTS * (get_global_id(0) - get_local_id(0)) + sml_idx;
                uint po2cnt = 1 << (32 - clz(cnt - 1));
                uint mask = ~(po2cnt - 1) | didx | (2 * seq_cnt - 1);
                bool b_ascending = !(popcount(mask) & 1);

                KERNEL_TYPE cache_sml = cache[sml_idx];
                KERNEL_TYPE cache_big = cache[big_idx];
                if (b_ascending == (cache_big < cache_sml)) {
                    KERNEL_TYPE temp = cache_sml;
                    cache[sml_idx] = cache_big;
                    cache[big_idx] = temp;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    storeCache(data, cnt, cache);
};
)";
}

template<typename T>
Kernel BSContext::buildKernel() {
    auto src = std::string(getKernelExtensionStr<T>()) + getKernelSourceStr();;

    size_t max_local_size;
    cl_ulong max_slm_size_bytes;
    clGetDeviceInfo(m_dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_local_size), &max_local_size, nullptr);
    clGetDeviceInfo(m_dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(max_slm_size_bytes), &max_slm_size_bytes, nullptr);
    auto slm_size_bytes = max_slm_size_bytes / 8;
    auto slm_size = slm_size_bytes / sizeof(T);
    auto local_size = std::min(slm_size / 2, max_local_size);
    auto max_subseq_cnt = slm_size / 2;
    auto num_thread_elems = max_subseq_cnt / local_size;

    auto build_options = std::string(" -DKERNEL_TYPE=")         + getKernelTypeStr<T>() +
                                     " -DLOCAL_SIZE="           + std::to_string(local_size) +
                                     " -DLOCAL_MEM_SIZE_BYTES=" + std::to_string(slm_size_bytes) +
                                     " -DMIN_VALUE="            + minVStr<T>() +
                                     " -DMAX_VALUE="            + maxVStr<T>();

    auto src_c_str = src.c_str();
    auto prog = clCreateProgramWithSource(m_ctx, 1, &src_c_str, nullptr, nullptr);
    auto prog_err = clBuildProgram(prog, 1, &m_dev, build_options.c_str(), nullptr, nullptr);
    if (prog_err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_sz;
        clGetProgramBuildInfo(prog, m_dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_sz);
        std::vector<char> log(log_sz);
        clGetProgramBuildInfo(prog, m_dev, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr);
        throw std::runtime_error{std::string("Failed to build program:\n") + log.data()};
    }
    auto slow_ker = clCreateKernel(prog, "bitonicSort", nullptr);
    auto fast_ker = clCreateKernel(prog, "bitonicSortFast", nullptr);
    auto start_ker = clCreateKernel(prog, "bitonicSortStart", nullptr);

    clReleaseProgram(prog);

    return {
        .slow = slow_ker,
        .fast = fast_ker,
        .start = start_ker,
        .local_size = local_size,
        .num_thread_elems = num_thread_elems,
        .max_subseq_cnt = max_subseq_cnt,
    };
}

template<typename T>
T ceilDiv(T top, T bot) {
    return top / bot + (top % bot != 0);
}

template<typename T>
T pad(T x, T m) {
    return ceilDiv(x, m) * m;
}

template<typename T>
void BSRunKernel(BSContext& ctx, cl_mem buf, size_t cnt) {
    auto q = ctx.queue();
    auto ker = ctx.kernel<T>();
    auto po2cnt = std::bit_ceil(cnt);
    auto pad_cnt = po2cnt - cnt;
    clSetKernelArg(ker.slow, 0, sizeof(buf), &buf);
    clSetKernelArg(ker.fast, 0, sizeof(buf), &buf);
    clSetKernelArg(ker.start, 0, sizeof(buf), &buf);
    cl_uint clcnt = cnt;
    assert(clcnt == cnt);
    clSetKernelArg(ker.slow, 1, sizeof(clcnt), &clcnt);
    clSetKernelArg(ker.fast, 1, sizeof(clcnt), &clcnt);
    clSetKernelArg(ker.start, 1, sizeof(clcnt), &clcnt);
    {
        size_t global_sz = pad(ceilDiv(cnt / 2, ker.num_thread_elems), ker.local_size);
        clEnqueueNDRangeKernel(q, ker.start, 1, nullptr, &global_sz, nullptr, 0, nullptr, nullptr);
    }
    for (cl_uint seq_cnt = 2 * ker.max_subseq_cnt; seq_cnt < cnt; seq_cnt *= 2) {
        cl_uint subseq_cnt = seq_cnt;
        for (; subseq_cnt > ker.max_subseq_cnt; subseq_cnt /= 2) {
            clSetKernelArg(ker.slow, 2, sizeof(seq_cnt), &seq_cnt);
            clSetKernelArg(ker.slow, 3, sizeof(subseq_cnt), &subseq_cnt);
            auto rem = pad_cnt & (2 * subseq_cnt - 1);
            auto disable_cnt = std::min<size_t>(rem, subseq_cnt) + (pad_cnt - rem) / 2;
            size_t global_sz = po2cnt / 2 - disable_cnt;
            clEnqueueNDRangeKernel(q, ker.slow, 1, nullptr, &global_sz, nullptr, 0, nullptr, nullptr);
        }
        clSetKernelArg(ker.fast, 2, sizeof(seq_cnt), &seq_cnt);
        clSetKernelArg(ker.fast, 3, sizeof(subseq_cnt), &subseq_cnt);
        size_t global_sz = pad(ceilDiv(cnt / 2, ker.num_thread_elems), ker.local_size);
        clEnqueueNDRangeKernel(q, ker.fast, 1, nullptr, &global_sz, nullptr, 0, nullptr, nullptr);
    }
}

template<typename T>
void BS(BSContext& ctx, std::span<T> data) {
    auto buf = clCreateBuffer(ctx.context(), CL_MEM_COPY_HOST_PTR, data.size_bytes(), data.data(), nullptr);
    BSRunKernel<T>(ctx, buf, data.size());
    clEnqueueReadBuffer(ctx.queue(), buf, true, 0, data.size_bytes(), data.data(), 0, nullptr, nullptr);
    clReleaseMemObject(buf);
}
}

template<typename T>
    requires Detail::IsCLTypeV<T>
bool bitonicSortTypeSupported() {
    return Detail::BSTypeSupported<T>(Detail::getBSContext());
}

template<typename T>
    requires Detail::IsCLTypeV<T>
void bitonicSort(std::span<T> data) {
    assert(bitonicSortTypeSupported<T>());
    Detail::BS(Detail::getBSContext(), data);
}

template<std::contiguous_iterator I>
void bitonicSort(I first, I last) {
    bitonicSort(std::span(first, last));
}

template<std::input_iterator I>
void bitonicSort(I first, I last) {
    std::vector data(first, last);
    bitonicSort(data);
    std::copy(first, data.begin(), data.end());
}
