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
    return "#pragma OPENCL EXTENSION cl_khr_fp16 : require\n";
}
template<> inline constexpr const char* getKernelExtensionStrImpl<CLDouble>() {
    return "#pragma OPENCL EXTENSION cl_khr_fp64 : require\n";
}

template<typename T>
constexpr const char* getKernelExtensionStr() {
    return getKernelExtensionStrImpl<CLType<T>>();
}

class BSContext {
private:
    cl_context m_ctx = nullptr;
    cl_device_id m_dev = nullptr;
    cl_command_queue m_q = nullptr;
    std::unordered_map<const char*, cl_kernel> m_kers;

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
    cl_kernel kernel();

private:
    template<typename T>
    cl_kernel buildKernel();
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
        clReleaseKernel(ker);
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
cl_kernel BSContext::kernel() {
    auto& ker = m_kers[getKernelTypeStr<T>()];
    if (!ker) {
        ker = buildKernel<T>();
    }
    return ker;
}

template<typename T>
cl_kernel BSContext::buildKernel() {
    auto src =
    std::string(getKernelExtensionStr<T>()) +
    R"(
    __attribute__((reqd_work_group_size(1, 1, 1)))
    __kernel void bitonicSort(
       __global KERNEL_TYPE* data, uint cnt,
       uint seq_cnt, uint subseq_cnt
    ) {
        size_t i = get_global_id(0);
        uint po2cnt = 1 << (32 - clz(cnt - 1));
        uint mask = ((2 * i)) | (2 * seq_cnt - 1) | (~(po2cnt - 1));
        bool b_ascending = !(popcount(mask) & 1);
        size_t block_base = (i & ~(subseq_cnt - 1)) * 2;
        size_t block_offt = i & (subseq_cnt - 1);
        size_t sml_idx = (block_base | block_offt);
        size_t big_idx = sml_idx + subseq_cnt;
        if (b_ascending == (data[big_idx] < data[sml_idx])) {
            KERNEL_TYPE temp = data[sml_idx];
            data[sml_idx] = data[big_idx];
            data[big_idx] = temp;
        }
    })";
    auto src_c_str = src.c_str();
    auto build_options = std::string("-DKERNEL_TYPE=") + getKernelTypeStr<T>();

    auto prog = clCreateProgramWithSource(m_ctx, 1, &src_c_str, nullptr, nullptr);
    auto prog_err = clBuildProgram(prog, 1, &m_dev, build_options.c_str(), nullptr, nullptr);
    if (prog_err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_sz;
        clGetProgramBuildInfo(prog, m_dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_sz);
        std::vector<char> log(log_sz);
        clGetProgramBuildInfo(prog, m_dev, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr);
        throw std::runtime_error{std::string("Failed to build program:\n") + log.data()};
    }

    auto ker = clCreateKernel(prog, "bitonicSort", nullptr);

    clReleaseProgram(prog);

    return ker;
}

template<typename T>
inline void BSRunKernel(BSContext& ctx, cl_mem buf, size_t cnt) {
    auto q = ctx.queue();
    auto ker = ctx.kernel<T>();
    auto po2cnt = std::bit_ceil(cnt);
    auto pad_cnt = po2cnt - cnt;
    clSetKernelArg(ker, 0, sizeof(buf), &buf);
    cl_uint clcnt = cnt;
    assert(clcnt == cnt);
    clSetKernelArg(ker, 1, sizeof(clcnt), &clcnt);
    for (cl_uint seq_cnt = 1; seq_cnt < cnt; seq_cnt *= 2) {
        for (cl_uint subseq_cnt = seq_cnt; subseq_cnt >= 1; subseq_cnt /= 2) {
            clSetKernelArg(ker, 2, sizeof(seq_cnt), &seq_cnt);
            clSetKernelArg(ker, 3, sizeof(subseq_cnt), &subseq_cnt);
            auto rem = pad_cnt & (2 * subseq_cnt - 1);
            auto disable_cnt = std::min<size_t>(rem, subseq_cnt) + (pad_cnt - rem) / 2;
            size_t global_sz = po2cnt / 2 - disable_cnt;
            clEnqueueNDRangeKernel(q, ker, 1, nullptr, &global_sz, nullptr, 0, nullptr, nullptr);
        }
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
