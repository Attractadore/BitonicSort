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
#include <fstream>
#include <filesystem>

#include <iostream>
#include <chrono>

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
    auto notify = [] (const char *errinfo, const void *private_info, size_t cb, void *user_data) {
        std::cerr << "OpenCL error: " << errinfo << "\n";
    };
    cl_int ctx_err;
    m_ctx = clCreateContextFromType(nullptr, CL_DEVICE_TYPE_GPU, notify, nullptr, &ctx_err);
    if (ctx_err == CL_INVALID_PLATFORM) {
        throw std::runtime_error{"No OpenCL platforms found"};
    }
    if (ctx_err == CL_DEVICE_NOT_FOUND) {
        throw std::runtime_error{"No OpenCL devices found"};
    }

    clGetContextInfo(m_ctx, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &m_dev, nullptr);

    m_q = clCreateCommandQueue(m_ctx, m_dev, CL_QUEUE_PROFILING_ENABLE, nullptr);
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
    if constexpr (std::is_floating_point_v<T>) {
        return "\"(-1.0 / 0.0)\"";
    }
    else {
        return std::to_string(std::numeric_limits<T>::min());
    }
};

template<typename T>
auto maxVStr() {
    if constexpr (std::is_floating_point_v<T>) {
        return "\"(1.0 / 0.0)\"";
    }
    else {
        return std::to_string(std::numeric_limits<T>::max());
    }
};

struct ProfileEvent {
    cl_event event;
    std::string name;
};

inline auto& getEventVector() {
    static std::vector<ProfileEvent> events;
    return events;
}

template<typename T>
void measure(T f, std::string_view msg) {
#if MEASURE
    auto& pe = getEventVector().emplace_back();
    pe.name = msg;
    f(&pe.event);
#else
    cl_event e;
    f(&e);
#endif
}

inline void measureFinalize() {
#if MEASURE
    size_t fast_run_time = 0;
    size_t fast_count = 0;
    for (const auto& pe: getEventVector()) {
        cl_ulong enque_time, submit_time, start_time, end_time;
        clGetEventProfilingInfo(pe.event, CL_PROFILING_COMMAND_QUEUED, sizeof(enque_time), &enque_time, nullptr);
        clGetEventProfilingInfo(pe.event, CL_PROFILING_COMMAND_SUBMIT, sizeof(submit_time), &submit_time, nullptr);
        clGetEventProfilingInfo(pe.event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, nullptr);
        clGetEventProfilingInfo(pe.event, CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, nullptr);
        auto in_queue_time = (start_time - enque_time) / 1000;
        auto run_time = (end_time - start_time) / 1000;
        std::cout << "Event \"" << pe.name << "\" Running: " << run_time << ", In queue: " << in_queue_time << "\n";
        if (pe.name.starts_with("Run fast kernel")) {
            fast_run_time += run_time;
            fast_count++;
        }
    }
    std::cout << "Fast kernel runtime is " << (fast_count ? (fast_run_time / fast_count): 0) << "\n";
#endif
}

inline std::string getKernelSourceStr() {
    auto file_path = "kernels.clc";
    std::ifstream f(file_path, std::ios::binary);
    auto sz = std::filesystem::file_size(file_path);
    std::string str(sz, '\0');
    f.read(str.data(), sz);
    return str;
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
    auto fast_ker = clCreateKernel(prog, "bitonicSortLocal", nullptr);
    auto start_ker = clCreateKernel(prog, "bitonicSortFastStart", nullptr);

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
#if 1
    measure([&] (cl_event* e) {
        size_t global_sz = pad(ceilDiv(cnt / 2, ker.num_thread_elems), ker.local_size);
        clEnqueueNDRangeKernel(q, ker.start, 1, nullptr, &global_sz, nullptr, 0, nullptr, e);
    }, "Run start kernel");
    for (cl_uint seq_cnt = 2 * ker.max_subseq_cnt; seq_cnt < cnt; seq_cnt *= 2) {
#else
    for (cl_uint seq_cnt = 1; seq_cnt < cnt; seq_cnt *= 2) {
#endif
        cl_uint subseq_cnt = seq_cnt;
#if 1
        for (; subseq_cnt > ker.max_subseq_cnt; subseq_cnt /= 2) {
#else
        for (; subseq_cnt; subseq_cnt /= 2) {
#endif
            clSetKernelArg(ker.slow, 2, sizeof(seq_cnt), &seq_cnt);
            clSetKernelArg(ker.slow, 3, sizeof(subseq_cnt), &subseq_cnt);
            auto rem = pad_cnt & (2 * subseq_cnt - 1);
            auto disable_cnt = std::min<size_t>(rem, subseq_cnt) + (pad_cnt - rem) / 2;
            size_t global_sz = po2cnt / 2 - disable_cnt;
            measure([&] (cl_event* e) {
                clEnqueueNDRangeKernel(q, ker.slow, 1, nullptr, &global_sz, nullptr, 0, nullptr, e);
            }, std::string("Run slow kernel for " + std::to_string(seq_cnt) + "/" + std::to_string(subseq_cnt)));
        }
#if 1
        clSetKernelArg(ker.fast, 2, sizeof(seq_cnt), &seq_cnt);
        clSetKernelArg(ker.fast, 3, sizeof(subseq_cnt), &subseq_cnt);
        size_t global_sz = pad(ceilDiv(cnt / 2, ker.num_thread_elems), ker.local_size);
        measure([&] (cl_event* e) {
            clEnqueueNDRangeKernel(q, ker.fast, 1, nullptr, &global_sz, nullptr, 0, nullptr, e);
        }, std::string("Run fast kernel for " + std::to_string(seq_cnt) + "/" + std::to_string(subseq_cnt)));
#endif
    }
}

template<typename T>
void BS(BSContext& ctx, std::span<T> data) {
    auto buf = clCreateBuffer(ctx.context(), CL_MEM_COPY_HOST_PTR, data.size_bytes(), data.data(), nullptr);
    BSRunKernel<T>(ctx, buf, data.size());
    clEnqueueReadBuffer(ctx.queue(), buf, true, 0, data.size_bytes(), data.data(), 0, nullptr, nullptr);
    measureFinalize();
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
