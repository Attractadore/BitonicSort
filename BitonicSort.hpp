#pragma once
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#include <CL/opencl.h>

#include <cassert>
#include <iterator>
#include <span>
#include <vector>

namespace Detail {
class BSContext {
private:
    cl_context m_ctx;
    cl_device_id m_dev;
    cl_command_queue m_q;
    cl_kernel m_ker;

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
    cl_kernel kernel() const;
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

    const char* src =
    "#define KERNEL_TYPE int\n" 
    "__attribute__((reqd_work_group_size(256, 1, 1)))\n"
    "__kernel void bitonicSort(__global KERNEL_TYPE* data, uint cnt, uint seq_cnt, uint subseq_cnt) {\n"
    "   size_t i = get_global_id(0);\n"
    "   size_t sml_idx = (i & (subseq_cnt - 1)) | ((i & ~(subseq_cnt - 1)) << 1);\n"
    "   size_t big_idx = sml_idx + subseq_cnt;\n"
    "   bool swap_cond = !(i & seq_cnt);\n"
    "   if (swap_cond == (data[big_idx] < data[sml_idx])) {\n"
    "       KERNEL_TYPE temp = data[sml_idx];\n"
    "       data[sml_idx] = data[big_idx];\n"
    "       data[big_idx] = temp;\n"
    "   }\n"
    "}\n";

    auto prog = clCreateProgramWithSource(m_ctx, 1, &src, nullptr, nullptr);
    auto prog_err = clBuildProgram(prog, 1, &m_dev, "", nullptr, nullptr);
    if (prog_err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_sz;
        clGetProgramBuildInfo(prog, m_dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_sz);
        std::vector<char> log(log_sz);
        clGetProgramBuildInfo(prog, m_dev, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr);
        throw std::runtime_error{std::string("Failed to build program:\n") + log.data()};
    }
    m_ker = clCreateKernel(prog, "bitonicSort", nullptr);
    clReleaseProgram(prog);
}

inline BSContext::~BSContext() {
    clFinish(m_q);
    clReleaseKernel(m_ker);
    clReleaseCommandQueue(m_q);
    clReleaseContext(m_ctx);
}

template<typename T>
inline cl_kernel BSContext::kernel() const {
    static_assert(std::same_as<T, int>);
    return m_ker;
}

template<typename T>
inline cl_event BSRunKernel(BSContext& ctx, cl_mem buf, size_t cnt) {
    auto dev = ctx.device();
    auto q = ctx.queue();
    auto ker = ctx.kernel<T>();

    cl_event prev_e = nullptr;
    cl_event cur_e = prev_e;
    size_t global_sz = cnt / 2;
    {
        size_t local_sz[3];
        clGetKernelWorkGroupInfo(ker, dev, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(local_sz), local_sz, nullptr);
        global_sz = (global_sz / local_sz[0] + (global_sz % local_sz[0] != 0)) * local_sz[0];
    }
    cl_uint clcnt = cnt;
    assert(clcnt == cnt);
    clSetKernelArg(ker, 0, sizeof(buf), &buf);
    clSetKernelArg(ker, 1, sizeof(clcnt), &clcnt);
    for (cl_uint seq_cnt = 1; seq_cnt <= cnt / 2; seq_cnt *= 2) {
        for (cl_uint subseq_cnt = seq_cnt; subseq_cnt >= 1; subseq_cnt /= 2) {
            clSetKernelArg(ker, 2, sizeof(seq_cnt), &seq_cnt);
            clSetKernelArg(ker, 3, sizeof(subseq_cnt), &subseq_cnt);
            clEnqueueNDRangeKernel(q, ker, 1, nullptr, &global_sz, nullptr, prev_e ? 1: 0, prev_e ? &prev_e: nullptr, &cur_e);
            prev_e = cur_e;
        }
    }

    return cur_e;
}

template<typename T>
void BS(BSContext& ctx, std::span<T> data) {
    auto buf = clCreateBuffer(ctx.context(), CL_MEM_COPY_HOST_PTR, data.size_bytes(), data.data(), nullptr);
    auto kernel_e = BSRunKernel<T>(ctx, buf, data.size());
    clEnqueueReadBuffer(ctx.queue(), buf, true, 0, data.size_bytes(), data.data(), 1, &kernel_e, nullptr);
}
}

template<std::contiguous_iterator I>
void bitonicSort(I first, I last) {
    static Detail::BSContext ctx;
    Detail::BS(ctx, std::span(first, last));
}

template<std::input_iterator I>
void bitonicSort(I first, I last) {
    std::vector data(first, last);
    bitonicSort(data.begin(), data.end());
    std::copy(first, data.begin(), data.end());
}
