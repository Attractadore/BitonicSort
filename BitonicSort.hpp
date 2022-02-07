#pragma once
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#include <CL/opencl.h>

#include <cassert>
#include <iterator>
#include <vector>

namespace Detail {
template<typename T>
constexpr const char* KernelTypeName();

template<>
constexpr const char* KernelTypeName<int>() {
    return "int";
}

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
inline cl_event BSRunKernel(BSContext& ctx, cl_mem buf, size_t cnt, cl_event start_e) {
    auto dev = ctx.device();
    auto q = ctx.queue();
    auto ker = ctx.kernel<T>();

    cl_event prev_e = start_e;
    cl_event cur_e = prev_e;
    size_t global_sz = cnt / 2;
    {
        size_t local_sz[3];
        clGetKernelWorkGroupInfo(ker, dev, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(local_sz), local_sz, nullptr);
        global_sz = (global_sz / local_sz[0] + (global_sz % local_sz[0] != 0)) * local_sz[0];
    }
    clSetKernelArg(ker, 0, sizeof(buf), &buf);
    cl_uint clcnt = cnt;
    assert(clcnt == cnt);
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

template<std::input_iterator I, typename Derived>
class BSBufferAccessBase {
protected:
    using T = std::iter_value_t<I>;
    I m_first, m_last;

public:

    BSBufferAccessBase(BSContext& ctx, I first, I last):
        m_first(first), m_last(last) {}

    cl_mem getDeviceBuffer() const {
        return static_cast<const Derived*>(this)->getDeviceBuffer();
    }

    size_t getBufferItemCount() const {
        return std::distance(m_first, m_last);
    }
    
    size_t getBufferSize() const {
        return sizeof(T) * getBufferItemCount();
    }
};

template<std::forward_iterator I>
class BSForwardBufferAccess: public BSBufferAccessBase<I, BSForwardBufferAccess<I>> {
    using BaseClass = BSBufferAccessBase<I, BSForwardBufferAccess<I>>;
    using T = BaseClass::T;
    cl_mem m_host_buf = nullptr;
    cl_mem m_dev_buf = nullptr;

public:
    BSForwardBufferAccess(BSContext& ctx, I first, I last);
    BSForwardBufferAccess(const BSForwardBufferAccess<I>&) = delete;
    BSForwardBufferAccess(BSForwardBufferAccess<I>&&) = delete;
    BSForwardBufferAccess& operator=(const BSForwardBufferAccess<I>&) = delete;
    BSForwardBufferAccess& operator=(BSForwardBufferAccess<I>&&) = delete;
    ~BSForwardBufferAccess() {
        clReleaseMemObject(m_host_buf);
        clReleaseMemObject(m_dev_buf);
    }

    [[nodiscard]]
    cl_event copyToDevice(BSContext& cl);
    [[nodiscard]]
    cl_event copyFromDevice(BSContext& cl, cl_event start_e);

    cl_mem getDeviceBuffer() const {
        return m_dev_buf;
    }
};

template<std::forward_iterator I>
BSForwardBufferAccess<I>::BSForwardBufferAccess(BSContext& ctx, I first, I last):
    BaseClass::BSBufferAccessBase(ctx, first, last) 
{
    auto buf_sz = this->getBufferSize();
    m_host_buf = clCreateBuffer(ctx.context(), CL_MEM_ALLOC_HOST_PTR, buf_sz, nullptr, nullptr);
    m_dev_buf = clCreateBuffer(ctx.context(), CL_MEM_HOST_NO_ACCESS, buf_sz, nullptr, nullptr);
}

template<std::forward_iterator I>
cl_event BSForwardBufferAccess<I>::copyToDevice(BSContext& ctx) {
    auto buf_sz = this->getBufferSize();
    auto write_map = reinterpret_cast<T*>(
        clEnqueueMapBuffer(ctx.queue(), m_host_buf, true, CL_MAP_WRITE_INVALIDATE_REGION, 0, buf_sz, 0, nullptr, nullptr, nullptr)
    );
    std::copy(this->m_first, this->m_last, write_map);
    cl_event unmap_write_e;
    clEnqueueUnmapMemObject(ctx.queue(), m_host_buf, write_map, 0, nullptr, &unmap_write_e);
    cl_event copy_e;
    clEnqueueCopyBuffer(ctx.queue(), m_host_buf, m_dev_buf, 0, 0, buf_sz, 1, &unmap_write_e, &copy_e);
    return copy_e;
}

template<std::forward_iterator I>
cl_event BSForwardBufferAccess<I>::copyFromDevice(BSContext& ctx, cl_event start_e) {
    auto cnt = this->getBufferItemCount();
    auto buf_sz = this->getBufferSize();
    cl_event copy_e;
    clEnqueueCopyBuffer(ctx.queue(), m_dev_buf, m_host_buf, 0, 0, buf_sz, start_e ? 1: 0, start_e ? &start_e: nullptr, &copy_e);
    auto read_map = reinterpret_cast<T*>(
        clEnqueueMapBuffer(ctx.queue(), m_host_buf, true, CL_MAP_READ, 0, buf_sz, 1, &copy_e, nullptr, nullptr)
    );
    std::copy_n(read_map, cnt, this->m_first);
    cl_event unmap_e;
    clEnqueueUnmapMemObject(ctx.queue(), m_host_buf, read_map, 0, nullptr, &unmap_e);
    return unmap_e;
}

template<std::contiguous_iterator I>
class BSContiguousBufferAccess: public BSBufferAccessBase<I, BSContiguousBufferAccess<I>> {
    using BaseClass = BSBufferAccessBase<I, BSContiguousBufferAccess<I>>;
    cl_mem m_dev_buf = nullptr;

public:
    BSContiguousBufferAccess(BSContext& ctx, I first, I last);
    BSContiguousBufferAccess(const BSContiguousBufferAccess<I>&) = delete;
    BSContiguousBufferAccess(BSContiguousBufferAccess<I>&&) = delete;
    BSContiguousBufferAccess& operator=(const BSContiguousBufferAccess<I>&) = delete;
    BSContiguousBufferAccess& operator=(BSContiguousBufferAccess<I>&&) = delete;
    ~BSContiguousBufferAccess() {
        clReleaseMemObject(m_dev_buf);
    }

    [[nodiscard]]
    cl_event copyToDevice(BSContext& cl);
    [[nodiscard]]
    cl_event copyFromDevice(BSContext& cl, cl_event start_e);

    cl_mem getDeviceBuffer() const {
        return m_dev_buf;
    }
};

template<std::contiguous_iterator I>
BSContiguousBufferAccess<I>::BSContiguousBufferAccess(BSContext& ctx, I first, I last):
    BaseClass::BSBufferAccessBase(ctx, first, last) 
{
    auto buf_sz = this->getBufferSize();
    m_dev_buf = clCreateBuffer(ctx.context(), CL_MEM_COPY_HOST_PTR, buf_sz, &*this->m_first, nullptr);
}

template<std::contiguous_iterator I>
cl_event BSContiguousBufferAccess<I>::copyToDevice(BSContext& ctx) {
    return nullptr;
}

template<std::contiguous_iterator I>
cl_event BSContiguousBufferAccess<I>::copyFromDevice(BSContext& ctx, cl_event start_e) {
    auto buf_sz = this->getBufferSize();
    cl_event copy_e;
    clEnqueueReadBuffer(ctx.queue(), m_dev_buf, false, 0, buf_sz, &*this->m_first, start_e ? 1: 0, start_e ? &start_e: nullptr, &copy_e);
    return copy_e;
}

template <typename I>
class BSBufferAccess;

template<std::forward_iterator I>
class BSBufferAccess<I>: public BSForwardBufferAccess<I> {
    using BSForwardBufferAccess<I>::BSForwardBufferAccess;
};

template<std::contiguous_iterator I>
class BSBufferAccess<I>: public BSContiguousBufferAccess<I> {
    using BSContiguousBufferAccess<I>::BSContiguousBufferAccess;
};

template<std::input_iterator I>
void BS(BSContext& ctx, I first, I last) {
    using T = std::iter_value_t<I>;
    BSBufferAccess<I> buf_acc(ctx, first, last);
    auto write_e = buf_acc.copyToDevice(ctx);
    auto kernel_e = BSRunKernel<T>(
        ctx,
        buf_acc.getDeviceBuffer(), buf_acc.getBufferItemCount(),
        write_e
    );
    auto read_e = buf_acc.copyFromDevice(ctx, kernel_e);
    clWaitForEvents(1, &read_e);
}
}

template<std::input_iterator I>
void bitonicSort(I first, I last) {
    static Detail::BSContext ctx;
    Detail::BS(ctx, first, last);
}
