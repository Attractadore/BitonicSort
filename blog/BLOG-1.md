# Part 1 -- Basic implementation of bitonic sort
Bitonic sort is a parallel sorting algorithm that is relatively easy to implement on the GPU. In this first part, I am going to talk about how I implemented an unoptimized version of bitonic sort for integers using OpenCL.

## OpenCL basics -- Platforms, Devices, Contexts
OpenCL is a general purpose computing API designed by the Khronos Group.

To start using OpenCL, we have to select a platform and a device.
Platforms roughly correspond to the various OpenCL devices drivers that you have installed on your system, while devices are the physical devices that these drivers can talk to. For most people, this will be either a CPU or a GPU. For simplicity, in my implementation of bitonic sort I search only for GPUs.

Once we have a selected a platform and a device, we can create a context. Contexts roughly correspond to a connection to a device through a driver. Multiple contexts can be created, so one can run multiple programs that uses OpenCL at the same time.

Fortunately, OpenCL doesn't force us to search for a platform and a device manually to create a context by providing the `clCreateContextFromType` function. With it, we can specify only the device type that we want:
```C++
    cl_int ctx_err;
    auto ctx = clCreateContextFromType(nullptr, CL_DEVICE_TYPE_GPU, nullptr, nullptr, &ctx_err);
    if (ctx_err == CL_INVALID_PLATFORM) {
        throw std::runtime_error{"No OpenCL platforms found"};
    }
    if (ctx_err == CL_DEVICE_NOT_FOUND) {
        throw std::runtime_error{"No OpenCL devices found"};
    }
```
When the first parameter is NULL, `clCreateContextFromType` selects the default platform and then searches it for a GPU. If there are no GPUs in the default platform, context creation will fail, even though there might be another platform that does contain a GPU. So a more robust approach would be to enumerate all platforms manually and then pick a suitable device.

## Command Queues
Any work that you want to be done on a device has to be submited through command queues, so let's create one. Queues are device specific, so we first query what device OpenCL has decided to use with our context:
```C++
    cl_device_id dev;
    clGetContextInfo(m_ctx, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &dev, nullptr);
    auto q = clCreateCommandQueue(m_ctx, m_dev, 0, nullptr);
```

## Caching the context
All of this stuff is getting a bit messy, so lets create a class to manage it:
```C++
class BSContext {
private:
    cl_context m_ctx = nullptr;
    cl_device_id m_dev = nullptr;
    cl_command_queue m_q = nullptr;

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
};
```

Nothing intersting happens in the constructor:
```C++
inline BSContext::BSContext() {
    // Just initialize all members
    ...
}
```

Or the destructor:
```C++
inline BSContext::~BSContext() {
    // Just clean everything up
    clFinish(m_q);
    clReleaseCommandQueue(m_q);
    clReleaseContext(m_ctx);
}
```
The clFinish call is used to wait for all work that was previously submitted to the queue to finish.

Context creation is quite expensive, so let's create a static variable so only one is ever created. We will than pass it by reference to the function that is actually doing the sorting:
```C++
template<typename T>
void bitonicSort(std::span<T> data) {
    static Detail::BSContext ctx;
    Detail::BS(ctx, data);
}
```

Let's also add a few simple wrappers since not all data that we might be requested to sort is necesserily contiguous:
```C++
template<std::input_iterator I>
void bitonicSort(I first, I last) {
    std::vector data(first, last);
    bitonicSort(data.begin(), data.end());
    std::copy(first, data.begin(), data.end());
}
```

Or not a pair of iterators:
```C++
template<std::contiguous_iterator I>
void bitonicSort(I first, I last) {
    bitonicSort(std::span(first, last));
}
```

## Buffers
Memory that OpenCL devices access while performing various operations is stored in OpenCL buffers. Let's create a buffer to store the data that we want to sort:
```C++
template<typename T>
void BS(BSContext& ctx, std::span<T> data) {
    auto buf = clCreateBuffer(ctx.context(), CL_MEM_COPY_HOST_PTR, data.size_bytes(), data.data(), nullptr);
    ...
}
```
As you can see, buffers are created within a context. CL_MEM_COPY_HOST_PTR means that we are going to provide an array with data for the buffer's initial contents.

## Kernels
Now that we have our data in device-accessible memory, let's write a program that will actually do something interesting with it. In OpenCL terminology, programs written for devices are called kernels. OpenCL kernels are written in OpenCL C and follow the SIMT execution model. SIMT means that while we have multiple threads, they are all executing the same instruction, and only the data differs. So OpenCL C is basicly C with functionality added for SIMT programming.

Let's take a look a kernel implementing bitonic sort:
```C
 #define KERNEL_TYPE int
 
 __attribute__((reqd_work_group_size(1, 1, 1)))
 __kernel void bitonicSort(__global KERNEL_TYPE* data, uint seq_cnt, uint subseq_cnt) {
    size_t i = get_global_id(0);
    size_t sml_idx = (i & (subseq_cnt - 1)) | ((i & ~(subseq_cnt - 1)) << 1);
    size_t big_idx = sml_idx + subseq_cnt;
    bool swap_cond = !(i & seq_cnt);
    if (swap_cond == (data[big_idx] < data[sml_idx])) 
        KERNEL_TYPE temp = data[sml_idx];
        data[sml_idx] = data[big_idx];
        data[big_idx] = temp;
    }
 };
```
I'm going to skip over reqd_work_group_size for now. Other than that, the only OpenCL C features that are used in this kernel are the `__kernel` and `__global` specifiers, as well as the `get_global_id(0)` function call. Let's go over them.

`__kernel` specifies that this function can be used as an entry point. We will see what this means a bit later.

`__global` is used to specify that a pointer points to RAM and not some other memory type. In OpenCL C you can choose to directly access data stored at different levels of the memory hierarchy, as opposed to C where memory is considered to be flat, so specifiers like `__global` have to be used.

`get_global_id(0)` is where things get interesting. Kernels are launched to perform work in N dimensions, and each dimension has an integral size. This is called an NDRange. Each thread is assigned a unique coordinate vector within this NDRange, which can be used however you like. In the case of bitonic sort, N is 1, so our coordinate is a simple 1D index which we can use to deside what elements in the buffer we are going to access. We retrive it by calling `get_global_id(0)` to get a thread's coordinate in dimension 0.

You might have also notices that the buffer's size is not explicitly specified. This is fine, since we can implicitly specify it by launching different amounts of threads.

## Bitonic sort
[Wikipedia](https://en.wikipedia.org/wiki/Bitonic_sorter) has an article that explains how bitonic sort works in detail. I'm going to quickly go over it.

A sequence of length n is bitonic if it does not decrease from 0 to k, and does not increase from k to n. The smallest bitonic sequence is a pair to 2 numbers.

Now let's sort a bitonic sequence of length 2n in ascending order. For each element k from 0 to n, we will compare it with element k+n and swap them if element k is greater. The elements from 0 to n now form a bitonic sequence, and so do the elements from n to 2n. At the same time, any element from 0 to n is smaller than any element from n to 2n. I will leave these two points without proof. After we sort the two new bitonic sequences in ascending order, we will have a single sequence sorted in ascending order.

We can obtain a bitonic sequence of length 2n by combining a sequence of length n sorted in ascending order and a sequence of length n sorted in descending order.

By combining these two procedures we can build a sorting algorithm for an array of length l, where l is a power of 2. Start by creating l/2 sorted sequences of length 2 -- this only requires swapping each pair of elements if they are not already in sorted order. These sequences can now be used to created l/4 bitonic sequences of length 4, which can then be sorted. Continue doing this until you get a sorted sequence of length l.

## Bitonic sort kernel
In the kernel that I wrote, each thread is responcible for comparing 2 elements and swapping them if necessary. `seq_cnt` is half the length of the top level sequence that we are currently sorting and is used to determine the swap condition `swap_cond` (the sort order). `subseq_cnt` is half the length of the subsequence of the top level sequence that we are currently sorting and is used to determine which items we are actually supposed to compare and swap.  

Currently, the only data type that is supported is `int`.

## Building the kernel
Before we can run a kernel, we first have to build it:
```C++
template<typename T>
inline void BSContext::buildKernel() {
    // Easiest to provide source inline, but could also load from file.
    const char* src = ...;
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
```
In OpenCL, kernel compilation happens through program objects. After creating a program object and specifying it's source code by using `clCreateProgramWithSource`, we compile it by calling `clBuildProgram`. If compilation fails (must often this happens if you made an error in your kernel source), we can extract a program's build log by using `clGetProgramBuildInfo`.
Once the program has been built, we can use it to create a kernel object using `clCreateKernel`. The second parameter specifies an entrypoint -- you can have multiple kernels is a single source file, so there hase to be a way to specify which entry point you want use for a kernel. Here I use `"bitonicSort"` as my entrypoint.

After we have created all the kernels that we need from a program object, we can free it by calling `clReleaseProgram`.

Since rebuilding the kernel all the time can be expensive, I cache it in `BSContext`. If it has not already been built, we first build it before giving it to the user when they request it:
```C++
template<typename T>
inline cl_kernel BSContext::kernel() {
    static_assert(std::same_as<T, int>);

    if (!m_ker) {
        buildKernel<T>();
    }

    return m_ker;
}
```

## Running the kernel
Now that we have build the kernel, we can send it to a command queue for execution. Let's create a separate function for that:
```C++
template<typename T>
void BS(BSContext& ctx, std::span<T> data) {
    auto buf = clCreateBuffer(ctx.context(), CL_MEM_COPY_HOST_PTR, data.size_bytes(), data.data(), nullptr);
    BSRunKernel<T>(ctx, buf, data.size());
    ...
}
```

```C++
template<typename T>
inline void BSRunKernel(BSContext& ctx, cl_mem buf, size_t cnt) {
    auto q = ctx.queue();
    auto ker = ctx.kernel<T>();
    ...
}
```

First, we need to determine how many threads we need to launch. Since each thread operates on 2 elements, we should launch half as many threads as there are elements:
```C++
    size_t global_sz = cnt / 2;
    assert((cnt & (cnt - 1)) == 0);
```
We also check that we are working on a buffer whose length is a power of 2.

Then we set the kernels arguments using `clSetKernelArg` and launch the kernel with `clEnqueueNDRangeKernel`:
```C++
    clSetKernelArg(ker, 0, sizeof(buf), &buf);
    for (cl_uint seq_cnt = 1; seq_cnt <= cnt / 2; seq_cnt *= 2) {
        for (cl_uint subseq_cnt = seq_cnt; subseq_cnt >= 1; subseq_cnt /= 2) {
            clSetKernelArg(ker, 1, sizeof(seq_cnt), &seq_cnt);
            clSetKernelArg(ker, 2, sizeof(subseq_cnt), &subseq_cnt);
            clEnqueueNDRangeKernel(ctx.queue(), ctx.kernel<T>(), 1, nullptr, &global_sz, nullptr, 0, nullptr, nullptr);
        }
    }
```

## Reading the results back
After the kernel has finished executing, we need some way to copy results from the buffer that we created back into CPU memory. We use `clEnqueueReadBuffer` for that:
```C++
template<typename T>
void BS(BSContext& ctx, std::span<T> data) {
    auto buf = clCreateBuffer(ctx.context(), CL_MEM_COPY_HOST_PTR, data.size_bytes(), data.data(), nullptr);
    BSRunKernel<T>(ctx, buf, data.size());
    clEnqueueReadBuffer(ctx.queue(), buf, true, 0, data.size_bytes(), data.data(), 0, nullptr, nullptr);
}
```

## Conclusion
We now have a parallel sorting algorithm that runs on an OpenCL device.
The complete implementation can be found [here](https://github.com/Attractadore/BitonicSort/blob/blog-1/BitonicSort.hpp).
In the next part we will look at how make it usable with types other than `int`.