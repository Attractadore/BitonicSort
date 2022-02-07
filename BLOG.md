# Part 1 -- Basic implementation of bitonic sort

Bitonic is a parallel sorting algorithm that is relatively easy to implement on the GPU.
For a more detail explanation, see [Wikipedia](https://en.wikipedia.org/wiki/Bitonic_sorter).

In this first part, I am going to talk how I implemented an unoptimized version of bitonic sort for integers.

## OpenCL
OpenCL is a general purpose computing API designed by the Khronos Group.
To start using OpenCL, we must create a context, and for that we have to select a platform and a device.
Platforms roughly correspond to the various OpenCL devices drivers that you have installed on your system, while devices are the physical devices that these drivers can talk to. For most people, this will be either a CPU or a GPU. In my implementation of bitonic sort I search only for GPUs for simplicity.
Once we have a selected a platform and a device, we can create a context. Contexts roughly correspond to a connection to a device through a driver. Multiple contexts can be created, so one can run multiple programs that uses OpenCL at the same time.

Fortunately, OpenCL doesn't force us to search for a platform and a device manually by providing the clCreateContextFromType function. With it, we can specify only the device type that we want:
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
When the first parameter is NULL, this function selects the default platform and then searches it for a GPU. If there are no GPUs in the default platform, context creation will fail, even though there might be another platform that does contain a GPU. So a more robust approach would be to enumerate all platforms manually and then pick the one that has the prefered device type.

Any work that you want to be done on a device has to be submited through command queues, so let's create one. Queues are device specific, so we first query what device OpenCL has decided to use with our context:
```C++
    cl_device_id dev;
    clGetContextInfo(m_ctx, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &dev, nullptr);
    auto q = clCreateCommandQueue(m_ctx, m_dev, 0, nullptr);
```

All of this is getting a bit messy, so lets create a class to manage it.
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
    // Initialize all members
    ...
}
```

But something interesting does happen in the destructor. Objects in OpenCL are reference counted. Calling the corresponding clRetain will increase the reference count by 1, while clRelease will decrement it by 1. Objects are created with a reference count of 1, and are freed when their reference count reaches 0.
```C++
inline BSContext::~BSContext() {
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

template<std::contiguous_iterator I>
void bitonicSort(I first, I last) {
    bitonicSort({first, last});
}
```

Memory that OpenCL devices access while performing various operations is stored in OpenCL buffers. Let's create a buffer:
```C++
template<typename T>
void BS(BSContext& ctx, std::span<T> data) {
    auto buf = clCreateBuffer(ctx.context(), CL_MEM_COPY_HOST_PTR, data.size_bytes(), data.data(), nullptr);
}
```
As you can see, buffers are created within a context. CL_MEM_COPY_HOST_PTR means that we are going to provide an array with data for the buffer's initial contents.

Now that we have our data in device-accessible memory, let's write a program that will do something with it.
OpenCL programs are written in OpenCL C and follow the SIMT execution model. SIMT means that while we have multiple threads, they are all executing the same instruction on different data.

```C
 #define KERNEL_TYPE int
 
 __attribute__((reqd_work_group_size(256, 1, 1)))
 __kernel void bitonicSort(__global KERNEL_TYPE* data, uint cnt, uint seq_cnt, uint subseq_cnt) {
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
