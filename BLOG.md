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

Any work that you want to be done on a device has to be submited through command queues,
so let's create one.
Queues are device specific, so we first query what device OpenCL has decided to use with our context:
```C++
    cl_device_id dev;
    clGetContextInfo(m_ctx, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &dev, nullptr);
    auto q = clCreateCommandQueue(m_ctx, m_dev, 0, nullptr);
```
