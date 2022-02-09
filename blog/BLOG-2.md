# Part 2 -- Type generic bitonic sort
Last time, we implemented bitonic sort in OpenCL. However, the only type that it supports is `int`, so let's change that.

First of all, since static objects would be created for every instantiation of a template, let's move our static `BSContext` instance from `bitonicSort` to a separate getter function:
```C++
namespace Detail {
inline BSContext& getBSContext() {
    static BSContext ctx;
    return ctx;
}
}

template<typename T>
void bitonicSort(std::span<T> data) {
    Detail::BS(Detail::getBSContext(), data);
}
```

Let's add a constraint to `bitonicSort` so it can only be called with that have an OpenCL C version. The full list of built-in scalar data types in OpenCL C can be found [here](https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/scalarDataTypes.html).
```C++
template<typename T>
    requires Detail::IsCLTypeV<T>
void bitonicSort(std::span<T> data) {
    Detail::BS(Detail::getBSContext(), data);
}
```

Let's take a look at `IsCLType`:
```C++
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

template<size_t BitWidth, ArithmeticCategory AC> struct CLTypeRepr { using type = void; };

template<typename T>
using CLType = CLTypeRepr<sizeof(T) * CHAR_BIT, ArithmeticCategoryV<T>>;

template<typename T>
using CLTypeT = typename CLType<T>::type;

template<typename T>
struct IsCLType: std::conditional_t<std::is_same_v<CLTypeT<T>, void>, std::false_type, std::true_type> {};

template<typename T>
constexpr auto IsCLTypeV = IsCLType<T>::value;
```
`CLTypeRepr` is used to identify types only by their bit width and arithmetic category. I added the necessary specializations to match all of OpenCL C's built-in types:
```C++
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
```
I chose not to rely on the typedefs that OpenCL provides to match device-side types on the host since they are incomplete for the purpose of mapping host types to device types. There is no typedef for `half` as a 16-bit floating-point type. This is understandable, since there is not standard type for `half` in C++. Instead, `cl_half` is the same as `cl_ushort`. Another problem is that `cl_long` corresponds to either `long` or `long long`, depending on the platform. And in C++ `long` is not the same type as `long long`, even though they might be of the same size. These two quirks mean that it's not possible to simply specialize `CLType` for all OpenCL typedefs and check its `type` field for `void` to decide whether we were given a valid type.

In OpenCL C, integer and float support is mandatory, while half and double support is optional. So let's allow the user check if they are supported before calling `bitonicSort`:
```C++
namespace Detail {
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
```
I call `clGetDeviceInfo` to retrieve the corresponding half or double precision capability flags. If the flags are not 0, than half or double operations are supported.

Now let's update our kernel source so it can be used with any valid type:
```C++
template<typename T>
inline cl_kernel BSContext::buildKernel() {
    auto src =
    std::string(getKernelExtensionStr<T>()) +
    "__attribute__((reqd_work_group_size(1, 1, 1)))\n"
    "__kernel void bitonicSort(__global KERNEL_TYPE* data, uint seq_cnt, uint subseq_cnt) {\n"
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
    auto src_c_str = src.c_str();
    auto build_options = std::string("-DKERNEL_TYPE=") + getKernelTypeStr<T>();

    auto prog = clCreateProgramWithSource(ctx, 1, &src_c_str, nullptr, nullptr);
    auto dev = ctx.device();
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
```
`getKernelTypeStr` returns the name of the OpenCL C type that corresponds to `T`. It is used to form the build options string that is passed to the kernel compiler. `getKernelExtensionStr` returns the string to enable all extensions that are required to use `T` as a kernel type.

OpenCL extensions are various pieces of optional functionality that might not be supported by all implementations. For example, `double` support requires the `cl_khr_fp64` extensions. To require it, the following `#pragma` can be used:
```
#pragma OPENCL EXTENSION cl_khr_fp64 : require
```
Kernel compilation will now fail if the device doesn't support the `cl_khr_fp64` extension.

Since can we now have multiple compiled kernels, let's add a map to `BSContext` to store them:
```C++
class BSContext {
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

template<typename T>
cl_kernel BSContext::kernel() {
    auto& ker = m_kers[getKernelTypeStr<T>()];
    if (!ker) {
        ker = buildKernel<T>();
    }
    return ker;
}
```

## Conclusion
In this, part we have made our bitonic sort type-generic.
The complete implementation can be found [here](https://github.com/Attractadore/BitonicSort/blob/blog-2/BitonicSort.hpp).
While the algorithm is now more useful, it's still limited by the fact that it can only be used sequences whose length is a power of 2. We will address this issue in the next part.
