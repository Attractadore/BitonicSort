#include "BitonicSort.hpp"

#include <chrono>
#include <execution>
#include <iostream>

int main() {
    size_t n;
    while (std::cin >> n) {
        std::vector<int> bdata(n);
        std::copy_n(std::istream_iterator<int>(std::cin), bdata.size(), bdata.begin());
        bitonicSort(bdata.begin(), bdata.begin() + 2);
        std::random_shuffle(bdata.begin(), bdata.end());
        std::vector sdata(bdata);
        std::random_shuffle(sdata.begin(), sdata.end());
        std::vector pdata(bdata);
        std::random_shuffle(pdata.begin(), pdata.end());

        auto t0 = std::chrono::steady_clock::now();
        bitonicSort(bdata.begin(), bdata.end());
        auto t1 = std::chrono::steady_clock::now();

        auto t2 = std::chrono::steady_clock::now();
#if NDEBUG
        std::sort(sdata.begin(), sdata.end());
#endif
        auto t3 = std::chrono::steady_clock::now();

        auto t4 = std::chrono::steady_clock::now();
#if NDEBUG
        std::sort(std::execution::par_unseq, pdata.begin(), pdata.end());
#endif
        auto t5 = std::chrono::steady_clock::now();

        auto bs = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        auto ss = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
        auto ps = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();
        std::cout << n << "," << bs << "," << ss << "," << ps << ",\n";
    }
}
