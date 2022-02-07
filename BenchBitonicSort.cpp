#include "BitonicSort.hpp"

#include <chrono>
#include <execution>
#include <iostream>

int main() {
    size_t n;
    while (std::cin >> n) {
        std::vector<int> bdata(n);
        std::copy_n(std::istream_iterator<int>(std::cin), bdata.size(), bdata.begin());
        std::vector sdata(bdata);
        std::vector pdata(bdata);

        auto t0 = std::chrono::steady_clock::now();
        bitonicSort(bdata.begin(), bdata.end());
        auto t1 = std::chrono::steady_clock::now();
        std::sort(sdata.begin(), sdata.end());
        auto t2 = std::chrono::steady_clock::now();
        std::sort(std::execution::par_unseq, pdata.begin(), pdata.end());
        auto t3 = std::chrono::steady_clock::now();

        auto bs = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        auto ss = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        auto ps = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
        std::cout << n << "," << bs << "," << ss << "," << ps << ",\n";
    }
}
