#include "BitonicSort.hpp"

#include <gtest/gtest.h>

#include <chrono>

TEST(TestBitonicSort, TestGenerated) {
    constexpr unsigned min_size = 1u << 1; 
    constexpr unsigned max_size = 1u << 27; 

    for (unsigned long long i = min_size; i <= max_size; i *= 2) {
        std::vector<int> data(i);
        for (auto& e: data) {
            e = rand();
        }
        std::vector<int> expected = data;
        auto t0 = std::chrono::steady_clock::now();
        bitonicSort(data.begin(), data.end());
        auto t1 = std::chrono::steady_clock::now();
        std::sort(expected.begin(), expected.end());
        auto t2 = std::chrono::steady_clock::now();
        std::cout << std::fixed;
        std::cout << data.size() << ", ";
        std::cout << (t1 - t0).count() / 1e9 << ", ";
        std::cout << (t2 - t1).count() / 1e9 << "\n";
        ASSERT_EQ(data, expected);
    }
}
