#include "BitonicSort.hpp"

#include <gtest/gtest.h>

TEST(TestBitonicSort, TestGeneratedInt) {
    constexpr unsigned min_size = 1u << 1; 
    constexpr unsigned max_size = 1u << 20;

    std::vector<int> data, expected;
    data.reserve(max_size);
    expected.reserve(max_size);
    for (unsigned long long i = min_size; i <= max_size; i *= 2) {
        data.resize(i);
        for (auto& e: data) {
            e = rand();
        }
        expected = data;
        bitonicSort(data.begin(), data.end());
        std::sort(expected.begin(), expected.end());
        ASSERT_EQ(data, expected);
    }
}
