#include "BitonicSort.hpp"

#include <gtest/gtest.h>

template<typename T>
void TestGeneratedPO2() {
    constexpr unsigned min_size = 1u << 0;
    constexpr unsigned max_size = 1u << 20;

    std::vector<T> data, expected;
    data.reserve(max_size);
    expected.reserve(max_size);
    bitonicSort(data.begin(), data.end());
    std::sort(expected.begin(), expected.end());
    ASSERT_EQ(data, expected);
    for (unsigned long long i = min_size; i <= max_size; i *= 2) {
        data.resize(i);
        //std::cout << "PO2 " << data.size() << "\n";
        for (auto& e: data) {
            e = rand();
        }
        expected = data;
        bitonicSort(data.begin(), data.end());
        std::sort(expected.begin(), expected.end());
        ASSERT_EQ(data, expected);
    }
}

template<typename T>
void TestGeneratedNonPO2() {
    constexpr unsigned min_size = 1u << 0;
    constexpr unsigned max_size = 1u << 20;

    std::vector<T> data, expected;
    data.reserve(max_size);
    expected.reserve(max_size);
    bitonicSort(data.begin(), data.end());
    std::sort(expected.begin(), expected.end());
    ASSERT_EQ(data, expected);
    for (unsigned long long i = min_size, j = 0; i <= max_size; i *= 10, j++) {
        data.resize(i + j);
        //std::cout << "NonPO2 " << data.size() << "\n";
        for (auto& e: data) {
            e = rand();
        }
        expected = data;
        bitonicSort(data.begin(), data.end());
        std::sort(expected.begin(), expected.end());
        ASSERT_EQ(data, expected);
    }
}

template<typename T>
void TestGenerated() {
    if (!bitonicSortTypeSupported<T>()) {
        return;
    }
    TestGeneratedPO2<T>();
    TestGeneratedNonPO2<T>();
}

TEST(TestBitonicSort, TestGeneratedChar) {
    TestGenerated<char>();
}

TEST(TestBitonicSort, TestGeneratedSChar) {
    TestGenerated<signed char>();
}

TEST(TestBitonicSort, TestGeneratedUChar) {
    TestGenerated<unsigned char>();
}

TEST(TestBitonicSort, TestGeneratedShort) {
    TestGenerated<short>();
}

TEST(TestBitonicSort, TestGeneratedUShort) {
    TestGenerated<unsigned short>();
}

TEST(TestBitonicSort, TestGeneratedInt) {
    TestGenerated<int>();
}

TEST(TestBitonicSort, TestGeneratedUInt) {
    TestGenerated<unsigned int>();
}

TEST(TestBitonicSort, TestGeneratedLong) {
    TestGenerated<long>();
}

TEST(TestBitonicSort, TestGeneratedULong) {
    TestGenerated<unsigned long>();
}

TEST(TestBitonicSort, TestGeneratedLongLong) {
    TestGenerated<long long>();
}

TEST(TestBitonicSort, TestGeneratedULongLong) {
    TestGenerated<unsigned long long>();
}

TEST(TestBitonicSort, TestGeneratedFloat) {
    TestGenerated<float>();
}

TEST(TestBitonicSort, TestGeneratedDouble) {
    TestGenerated<double>();
}

TEST(TestBitonicSort, TestGeneratedMultiple) {
    TestGenerated<short>();
    TestGenerated<int>();
    TestGenerated<float>();
}
