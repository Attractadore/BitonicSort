#include "BitonicSort.hpp"

#include <iostream>
#include <vector>
#include <iostream>

int main() {
    size_t n;
    std::cin >> n;
    std::vector<int> data(n);
    std::copy_n(std::istream_iterator<int>(std::cin), data.size(), data.begin());
    bitonicSort(data.begin(), data.end());
    std::copy(data.begin(), data.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << "\n";
}
