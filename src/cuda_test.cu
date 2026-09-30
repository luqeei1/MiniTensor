#include "tensor/CudaBuffer.hpp"
#include "tensor/CudaOps.hpp"

#include <iostream>

int main()
{
    constexpr std::size_t n = 5;

    float host_a[n] = {1, 2, 3, 4, 5};
    float host_b[n] = {10, 20, 30, 40, 50};
    float host_c[n] = {};

    CudaBuffer a(n);
    CudaBuffer b(n);
    CudaBuffer c(n);

    a.copy_from_host(host_a, n);
    b.copy_from_host(host_b, n);

    cuda_subtract(
        a.data(),
        b.data(),
        c.data(),
        n
    );

    c.copy_to_host(host_c, n);

    for (float value : host_c)
    {
        std::cout << value << " ";
    }

    std::cout << '\n';

    return 0;
}