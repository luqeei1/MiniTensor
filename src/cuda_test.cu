#include "tensor/CudaBuffer.hpp"

#include <cuda_runtime.h>
#include <iostream>

int main()
{
    constexpr std::size_t n = 5;

    float host_in[n] = {1, 2, 3, 4, 5};
    float host_out[n] = {};

    CudaBuffer buffer(n);

    cudaMemcpy(
        buffer.data(),
        host_in,
        n * sizeof(float),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        host_out,
        buffer.data(),
        n * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    for (float value : host_out)
    {
        std::cout << value << " ";
    }

    std::cout << '\n';

    return 0;
}