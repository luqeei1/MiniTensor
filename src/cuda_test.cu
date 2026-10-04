#include "tensor/CudaBuffer.hpp"
#include "tensor/CudaOps.hpp"
#include "tensor/Tensor.hpp"

#include <iostream>

int main()
{
    Tensor a = Tensor::ones({5}).to(Device::CUDA);
    Tensor b = Tensor::ones({5}).to(Device::CUDA);

    Tensor c = a - b;

    Tensor result = c.to(Device::CPU);

    for (std::size_t i = 0; i < result.size(); ++i)
    {
        std::cout << result.at({i}) << " ";
    }

    std::cout << '\n';

    return 0;
}