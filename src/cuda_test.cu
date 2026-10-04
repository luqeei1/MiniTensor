#include "tensor/CudaBuffer.hpp"
#include "tensor/CudaOps.hpp"
#include "tensor/Tensor.hpp"

#include <iostream>

int main()
{
    Tensor a = Tensor::full({2, 3}, 1.0f).to(Device::CUDA);
    Tensor b = Tensor::full({3, 2}, 2.0f).to(Device::CUDA);

    Tensor c = a.matmul(b).to(Device::CPU);

    for (std::size_t i = 0; i < c.shape()[0]; ++i)
    {
        for (std::size_t j = 0; j < c.shape()[1]; ++j)
        {
            std::cout << c.at({i, j}) << " ";
        }

        std::cout << '\n';
    }
}