#include "tensor/CudaBuffer.hpp"
#include "tensor/CudaOps.hpp"
#include "tensor/Tensor.hpp"

#include <iostream>

int main()
{
    // Start with a CPU tensor
    Tensor cpu = Tensor::ones({5});

    std::cout << "Original CPU tensor: ";

    for (std::size_t i = 0; i < cpu.size(); ++i)
    {
        std::cout << cpu.at({i}) << " ";
    }

    std::cout << '\n';

    // CPU -> GPU
    Tensor gpu = cpu.to(Device::CUDA);

    std::cout << "Moved to CUDA: "
              << (gpu.device() == Device::CUDA)
              << '\n';

    // GPU -> CPU
    Tensor cpu_again = gpu.to(Device::CPU);

    std::cout << "Back on CPU: ";

    for (std::size_t i = 0; i < cpu_again.size(); ++i)
    {
        std::cout << cpu_again.at({i}) << " ";
    }

    std::cout << '\n';

    return 0;
}