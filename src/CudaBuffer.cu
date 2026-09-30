#include "tensor/CudaBuffer.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

CudaBuffer::CudaBuffer(std::size_t size)
:
 ptr_(nullptr),
 size_(size)
{
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr_), size_ * sizeof(float));
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to allocate device memory");
    }
}

CudaBuffer::~CudaBuffer()
{
    if (ptr_ != nullptr)
    {
        cudaFree(ptr_);
    }
}

float* CudaBuffer::data()
{
    return ptr_;
}

const float* CudaBuffer::data() const
{
    return ptr_;
}