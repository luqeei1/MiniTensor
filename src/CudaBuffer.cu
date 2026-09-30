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

void CudaBuffer::copy_from_host(const float* src, std::size_t count)
{
    if(count > size_){ throw std::runtime_error("Count exceeds buffer size"); }
    cudaError_t err = cudaMemcpy(ptr_, src, count * sizeof(float), cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
        throw std::runtime_error("Failed to copy data from host to device");
    }
}

void CudaBuffer::copy_to_host(float* dst, std::size_t count) const
{
    if(count > size_) {throw std::runtime_error("Count exceeds buffer size");}
    cudaError_t err = cudaMemcpy(dst, ptr_, count * sizeof(float), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
    {
        throw std::runtime_error("Failed to copy data from device to host");    
    }
}

CudaBuffer::CudaBuffer(CudaBuffer&& other) noexcept // Move constructor which promises not to throw exceptions
:
 ptr_(other.ptr_),
 size_(other.size_)
{
    other.ptr_ = nullptr;
    other.size_ = 0;
}

CudaBuffer& CudaBuffer::operator=(CudaBuffer&& other) noexcept // this is called when destination object is already constructed and we want to move the resources from other to this object
{
    if (this != &other)
    {
        if (ptr_ != nullptr)
        {
            cudaFree(ptr_);
        }
        ptr_ = other.ptr_;
        size_ = other.size_;
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this; 
}