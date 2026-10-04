#include "tensor/CudaOps.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

__global__ void add_kernel(const float* a, const float* b, float* c, std::size_t count) //this is a CUDA kernel function that runs on the GPU, launched by the cpu
{ // a, b, and c are pointers to arrays of floats, and count is the number of elements in these arrays

    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the global thread index
    if (idx < count) // Ensure we don't go out of bounds
    {
        c[idx] = a[idx] + b[idx]; // Perform the addition. C here is the output array where the result of adding a and b is stored
    }
}

__global__ void multiply_kernel(const float* a, const float* b, float* c, std::size_t count) 
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) 
    {
        c[idx] = a[idx] * b[idx]; 
    }
}

__global__ void subtract_kernel(const float* a, const float* b, float* c, std::size_t count) 
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) 
    {
        c[idx] = a[idx] - b[idx]; 
    }
}

__global__ void matmul_kernel(const float* a, const float* b, float* c, std::size_t m, std::size_t n, std::size_t k)
{
    std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < k)
    {
        float sum = 0.0f;
        for (std::size_t i = 0; i < n; ++i)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

void cuda_add( const float* a, const float* b, float* c, std::size_t count)
{
    std::size_t threads_per_block = 256; // Number of threads per block
    std::size_t number_of_blocks = (count + threads_per_block - 1) / threads_per_block; // Calculate the number of blocks needed. equiv to ceil(count / threads_per_block)
    add_kernel<<<number_of_blocks, threads_per_block>>>(a,b,c,count); // Launch the kernel on the GPU with the specified number of blocks and threads per block

    cudaError_t err = cudaGetLastError(); // Check for any errors during kernel launch
    if (err != cudaSuccess) // If there was an error
    {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err))); // Throw a runtime error with the error message
    }
    
}

void cuda_multiply(const float* a, const float* b, float* c, std::size_t count)
{
    std::size_t threads_per_block = 256; 
    std::size_t number_of_blocks = (count + threads_per_block - 1) / threads_per_block;
    multiply_kernel<<<number_of_blocks, threads_per_block>>>(a,b,c,count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
}

void cuda_subtract(const float* a, const float* b, float* c, std::size_t count)
{
    std::size_t threads_per_block = 256; 
    std::size_t number_of_blocks = (count + threads_per_block - 1) / threads_per_block;
    subtract_kernel<<<number_of_blocks, threads_per_block>>>(a,b,c,count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
}

void cuda_matmul(const float* a, const float* b, float* c, std::size_t m, std::size_t n, std::size_t k)
{
    dim3 threads_per_block(16, 16); // 16x16 threads per block
    dim3 number_of_blocks((k + threads_per_block.x - 1) / threads_per_block.x,
                          (m + threads_per_block.y - 1) / threads_per_block.y); // Calculate the number of blocks needed in each dimension
    matmul_kernel<<<number_of_blocks, threads_per_block>>>(a,b,c,m,n,k);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
}




