#pragma once 

#include <cstddef>
#include <vector> 
#include "CudaBuffer.hpp"
#include <memory>

enum class Device
{
    CPU,
    CUDA
};

class Tensor 
{
    public: 
        explicit Tensor(const std::vector<std::size_t>& shape, Device device = Device::CPU);
        std::size_t size() const; 
        std::size_t ndim() const; 

        const std::vector<std::size_t> &shape() const; 
        const std::vector<std::size_t> &strides() const; 
        Device device() const;
        Tensor to(Device device) const;

        float& at(const std::vector<std::size_t>& indices);
        const float& at(const std::vector<std::size_t>& indices) const;  // const overload for read-only access 

        static Tensor zeros(const std::vector<std::size_t>& shape);
        static Tensor ones(const std::vector<std::size_t>& shape);
        static Tensor full(const std::vector<std::size_t>& shape, float value);

        Tensor operator+(const Tensor& other) const;
        Tensor operator-(const Tensor& other) const;
        Tensor operator*(const Tensor& other) const;

        Tensor reshape(const std::vector<std::size_t>& new_shape) const;
        Tensor transpose() const; 
        Tensor matmul(const Tensor& other) const; 


        
    private:
        std::vector<std::size_t> shape_;
        std::vector<std::size_t> strides_;
        std::vector<float> data_;
        Device device_;
        std::unique_ptr<CudaBuffer> cuda_data_; // Pointer to CudaBuffer for GPU storage
        

        static std::size_t calculate_size(const std::vector<std::size_t>& shape); 
        static std::vector<std::size_t> calculate_strides(const std::vector<std::size_t>& shape);
        std::size_t calculate_offset(const std::vector<std::size_t>& indices) const; 

};