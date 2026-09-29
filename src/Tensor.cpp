#include "tensor/Tensor.hpp"
#include <stdexcept>
#include <string>

Tensor::Tensor(const std::vector<std::size_t>& shape): 
shape_(shape),
strides_(calculate_strides(shape_)),
data_(calculate_size(shape_)) 
{
}

std::size_t Tensor::calculate_size(const std::vector<std::size_t>& shape) 
{
    std::size_t size = 1;
    for (const std::size_t dim : shape) {
        size *= dim;
    }
    return size;
}

std::vector<std::size_t> Tensor::calculate_strides(const std::vector<std::size_t>& shape)
{
    if(shape.empty())
    {
        return {};
    }

    std::vector<std::size_t> strides(shape.size());
    strides[shape.size()-1] = 1;
    for(std::size_t i = shape.size() -1; i > 0; --i)
    {
        strides[i-1] = strides[i] * shape[i];
    }

    return strides;
}

std::size_t Tensor::calculate_offset(const std::vector<std::size_t>& indices) const
{
    if(indices.size() != shape_.size()) { throw std::out_of_range("Tensor index out of bounds"); }
    std::size_t offset = 0;
    for(std::size_t i = 0; i < indices.size(); ++i)
    {
        if(indices[i] >= shape_[i]) {throw std::out_of_range("Index is out of bounds");}
        offset += indices[i] * strides_[i];    
    }
    return offset; 
}

std::size_t Tensor::size() const
{
    return data_.size();    
} 

std::size_t Tensor::ndim() const
{
    return shape_.size();
}

const std::vector<std::size_t>& Tensor::shape() const
{
    return shape_;
} 
const std::vector<std::size_t>& Tensor::strides() const
{
    return strides_;
}

float& Tensor::at(const std::vector<std::size_t>& indices)
{
    std::size_t offset = calculate_offset(indices);
    return data_[offset];
}

const float& Tensor::at(const std::vector<std::size_t>& indices) const 
{
    std::size_t offset = calculate_offset(indices);
    return data_[offset];
}

Tensor Tensor::zeros(const std::vector<std::size_t>& shape)
{
    return Tensor(shape);
}

Tensor Tensor::full(const std::vector<std::size_t>& shape, float value)
{
    Tensor tensor(shape);
    for(float& element : tensor.data_)
    {
        element = value;
    }
    return tensor;
}
Tensor Tensor::ones(const std::vector<std::size_t>& shape)
{
    return full(shape, 1.0f);
}

Tensor Tensor::operator+(const Tensor& other) const
{
    if (shape_ != other.shape_) {throw std::runtime_error("Shapes do not match so cannot perform addition");}
    Tensor result(shape_);
    for (std::size_t i = 0; i < data_.size(); ++i)
    {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const
{
    if( shape_ != other.shape_ ) { throw std::runtime_error("Shapes do not match so cannot perform subtraction");}
    Tensor result(shape_);
    for( std::size_t i = 0; i < data_.size(); ++i)
    {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const
{
    if( shape_ != other.shape_ ) { throw std::runtime_error("Shapes do not match so cannot perform multiplication");}
    Tensor result(shape_);
    for( std::size_t i = 0; i < data_.size(); ++i)
    {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

Tensor Tensor::reshape(const std::vector<std::size_t>& new_shape) const
{
    std::size_t new_size = calculate_size(new_shape);
    if (new_size != size()) {throw std::runtime_error("New shape does not match the total number of elements in the tensor");}
    Tensor result(new_shape); 
    result.data_ = data_; 
    return result;
}

Tensor Tensor::transpose() const
{
    if (ndim() != 2) {throw std::runtime_error("Transpose is only implemented for 2D tensors");}
    std::vector<std::size_t> new_shape = {shape_[1], shape_[0]};
    Tensor result(new_shape);
    for (std::size_t i = 0; i < shape_[0]; ++i)
    {
        for (std::size_t j = 0; j < shape_[1]; ++j)
        {
            result.at({j,i}) = at({i,j});
        }
    }
    return result; 
}

Tensor Tensor::matmul(const Tensor& other) const 
{
    if (ndim() != 2 || other.ndim() != 2) {throw std::invalid_argument("Matrix multiplication is only implemented for 2D tensors");}
    if (shape_[1] != other.shape_[0]) {throw std::invalid_argument("Inner dimensions do not match for matrix multiplication");}
    
    std::vector<std::size_t> new_shape = {shape_[0], other.shape_[1]};
    Tensor result(new_shape);
    
    for (std::size_t i = 0; i < shape_[0]; ++i)
    {
        for (std::size_t j = 0; j < other.shape_[1]; ++j)
        {
            float sum = 0.0f;
            for (std::size_t k = 0; k < shape_[1]; ++k)
            {
                sum += at({i, k}) * other.at({k, j});
            }
            result.at({i, j}) = sum;
        }
    }
    
    return result;
}



