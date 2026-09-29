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



