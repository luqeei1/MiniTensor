#include "tensor/Tensor.hpp"

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
    std::vector<std::size_t> strides(shape.size());
    strides[shape.size()-1] = 1;
    for(std::size_t i = shape.size() -1; i > 0; i--)
    {
        strides[i-1] = strides[i] * shape[i];
    }

    return strides;
}
