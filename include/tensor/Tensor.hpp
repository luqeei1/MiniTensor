#pragma once 

#include <cstddef>
#include <vector> 

class Tensor 
{
    public: 
        explicit Tensor(const std::vector<std::size_t>& shape);
        std::size_t size() const; 
        std::size_t ndim() const; 

        const std::vector<std::size_t> &shape() const; 
        const std::vector<std::size_t> &strides() const; 
    
    private:
        std::vector<std::size_t> shape_;
        std::vector<std::size_t> strides_;
        std::vector<float> data_;
        

        static std::size_t calculate_size(const std::vector<std::size_t>& shape); 
        static std::vector<std::size_t> calculate_strides(const std::vector<std::size_t>& shape);

};