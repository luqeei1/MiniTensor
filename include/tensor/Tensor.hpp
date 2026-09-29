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
        float& at(const std::vector<std::size_t>& indices);
        const float& at(const std::vector<std::size_t>& indices) const;  // const overload for read-only access 

        static Tensor zeros(const std::vector<std::size_t>& shape);
        static Tensor ones(const std::vector<std::size_t>& shape);
        static Tensor full(const std::vector<std::size_t>& shape, float value);

        Tensor operator+(const Tensor& other) const;
        Tensor operator-(const Tensor& other) const;
        Tensor operator*(const Tensor& other) const;
    
    private:
        std::vector<std::size_t> shape_;
        std::vector<std::size_t> strides_;
        std::vector<float> data_;
        

        static std::size_t calculate_size(const std::vector<std::size_t>& shape); 
        static std::vector<std::size_t> calculate_strides(const std::vector<std::size_t>& shape);
        std::size_t calculate_offset(const std::vector<std::size_t>& incdices) const; 

};