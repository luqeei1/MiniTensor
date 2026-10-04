#pragma once

#include <cstddef>

#include <stdexcept>

void cuda_add(const float* a, const float* b, float* c, std::size_t count);
void cuda_multiply(const float* a, const float* b, float*c, std::size_t count);
void cuda_subtract(const float* a, const float* b, float* c, std::size_t count);
void cuda_matmul(const float* a, const float*b, float* c, std::size_t m, std::size_t n, std::size_t k);