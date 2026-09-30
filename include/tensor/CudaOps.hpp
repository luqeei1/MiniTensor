#pragma once

#include <cstddef>

void cuda_add(const float* a, const float* b, float* c, std::size_t count);