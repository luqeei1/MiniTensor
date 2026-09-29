#include "tensor/Tensor.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

int main()
{

    // 1. Construction, size, dimensions, shape and strides
    Tensor tensor({2, 3, 4});

    assert(tensor.size() == 24);
    assert(tensor.ndim() == 3);

    assert((tensor.shape() == std::vector<std::size_t>{2, 3, 4}));
    assert((tensor.strides() == std::vector<std::size_t>{12, 4, 1}));

    std::cout << "[PASS] Construction and metadata\n";

    // 2. Element access and modification
    tensor.at({1, 2, 3}) = 42.0f;

    assert(tensor.at({1, 2, 3}) == 42.0f);

    // Check that a different element was not changed.
    assert(tensor.at({0, 0, 0}) == 0.0f);

    std::cout << "[PASS] Element indexing\n";


    // 3. Const element access
    const Tensor const_tensor = Tensor::full({2, 2}, 7.0f);

    assert(const_tensor.at({0, 0}) == 7.0f);
    assert(const_tensor.at({1, 1}) == 7.0f);

    std::cout << "[PASS] Const element access\n";


    // 4. zeros()
    Tensor zeros = Tensor::zeros({2, 3});

    assert(zeros.size() == 6);

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            assert(zeros.at({i, j}) == 0.0f);
        }
    }

    std::cout << "[PASS] zeros()\n";


    // 5. ones()
    Tensor ones = Tensor::ones({2, 3});

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            assert(ones.at({i, j}) == 1.0f);
        }
    }

    std::cout << "[PASS] ones()\n";


    // 6. full()
    Tensor full = Tensor::full({2, 3}, 5.0f);

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            assert(full.at({i, j}) == 5.0f);
        }
    }

    std::cout << "[PASS] full()\n";


    // 7. Element-wise addition
    Tensor a = Tensor::ones({2, 3});
    Tensor b = Tensor::full({2, 3}, 2.0f);

    Tensor sum = a + b;

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            assert(sum.at({i, j}) == 3.0f);
        }
    }

    std::cout << "[PASS] Addition\n";


    // 8. Element-wise subtraction
    Tensor difference = b - a;

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            assert(difference.at({i, j}) == 1.0f);
        }
    }

    std::cout << "[PASS] Subtraction\n";


    // 9. Element-wise multiplication
    Tensor product = a * b;

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            assert(product.at({i, j}) == 2.0f);
        }
    }

    std::cout << "[PASS] Multiplication\n";


    // 10. Out-of-bounds index
    bool out_of_bounds_thrown = false;

    try
    {
        tensor.at({2, 0, 0});
    }
    catch (const std::out_of_range&)
    {
        out_of_bounds_thrown = true;
    }

    assert(out_of_bounds_thrown);

    std::cout << "[PASS] Out-of-bounds detection\n";

    // 11. Incorrect number of indices
    bool incorrect_dimensions_thrown = false;

    try
    {
        tensor.at({1, 2});
    }
    catch (const std::out_of_range&)
    {
        incorrect_dimensions_thrown = true;
    }

    assert(incorrect_dimensions_thrown);

    std::cout << "[PASS] Incorrect index dimensions detection\n";

    // 12. Mismatched shapes for arithmetic
    bool shape_mismatch_thrown = false;

    try
    {
        Tensor x = Tensor::ones({2, 3});
        Tensor y = Tensor::ones({3, 2});

        Tensor z = x + y;
    }
    catch (const std::runtime_error&)
    {
        shape_mismatch_thrown = true;
    }

    assert(shape_mismatch_thrown);

    std::cout << "[PASS] Shape mismatch detection\n";


    std::cout << "\nAll MiniTensor tests passed.\n";

    return 0;
}