#include "tensor/Tensor.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>


TEST(TensorConstruction, StoresCorrectMetadata)
{
    Tensor tensor({2, 3, 4});

    EXPECT_EQ(tensor.size(), 24);
    EXPECT_EQ(tensor.ndim(), 3);

    EXPECT_EQ(
        tensor.shape(),
        (std::vector<std::size_t>{2, 3, 4})
    );

    EXPECT_EQ(
        tensor.strides(),
        (std::vector<std::size_t>{12, 4, 1})
    );
}


TEST(TensorIndexing, AllowsElementModification)
{
    Tensor tensor({2, 3, 4});

    tensor.at({1, 2, 3}) = 42.0f;

    EXPECT_FLOAT_EQ(
        tensor.at({1, 2, 3}),
        42.0f
    );

    EXPECT_FLOAT_EQ(
        tensor.at({0, 0, 0}),
        0.0f
    );
}


TEST(TensorIndexing, SupportsConstElementAccess)
{
    const Tensor tensor = Tensor::full({2, 2}, 7.0f);

    EXPECT_FLOAT_EQ(
        tensor.at({0, 0}),
        7.0f
    );

    EXPECT_FLOAT_EQ(
        tensor.at({1, 1}),
        7.0f
    );
}


TEST(TensorFactories, CreatesZerosTensor)
{
    Tensor tensor = Tensor::zeros({2, 3});

    EXPECT_EQ(tensor.size(), 6);

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            EXPECT_FLOAT_EQ(
                tensor.at({i, j}),
                0.0f
            );
        }
    }
}


TEST(TensorFactories, CreatesOnesTensor)
{
    Tensor tensor = Tensor::ones({2, 3});

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            EXPECT_FLOAT_EQ(
                tensor.at({i, j}),
                1.0f
            );
        }
    }
}


TEST(TensorFactories, CreatesFullTensor)
{
    Tensor tensor = Tensor::full({2, 3}, 5.0f);

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            EXPECT_FLOAT_EQ(
                tensor.at({i, j}),
                5.0f
            );
        }
    }
}


TEST(TensorArithmetic, AddsElementWise)
{
    Tensor a = Tensor::ones({2, 3});
    Tensor b = Tensor::full({2, 3}, 2.0f);

    Tensor result = a + b;

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            EXPECT_FLOAT_EQ(
                result.at({i, j}),
                3.0f
            );
        }
    }
}


TEST(TensorArithmetic, SubtractsElementWise)
{
    Tensor a = Tensor::ones({2, 3});
    Tensor b = Tensor::full({2, 3}, 2.0f);

    Tensor result = b - a;

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            EXPECT_FLOAT_EQ(
                result.at({i, j}),
                1.0f
            );
        }
    }
}


TEST(TensorArithmetic, MultipliesElementWise)
{
    Tensor a = Tensor::ones({2, 3});
    Tensor b = Tensor::full({2, 3}, 2.0f);

    Tensor result = a * b;

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            EXPECT_FLOAT_EQ(
                result.at({i, j}),
                2.0f
            );
        }
    }
}


TEST(TensorIndexing, ThrowsForOutOfBoundsIndex)
{
    Tensor tensor({2, 3, 4});

    EXPECT_THROW(
        tensor.at({2, 0, 0}),
        std::out_of_range
    );
}


TEST(TensorIndexing, ThrowsForIncorrectNumberOfIndices)
{
    Tensor tensor({2, 3, 4});

    EXPECT_THROW(
        tensor.at({1, 2}),
        std::out_of_range
    );
}


TEST(TensorArithmetic, ThrowsForShapeMismatch)
{
    Tensor a = Tensor::ones({2, 3});
    Tensor b = Tensor::ones({3, 2});

    EXPECT_THROW(
        a + b,
        std::runtime_error
    );

    EXPECT_THROW(
        a - b,
        std::runtime_error
    );

    EXPECT_THROW(
        a * b,
        std::runtime_error
    );
}

TEST(TensorShape, Transposes2DTensor)
{
    Tensor tensor({2, 3});

    tensor.at({0, 0}) = 1.0f;
    tensor.at({0, 1}) = 2.0f;
    tensor.at({0, 2}) = 3.0f;
    tensor.at({1, 0}) = 4.0f;
    tensor.at({1, 1}) = 5.0f;
    tensor.at({1, 2}) = 6.0f;

    Tensor transposed = tensor.transpose();

    EXPECT_EQ(
        transposed.shape(),
        (std::vector<std::size_t>{3, 2})
    );

    EXPECT_EQ(
        transposed.strides(),
        (std::vector<std::size_t>{2, 1})
    );

    EXPECT_FLOAT_EQ(transposed.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(transposed.at({0, 1}), 4.0f);

    EXPECT_FLOAT_EQ(transposed.at({1, 0}), 2.0f);
    EXPECT_FLOAT_EQ(transposed.at({1, 1}), 5.0f);

    EXPECT_FLOAT_EQ(transposed.at({2, 0}), 3.0f);
    EXPECT_FLOAT_EQ(transposed.at({2, 1}), 6.0f);
}

TEST(TensorShape, ThrowsWhenTransposingNon2DTensor)
{
    Tensor tensor({2, 3, 4});

    EXPECT_THROW(
        tensor.transpose(),
        std::runtime_error
    );
}

TEST(TensorMatmul, MultipliesMatricesCorrectly)
{
    Tensor a({2, 3});
    Tensor b({3, 2});

    a.at({0, 0}) = 1.0f;
    a.at({0, 1}) = 2.0f;
    a.at({0, 2}) = 3.0f;
    a.at({1, 0}) = 4.0f;
    a.at({1, 1}) = 5.0f;
    a.at({1, 2}) = 6.0f;

    b.at({0, 0}) = 7.0f;
    b.at({0, 1}) = 8.0f;
    b.at({1, 0}) = 9.0f;
    b.at({1, 1}) = 10.0f;
    b.at({2, 0}) = 11.0f;
    b.at({2, 1}) = 12.0f;

    Tensor result = a.matmul(b);

    EXPECT_EQ(
        result.shape(),
        (std::vector<std::size_t>{2, 2})
    );

    EXPECT_FLOAT_EQ(result.at({0, 0}), 58.0f);
    EXPECT_FLOAT_EQ(result.at({0, 1}), 64.0f);
    EXPECT_FLOAT_EQ(result.at({1, 0}), 139.0f);
    EXPECT_FLOAT_EQ(result.at({1, 1}), 154.0f);
}

TEST(TensorMatmul, ThrowsForIncompatibleShapes)
{
    Tensor a({2, 3});
    Tensor b({4, 2});

    EXPECT_THROW(
        a.matmul(b),
        std::invalid_argument
    );
}

TEST(TensorMatmul, ThrowsForNon2DTensors)
{
    Tensor a({2, 3, 4});
    Tensor b({4, 2});

    EXPECT_THROW(
        a.matmul(b),
        std::invalid_argument
    );
}