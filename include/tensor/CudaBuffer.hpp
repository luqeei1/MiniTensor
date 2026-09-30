#include <cstddef> 

class CudaBuffer
{ 
    public: 
        explicit CudaBuffer(std::size_t size);
        ~CudaBuffer(); 

        CudaBuffer(const CudaBuffer&) = delete; // Delete copy constructor
        CudaBuffer& operator=(const CudaBuffer&) = delete; // Delete copy assignment operator
        CudaBuffer(CudaBuffer&& other) noexcept; // Move constructor which promises not to throw exceptions
        CudaBuffer& operator=(CudaBuffer&& other) noexcept; // Move assignment operator

        void copy_from_host(const float* src, std::size_t count); // copy data from  cpu to gpu
        void copy_to_host(float* dst, std::size_t count) const; // copy data from gpu to cpu
        
        float* data();
        const float* data() const; 

    private:
        float* ptr_;
        std::size_t size_;
    
};
