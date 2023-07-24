///
/// vecAddKernel00.cu
/// Based on code from the CUDA Programming Guide
/// This Kernel adds two Vectors A and B in C on GPU
/// using coalesced memory access.
/// 

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int threadGlobalIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int threadCount = gridDim.x * blockDim.x;
    int vecSize = N*threadCount;
    int i;

    // loop so that threads in the same warp access contiguous memory locations
    //  this coalesces memory accesses
    for(i=threadGlobalIndex; i<vecSize; i+=threadCount){
        C[i] = A[i] + B[i];
    }
}
