///
/// vecAddKernel00.cu
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
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

    for(i=threadGlobalIndex; i<vecSize; i+=threadCount){
        C[i] = A[i] + B[i];
    }
}