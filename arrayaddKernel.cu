
__global__ void AddArraysST(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

__global__ void AddArraysMT(const float* A, const float* B, float* C, int N) {
    int threadId = threadIdx.x;
    int threadCount = blockDim.x;

    for (int i = threadId; i < N; i += threadCount) {
        C[i] = A[i] + B[i];
    }
}

__global__ void AddArraysMBMT(const float* A, const float* B, float* C, int N) {
    // assume that this is called with threadCount > N
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadId < N)
        C[threadId] = A[threadId] + B[threadId];
}