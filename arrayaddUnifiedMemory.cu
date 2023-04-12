
// Includes
#include <stdio.h>
#include <string>
#include <iostream>
#include "timer.h"
#include "arrayaddKernel.h"

// Defines
#define epsilon (float)1e-4
#define verbose 0

const std::string defaultCompute("host");
const std::string defaultThreading("st");

// variables for host and device vectors
float* A; 
float* B; 
float* C;

void cleanup(bool noError) {
    cudaError_t error;
        
    // Free device vectors
    if (A)
        cudaFree(A);
    if (B)
        cudaFree(B);
    if (C)
        cudaFree(C);
        
    error = cudaDeviceReset();
    
    if (!noError) {
        std::cerr << "cleanup called after error" << std::endl;
    } else if (error != cudaSuccess) {
        std::cerr << "cuda error during cleanup cudaDeviceReset: " << cudaGetErrorString(error) << std::endl;
    }
    
    fflush( stdout);
    fflush( stderr);
}


void checkCUDAError(std::string msg) {
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        std::cerr << "Cuda error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        cleanup(false);
        exit(1);
    }
}

double deviceAddArrays(float * dest, float * srcA, float * srcB, int N, std::string threading) {
    // variables for device vectors
    double time;
    size_t size = N * sizeof(float);

    // set up dimensions for grid and block
    if (threading == "st") {
        dim3 dimBlock(1);
        dim3 dimGrid(1);
        
        // call kernel
        // warm up
        AddArraysST<<<dimGrid, dimBlock>>>(A, B, C, N);
        checkCUDAError("AddArrays warmup");
        cudaDeviceSynchronize();

        // Initialize timer  
        initialize_timer();
        start_timer();

        // Invoke kernel
        AddArraysST<<<dimGrid, dimBlock>>>(A, B, C, N);
        checkCUDAError("AddArrays trial");
        cudaDeviceSynchronize();

        // Compute and return elapsed time 
        stop_timer();
        time = elapsed_time();

    } else if (threading == "mt") {
        dim3 dimBlock(256);
        dim3 dimGrid(1);
        
        // call kernel
        // warm up
        AddArraysMT<<<dimGrid, dimBlock>>>(A, B, C, N);
        checkCUDAError("AddArrays warmup");
        cudaDeviceSynchronize();

        // Initialize timer  
        initialize_timer();
        start_timer();

        // Invoke kernel
        AddArraysMT<<<dimGrid, dimBlock>>>(A, B, C, N);
        checkCUDAError("AddArrays trial");
        cudaDeviceSynchronize();

        // Compute and return elapsed time 
        stop_timer();
        time = elapsed_time();
    } else {
        int blockWidth = 256;
        dim3 dimBlock(blockWidth);
        dim3 dimGrid((int)ceil(float(N) / blockWidth));
        
        // call kernel
        // warm up
        AddArraysMBMT<<<dimGrid, dimBlock>>>(A, B, C, N);
        checkCUDAError("AddArrays warmup");
        cudaDeviceSynchronize();

        // Initialize timer  
        initialize_timer();
        start_timer();

        // Invoke kernel
        AddArraysMBMT<<<dimGrid, dimBlock>>>(A, B, C, N);
        checkCUDAError("AddArrays trial");
        cudaDeviceSynchronize();

        // Compute and return elapsed time 
        stop_timer();
        time = elapsed_time();
    }

    // return elapsed time
    return time;
}

double hostAddArrays(float * dest, float * srcA, float * srcB, int N) {
    // Initialize timer  
    initialize_timer();
    start_timer();

    for (int i = 0; i < N; ++i) {
        dest[i] = srcA[i] + srcB[i];
    }

    // Compute and return elapsed time 
    stop_timer();
    return elapsed_time();
}

int main(int argc, char** argv) {
    // extract args
    // N - the number of elements per array, in millions; defaults to 1
    // [host|device] - whether to use the host or device to calculate; defaults to host
    // [st|mt|mbmt] - "single thread", "multi thread", or "multi-block multi thread (saturated)"; defaults to "st"

    int N;
    size_t size;
    std::string compute, threading;

    // Variables for host vectors.
    float expected;

    // extract args
    N       = (argc < 2 ? 1 : std::stoi(argv[1])) * 1e6;
    compute = (argc < 3 ? defaultCompute : argv[2]);
    threading    = (argc < 4 ? defaultThreading : argv[3]); 

    std::cout << "Parameters:" << std::endl;
    std::cout << "N: " << N << std::endl;
    std::cout << "Compute: " << compute << std::endl;
    std::cout << "Threading: " << (compute == "device" ? threading : "N/A") << std::endl;

    int nFlops(N), nBytes(3*sizeof(float)*N);
    double time, nFlopsPerSec, nGFlopsPerSec, nBytesPerSec, nGBytesPerSec;
    int errors(0);

    // allocate and initialize arrays
    // allocate and initialize arrays
    size = N * sizeof(float);
    if (compute == "device") {
        // allocate vectors in unified memory
        cudaMallocManaged((void**)&A, size);
        checkCUDAError("makeArrayOnDevice cudaMalloc");
        cudaMallocManaged((void**)&B, size);
        checkCUDAError("makeArrayOnDevice cudaMalloc");
        cudaMallocManaged((void**)&C, size);
        checkCUDAError("makeArrayOnDevice cudaMalloc");   
    } else {
        A = (float*) malloc(size);
        if (A == 0) cleanup(false);
        B = (float*) malloc(size);
        if (B == 0) cleanup(false);
        C = (float*) malloc(size);
        if (C == 0) cleanup(false);
    }

    expected = 3.0f; // to validate the results after adding
    for (int i = 0; i < N; ++i) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // add arrays
    if (compute == "host") {
        time = hostAddArrays(C, A, B, N);
    } else {
        time = deviceAddArrays(C, A, B, N, threading);
    }

    // Compute floating point operations per second.
    nFlopsPerSec = nFlops/time;
    nGFlopsPerSec = nFlopsPerSec*1e-9;

    // Compute transfer rates.
    nBytesPerSec = nBytes/time;
    nGBytesPerSec = nBytesPerSec*1e-9;

    // Report timing data.
    printf("Time: %lf (sec), GFlopsS: %lf, GBytesS: %lf\n", 
        time, nGFlopsPerSec, nGBytesPerSec);
    
    // check result in h_C and report
    errors = 0;
    for (int i = 0; i < N; ++i) {
        if (fabs(C[i] - expected) > 1e-5)
            ++errors;
    }

    if (errors > 0) {
        std::cout << "Test FAILED with " << errors << " errors" << std::endl;
    } else {
        std::cout << "Test PASSED" << std::endl;
    }

    // cleanup
    cleanup(true);
}