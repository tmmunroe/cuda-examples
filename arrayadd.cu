
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
float* h_A; 
float* h_B; 
float* h_C;
float* d_A; 
float* d_B; 
float* d_C;

void cleanup(bool noError) {
    cudaError_t error;
        
    // Free device vectors
    if (d_A)
        cudaFree(d_A);
    if (d_B)
        cudaFree(d_B);
    if (d_C)
        cudaFree(d_C);

    // Free host memory
    if (h_A)
        free(h_A);
    if (h_B)
        free(h_B);
    if (h_C)
        free(h_C);
        
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

void makeArrayOnDevice(float **deviceArray, float *hostArray, size_t size) {
    // allocate array
    cudaMalloc((void**)deviceArray, size);
    checkCUDAError("makeArrayOnDevice cudaMalloc");

    if (hostArray) {
        cudaMemcpy(*deviceArray, hostArray, size, cudaMemcpyHostToDevice);
        checkCUDAError("makeArrayOnDevice cudaMemcpy");
    }
}

double deviceAddArrays(float * dest, float * srcA, float * srcB, int N, std::string threading) {
    // variables for device vectors
    double time;
    size_t size = N * sizeof(float);

    // allocate vectors in device memory and do any copies from host to device
    makeArrayOnDevice(&d_A, h_A, size);
    makeArrayOnDevice(&d_B, h_B, size);
    makeArrayOnDevice(&d_C, nullptr, size);
    
    // set up dimensions for grid and block
    if (threading == "st") {
        dim3 dimBlock(1);
        dim3 dimGrid(1);
        
        // call kernel
        // warm up
        AddArraysST<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
        checkCUDAError("AddArrays warmup");
        cudaDeviceSynchronize();

        // Initialize timer  
        initialize_timer();
        start_timer();

        // Invoke kernel
        AddArraysST<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
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
        AddArraysMT<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
        checkCUDAError("AddArrays warmup");
        cudaDeviceSynchronize();

        // Initialize timer  
        initialize_timer();
        start_timer();

        // Invoke kernel
        AddArraysMT<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
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
        AddArraysMBMT<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
        checkCUDAError("AddArrays warmup");
        cudaDeviceSynchronize();

        // Initialize timer  
        initialize_timer();
        start_timer();

        // Invoke kernel
        AddArraysMBMT<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
        checkCUDAError("AddArrays trial");
        cudaDeviceSynchronize();

        // Compute and return elapsed time 
        stop_timer();
        time = elapsed_time();
    }

    // copy device back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    checkCUDAError("cudaMemcpy deviceToHost");

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

    // allocate and initialize arrays
    size = N * sizeof(float);

    h_A = (float*) malloc(size);
    if (h_A == 0) cleanup(false);
    h_B = (float*) malloc(size);
    if (h_B == 0) cleanup(false);
    h_C = (float*) malloc(size);
    if (h_C == 0) cleanup(false);

    expected = 3.0f; // to validate the results after adding
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    int nFlops(N), nBytes(3*sizeof(float)*N);
    double time, nFlopsPerSec, nGFlopsPerSec, nBytesPerSec, nGBytesPerSec;
    int errors(0);

    // add arrays
    if (compute == "host") {
        time = hostAddArrays(h_C, h_A, h_B, N);
    } else {
        time = deviceAddArrays(h_C, h_A, h_B, N, threading);
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
        if (fabs(h_C[i] - expected) > 1e-5)
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