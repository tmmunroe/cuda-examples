
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

void deviceAddArrays(float * dest, float * srcA, float * srcB, int N) {
    // variables for device vectors
    float* d_A; 
    float* d_B; 
    float* d_C;

    // allocate arrays on device

    // copy arrays to device

    // call kernel

    // copy dest back to host

    // free arrays on device
    if (d_A) free(d_A);
    if (d_B) free(d_B);
    if (d_C) free(d_C);

    // report any errors
}

void hostAddArrays(float * dest, float * srcA, float * srcB, int N) {
    std::cout << "Adding arrays on host..." << std::endl;
    for (int i = 0; i < N; ++i) {
        dest[i] = srcA[i] + srcB[i];
    }
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
    float* h_A; 
    float* h_B; 
    float* h_C;
    float expected;

    // extract args
    N       = (argc < 2 ? 1 : std::stoi(argv[1])) * 1e6;
    compute = (argc < 3 ? defaultCompute : argv[2]);
    threading    = (argc < 4 ? defaultThreading : argv[3]); 

    std::cout << "Parameters:" << std::endl;
    std::cout << "N: " << N << std::endl;
    std::cout << "Compute: " << compute << std::endl;
    std::cout << "Threading: " << threading << std::endl;

    // allocate and initialize arrays
    size = N * sizeof(float);

    h_A = (float*) malloc(size);
    h_B = (float*) malloc(size);
    h_C = (float*) malloc(size);

    expected = 3.0f; // to validate the results after adding
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    try {
        int nFlops(N), nBytes(3*sizeof(float)*N);
        double time, nFlopsPerSec, nGFlopsPerSec, nBytesPerSec, nGBytesPerSec;
        int errors(0);

        // Initialize timer  
        initialize_timer();
        start_timer();

        // add arrays
        if (compute == "host") {
            hostAddArrays(h_C, h_A, h_B, N);
        } else {
            deviceAddArrays(h_C, h_A, h_B, N);
        }

        // Compute elapsed time 
        stop_timer();
        time = elapsed_time();

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
    } catch (const std::exception &ex) {
        std::cout << ex.what() << std::endl;
    }

    // free arrays
    if (h_A) free(h_A);
    if (h_B) free(h_B);
    if (h_C) free(h_C);
}