// Includes
#include <stdio.h>
#include <string>
#include <iostream>
#include "timer.h"
#include "convKernel.h"
#include <cudnn.h>

/*
An input tensor I with dimensions: C, H, W. Each element of I is generated as
follows:
I[c, x, y] = c · (x + y)
Page 4
• A set of convolution filters with dimensions: K, C, FH, FW. Each element of the
filter F is generated as follows:
F[k, c, i, j] = (c + k) · (i + j)
• Dimensions are: H = 1024, W = 1024, C = 3, FW = 3, FH = 3, K = 64.

The output tensor O with dimensions: K,W,H
*/
#define CUDNN_CALL(x) do { \
    cudnnStatus_t ___s = (x); \
    if (___s != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "%s:%d ERROR: %s\n", __FILE__, __LINE__, cudnnGetErrorString(___s)); \
        exit(-1); \
    } \
} while (0)

void reportChecksumAndTime(double checksum, double millseconds) {
    printf("%0.6lf,%0.6lf\n", checksum, millseconds);
}

void checkCUDAError(std::string msg) {
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        std::cerr << "Cuda error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

double seconds2milliseconds(double seconds) {
    return seconds*1000;
}

void fillInput(Tensor tensor) {
    double value;
    for (int c = 0; c < tensor.dims[2]; ++c) {
        for (int y = 0; y < tensor.dims[1]; ++y) {
            for (int x = 0; x < tensor.dims[0]; ++x) {
                value = c*double(x+y);
                setCellValue(tensor, value, x, y, c);
            }
        }
    }
}

void fillFilter(Tensor tensor) {
    double value;
    for (int k = 0; k < tensor.dims[3]; ++k) {
        for (int c = 0; c < tensor.dims[2]; ++c) {
            for (int y = 0; y < tensor.dims[1]; ++y) {
                for (int x = 0; x < tensor.dims[0]; ++x) {
                    value = (c+k)*double(x+y);
                    setCellValue(tensor, value, x, y, c);
                }
            }
        }
    }
}

void fillPaddedInput(Tensor paddedInput, Tensor input, int padding, double padValue) {
    double value;
    for (int c = 0; c < input.dims[2]; ++c) {
        for (int y = 0; y < input.dims[1]; ++y) {
            for (int x = 0; x < input.dims[0]; ++x) {
                value = cellValue(input, x, y, c);
                setCellValue(paddedInput, value, x+padding, y+padding, c);
            }
        }

        for (int y = 0; y < padding; ++y) {
            for (int x = 0; x < padding; ++x) {
                setCellValue(paddedInput, padValue, x, y, c);
                setCellValue(paddedInput, padValue,
                    paddedInput.dims[0] - 1 - x,
                    paddedInput.dims[1] - 1 - y,
                    c
                );
            }
        }
    }
}

void fillOnes(Tensor tensor) {
    for (int c = 0; c < tensor.dims[2]; ++c) {
        for (int y = 0; y < tensor.dims[1]; ++y) {
            for (int x = 0; x < tensor.dims[0]; ++x) {
                setCellValue(tensor, 1.0, x, y, c);
            }
        }
    }
}

void fillFilterOnes(Tensor tensor) {
    for (int k = 0; k < tensor.dims[3]; ++k) {
        for (int c = 0; c < tensor.dims[2]; ++c) {
            for (int y = 0; y < tensor.dims[1]; ++y) {
                for (int x = 0; x < tensor.dims[0]; ++x) {
                    setCellValue(tensor, 1.0, x, y, c, k);
                }
            }
        }
    }
}

double convSimple(Tensor input, Tensor paddedInput, Tensor output, Tensor filters, int padding, bool verbose) {
    Tensor devicePaddedInput = createDeviceTensor(paddedInput, true);
    Tensor deviceOutput = createDeviceTensor(output, false);
    Tensor deviceFilters = createDeviceTensor(filters, true);

    cudaDeviceSynchronize();

    //define dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(deviceOutput.dims[0]/dimBlock.x, deviceOutput.dims[1]/dimBlock.y);

    // warm up
    Conv<<<dimGrid, dimBlock>>>(devicePaddedInput, deviceOutput, deviceFilters, padding);
    cudaDeviceSynchronize();

    // Initialize timer  
    initialize_timer();
    start_timer();

    // simple convolution
    Conv<<<dimGrid, dimBlock>>>(devicePaddedInput, deviceOutput, deviceFilters, padding);
    cudaDeviceSynchronize();

    // Compute and return elapsed time 
    stop_timer();
    double time = elapsed_time();
    
    checkCUDAError("Simple convolutions");

    // copy to host
    size_t size = output.strides[2] * sizeof(double);
    cudaMemcpy(output.elements, deviceOutput.elements, size, cudaMemcpyDeviceToHost);

    cudaFree(devicePaddedInput.elements);
    cudaFree(deviceOutput.elements);
    cudaFree(deviceFilters.elements);

    return time;
}

double convTiled(Tensor input, Tensor paddedInput, Tensor output, Tensor filters, int padding, bool verbose) {

    // create tensors on device
    Tensor devicePaddedInput = createDeviceTensor(paddedInput, true);
    Tensor deviceOutput = createDeviceTensor(output, false);
    Tensor deviceFilters = createDeviceTensor(filters, true);

    cudaDeviceSynchronize();

    //define dimensions
    dim3 dimBlock(256, 1);
    dim3 dimGrid(deviceOutput.dims[0]/dimBlock.x, deviceOutput.dims[1]/dimBlock.y);
    
    // tiled convolution
    int filterElementCount = elementsCount(filters);
    
    int inputBlockSize = (dimBlock.x)+(2*padding);
    int inputElementCount = inputBlockSize*filters.dims[1]*paddedInput.dims[2];

    int buffer = 0; // some headroom for allocation

    size_t sharedMemory = (filterElementCount + inputElementCount + buffer) * sizeof(double);
    if (verbose) {
        printf("Size of Shared Memory: %zu\n\n", sharedMemory);
    }

    // warm up
    ConvTiled<<<dimGrid, dimBlock, sharedMemory>>>(devicePaddedInput, deviceOutput, deviceFilters, padding);
    cudaDeviceSynchronize();
    
    // Initialize timer  
    initialize_timer();
    start_timer();

    ConvTiled<<<dimGrid, dimBlock, sharedMemory>>>(devicePaddedInput, deviceOutput, deviceFilters, padding);
    cudaDeviceSynchronize();

    // Compute and return elapsed time 
    stop_timer();
    double time = elapsed_time();

    checkCUDAError("Tiled convolutions");

    // copy to host
    size_t size = output.strides[2] * sizeof(double);
    cudaMemcpy(output.elements, deviceOutput.elements, size, cudaMemcpyDeviceToHost);

    cudaFree(devicePaddedInput.elements);
    cudaFree(deviceOutput.elements);
    cudaFree(deviceFilters.elements);

    return time;
}

double convCudnn(Tensor input, Tensor paddedInput, Tensor output, Tensor filters, int padding, bool verbose) {
    int stride = 1;
    int dilation = 1;

    // create cudnn context
    cudnnHandle_t cudnn;
    CUDNN_CALL(cudnnCreate(&cudnn));

    // create tensors
    cudnnTensorDescriptor_t inDesc;
    size_t inputSize;
    double * d_inputData;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&inDesc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(inDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 
                1, input.dims[2], input.dims[1], input.dims[0]));

    cudnnTensorDescriptor_t outDesc;
    size_t outputSize;
    double * d_outputData;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&outDesc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(outDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 
                1, output.dims[2], output.dims[1], output.dims[0]));

    cudnnFilterDescriptor_t filterDesc;
    size_t filterSize;
    double * d_filterData;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filterDesc));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW,
                filters.dims[3], filters.dims[2], filters.dims[1], filters.dims[0]));
    

    cudnnConvolutionDescriptor_t convDesc;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&convDesc));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(convDesc,
        padding, padding,
        stride, stride,
        dilation, dilation,
        CUDNN_CONVOLUTION,
        CUDNN_DATA_DOUBLE)
    );

    // find best algo
    int returnedAlgoCount;
    int requestedAlgoCount = 1000;
    cudnnConvolutionFwdAlgoPerf_t perfResults[1000];
    CUDNN_CALL(cudnnFindConvolutionForwardAlgorithm(
        cudnn, inDesc, filterDesc, convDesc, outDesc,
        requestedAlgoCount, &returnedAlgoCount, perfResults)
    );
    cudnnConvolutionFwdAlgo_t algo = perfResults[0].algo;
    if (verbose) {
        std::cout << "Algo: " << algo << std::endl;
    }

    // get workspace size and allocate space for it
    size_t workspaceSizeInBytes;
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
        inDesc, filterDesc, convDesc, outDesc, algo, &workspaceSizeInBytes)
    );

    //allocate workspace on device
    void * workSpace;
    cudaMalloc(&workSpace, workspaceSizeInBytes);

    // allocate device tensors and copy host tensors/filters
    inputSize = sizeInBytes(input);
    cudaMalloc(&d_inputData, inputSize);
    cudaMemcpy(d_inputData, input.elements, inputSize, cudaMemcpyHostToDevice);

    filterSize = sizeInBytes(filters);
    cudaMalloc(&d_filterData, filterSize);
    cudaMemcpy(d_filterData, filters.elements, filterSize, cudaMemcpyHostToDevice);

    outputSize = sizeInBytes(output);
    cudaMalloc(&d_outputData, outputSize);

    // set up scaling parameters
    double alpha[1]{1.0};
    double beta[1]{0.0};

    // warm up
    CUDNN_CALL(cudnnConvolutionForward(cudnn, (void*)alpha,
        inDesc, (void*)d_inputData,
        filterDesc, (void*)d_filterData,
        convDesc, algo,
        workSpace, workspaceSizeInBytes,
        (void*)beta,
        outDesc, (void*)d_outputData)
    );
    cudaDeviceSynchronize();

    // Initialize timer  
    initialize_timer();
    start_timer();

    // call convolution kernel
    cudnnConvolutionForward(cudnn, (void*)alpha,
        inDesc, (void*)d_inputData,
        filterDesc, (void*)d_filterData,
        convDesc, algo,
        workSpace, workspaceSizeInBytes,
        (void*)beta,
        outDesc, (void*)d_outputData);

    cudaDeviceSynchronize();

    // Compute and return elapsed time 
    stop_timer();
    double time = elapsed_time();
    
    cudaMemcpy(output.elements, d_outputData, outputSize, cudaMemcpyDeviceToHost);

    // check results


    // release memory
    cudaFree(workSpace);
    cudaFree(d_inputData);
    cudaFree(d_filterData);
    cudaFree(d_outputData);


    // destroy tensors
    CUDNN_CALL(cudnnDestroyTensorDescriptor(inDesc));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filterDesc));

    // destroy cudnn context
    CUDNN_CALL(cudnnDestroy(cudnn));

    return time;
}

double calculateChecksum(Tensor output) {
    double checksum(0);
    for (int z= 0; z < output.dims[2]; ++z) {
        for (int y = 0; y < output.dims[1]; ++y) {
            for (int x = 0; x < output.dims[0]; ++x) {
                checksum += cellValue(output, x, y, z);
            }
        }
    }
    return checksum;
}

void checkTestResults(Tensor output) {
    // check result
    double value;
    double expectedValue;

    int errors = 0;
    for (int c = 0; c < output.dims[2]; ++c) {
        for (int y = 0; y < output.dims[1]; ++y) {
            for (int x = 0; x < output.dims[0]; ++x) {
                if (x > 0 && y > 0 && x < output.dims[0]-1 && y < output.dims[1]-1) {
                    expectedValue = 27;
                } else if (x > 0 && x < output.dims[0]-1) {
                    expectedValue = 18;
                } else if (y > 0 && y < output.dims[1]-1) {
                    expectedValue = 18;
                } else { // both edges are off input grid
                    expectedValue = 12;
                }

                value = cellValue(output, x, y, c);
                if (fabs(value - expectedValue) > 1e-5) {
                    //printf("Error at (%d, %d, %d).. value %lf, expected %lf\n", x, y, c, value, expectedValue);
                    ++errors;
                }
            }
        }
    }

    // report result
    if (errors != 0) {
        std::cout << "Test FAILED with " << errors << " errors" << std::endl;

        printf("Section of output: ");
        printTensor(output, 3, 3, 3);
    } else {
        std::cout << "Test PASSED" << std::endl;
    }
}

int main(int argc, char ** argv) {
    bool isTestCase = false;
    bool verbose = false;
    std::string mode("all");
    if (argc > 3) {
        mode = std::string(argv[1]);
        isTestCase = std::string("test") == argv[2];
        verbose = std::string("verbose") == argv[3];
    } else if (argc > 2) {
        mode = std::string(argv[1]);
        isTestCase = std::string("test") == argv[2];
    } else if (argc > 1) {
        mode = std::string(argv[1]);
    }

    // tensor specifications
    int padding = 1;
    TensorDescriptor inputDescriptor{.dim=3, .dims={1024, 1024, 3}};
    TensorDescriptor outputDescriptor{.dim=3, .dims={1024, 1024, 64}};
    TensorDescriptor paddedInputDescriptor{.dim=3, 
        .dims={
            inputDescriptor.dims[0]+(padding*2),
            inputDescriptor.dims[1]+(padding*2),
            inputDescriptor.dims[2]
        }};

    const int filterDepth(inputDescriptor.dims[2]);
    const int filterCount(outputDescriptor.dims[2]);
    TensorDescriptor filtersDescriptor{.dim=4, .dims={3, 3, filterDepth, filterCount}};

    if (verbose) {
        printf("\nInput Descriptor: \n");
        printTensorDescriptor(inputDescriptor);
        printf("\nPadded Input Descriptor: \n");
        printTensorDescriptor(paddedInputDescriptor);
        printf("\nOutput Descriptor: \n");
        printTensorDescriptor(outputDescriptor);
        printf("\nFilters Descriptor: \n");
        printTensorDescriptor(filtersDescriptor);

    }

    // create tensors for input, output, and an array of tensors for the filters
    Tensor input = createHostTensor(inputDescriptor);
    Tensor paddedInput = createHostTensor(paddedInputDescriptor);
    Tensor output = createHostTensor(outputDescriptor);
    Tensor filters = createHostTensor(filtersDescriptor);

    // initialize input tensor and filters with values
    if (isTestCase) {
        std::cout << "filling with test case values..." << std::endl;
        fillOnes(input);
        fillFilterOnes(filters);
    } else {
        fillInput(input);
        fillFilter(filters);
    }

    fillPaddedInput(paddedInput, input, padding, 0.0);

    if (verbose) {
        printf("\n\nSection of filter 38: \n");
        printTensor(tensorLayer(filters, 4, 38), 3, 3, 3);

        printf("\n\nSection of input: \n");
        printTensor(input, 3, 3, 3);

        printf("\n\nSection of padded input: \n");
        printTensor(paddedInput, 3, 3, 3);
    }

    // run requested convolutions
    bool ranConv(false);
    if (mode == "simple" || mode == "all") {
        double timeSimple = convSimple(input, paddedInput, output, filters, padding, verbose);
        double checksumSimple = calculateChecksum(output);
        if (isTestCase) {
            printf("Checking test results for Simple Convolutions...\n");
            checkTestResults(output);
        }
        reportChecksumAndTime(checksumSimple, seconds2milliseconds(timeSimple));
        ranConv = true;
    }

    if (mode == "tiled" || mode == "all") {
        double timeTiled = convTiled(input, paddedInput, output, filters, padding, verbose);
        double checksumTiled = calculateChecksum(output);
        if (isTestCase) {
            printf("Checking test results for Tiled Convolutions...\n");
            checkTestResults(output);
        }
        reportChecksumAndTime(checksumTiled, seconds2milliseconds(timeTiled));
        ranConv = true;
    }
        
    if (mode == "cudnn" || mode == "all") {
        double timeCudnn = convCudnn(input, paddedInput, output, filters, padding, verbose);
        double checksumCudnn = calculateChecksum(output);
        if (isTestCase) {
            printf("Checking test results for CuDNN Convolutions...\n");
            checkTestResults(output);
        }
        
        reportChecksumAndTime(checksumCudnn, seconds2milliseconds(timeCudnn));
        ranConv = true;
    }
    
    if (!ranConv) {
        throw std::string("unrecognized mode: " + mode);
    }

    // cleanup
    free(input.elements);
    free(paddedInput.elements);
    free(output.elements);
    free(filters.elements);
}
