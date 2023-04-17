// Includes
#include <stdio.h>
#include <string>
#include <iostream>
#include "timer.h"
#include "convKernel.h"

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
    std::string mode("simple");
    if (argc > 3) {
        isTestCase = std::string("test") == argv[1];
        verbose = std::string("verbose") == argv[2];
        mode = std::string(argv[3]);
    } else if (argc > 2) {
        isTestCase = std::string("test") == argv[1];
        verbose = std::string("verbose") == argv[2];
    } else if (argc > 1) {
        isTestCase = std::string("test") == argv[1];
    }

    double time;

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

    printf("Created all tensors\n");
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
        printf("\n\nSection of filter: \n");
        printTensor(tensorLayer(filters, 4, 2), 3, 3, 3);

        printf("\n\nSection of input: \n");
        printTensor(input, 3, 3, 3);

        printf("\n\nSection of padded input: \n");
        printTensor(paddedInput, 3, 3, 3);
    }

    // create tensors on device
    Tensor devicePaddedInput = createDeviceTensor(paddedInput, true);
    Tensor deviceOutput = createDeviceTensor(output, false);
    Tensor deviceFilters = createDeviceTensor(filters, true);

    //define dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(deviceOutput.dims[0]/BLOCK_SIZE, deviceOutput.dims[1]/BLOCK_SIZE);
    cudaDeviceSynchronize();

    // Initialize timer  
    initialize_timer();
    start_timer();

    if (mode == "simple") { 
        // simple convolution
        Conv<<<dimGrid, dimBlock>>>(devicePaddedInput, deviceOutput, deviceFilters, padding);
    } else if (mode == "tiled") {
        // tiled convolution
        int filterElementCount = elementsCount(filters);
        
        int inputBlockSize = BLOCK_SIZE+(2*padding);
        int inputElementCount = inputBlockSize*inputBlockSize*paddedInput.dims[2];

        int buffer = 64; // some headroom for allocation

        int sharedMemory = (filterElementCount + inputElementCount + buffer) * sizeof(double);
        ConvTiled<<<dimGrid, dimBlock, sharedMemory>>>(devicePaddedInput, deviceOutput, deviceFilters, padding);
    } else {
        throw std::string("unrecognized mode: " + mode);
    }

    cudaDeviceSynchronize();

    // Compute and return elapsed time 
    stop_timer();
    time = elapsed_time();

    // copy to host
    size_t size = output.strides[2] * sizeof(double);
    cudaMemcpy(output.elements, deviceOutput.elements, size, cudaMemcpyDeviceToHost);

    // check result
    if (isTestCase) {
        printf("Checking test results...\n");
        checkTestResults(output);
    } else {
        double checksum = calculateChecksum(output);
        printf("%0.2lf,%0.3lf\n", checksum, seconds2milliseconds(time));
    }

    // cleanup
    free(input.elements);
    free(paddedInput.elements);
    free(output.elements);
    free(filters.elements);

    cudaFree(devicePaddedInput.elements);
    cudaFree(deviceOutput.elements);
    cudaFree(deviceFilters.elements);
}
