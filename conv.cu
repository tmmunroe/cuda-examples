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
    for (int c = 0; c < tensor.depth; ++c) {
        for (int y = 0; y < tensor.height; ++y) {
            for (int x = 0; x < tensor.width; ++x) {
                value = c*double(x+y);
                setCellValueHost(tensor, value, x, y, c);
            }
        }
    }
}

void fillFilter(Tensor tensor, int k) {
    double value;
    for (int c = 0; c < tensor.depth; ++c) {
        for (int y = 0; y < tensor.height; ++y) {
            for (int x = 0; x < tensor.width; ++x) {
                value = (c+k)*double(x+y);
                setCellValueHost(tensor, value, x, y, c);
            }
        }
    }
}

void fillOnes(Tensor tensor) {
    for (int c = 0; c < tensor.depth; ++c) {
        for (int y = 0; y < tensor.height; ++y) {
            for (int x = 0; x < tensor.width; ++x) {
                setCellValueHost(tensor, 1.0, x, y, c);
            }
        }
    }
}

double calculateChecksum(Tensor output) {
    double checksum(0);
    for (int z=0; z<output.depth; ++z) {
        for (int y=0; y<output.height; ++y) {
            for (int x=0; x<output.width; ++x) {
                checksum += cellValueHost(output, x, y, z);
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
    for (int c = 0; c < output.depth; ++c) {
        for (int y = 0; y < output.height; ++y) {
            for (int x = 0; x < output.width; ++x) {
                if (x > 0 && y > 0 && x < output.width-1 && y < output.height-1) {
                    expectedValue = 27;
                } else if (x > 0 && x < output.width-1) {
                    expectedValue = 18;
                } else if (y > 0 && y < output.height-1) {
                    expectedValue = 18;
                } else { // both edges are off input grid
                    expectedValue = 12;
                }

                value = cellValueHost(output, x, y, c);
                if (fabs(value - expectedValue) > 1e-5) {
                    printf("Error at (%d, %d, %d).. value %lf, expected %lf\n", x, y, c, value, expectedValue);
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
    double time;

    // tensor specifications
    const TensorDescriptor inputDescriptor{1024, 1024, 3};
    const TensorDescriptor outputDescriptor{1024, 1024, 64};
    const TensorDescriptor filterDescriptor{3, 3, inputDescriptor.depth};
    const int filterCount(outputDescriptor.depth);

    bool isTestCase = false;
    bool verbose = false;
    if (argc > 2) {
        isTestCase = std::string("test") == argv[1];
        verbose = std::string("verbose") == argv[2];
    } else if (argc > 1) {
        isTestCase = std::string("test") == argv[1];
    }

    // create tensors for input, output, and an array of tensors for the filters
    Tensor input = createHostTensor(inputDescriptor);
    Tensor output = createHostTensor(outputDescriptor);
    Tensor * filters = createHostTensors(filterDescriptor, filterCount);

    printf("Created all tensors\n");
    // initialize input tensor and filters with values
    if (isTestCase) {
        std::cout << "filling with test case values..." << std::endl;
        fillOnes(input);
        for (int k = 0; k < filterCount; ++k) {
            fillOnes(filters[k]);
        }
    } else {
        fillInput(input);
        for (int k = 0; k < filterCount; ++k) {
            fillFilter(filters[k], k);
        }
    }

    if (verbose) {
        printf("Section of filter: ");
        printTensor(filters[0], 3, 3, 3);

        printf("Section of input: ");
        printTensor(input, 3, 3, 3);
    }

    // create tensors on device
    Tensor device_input = createDeviceTensor(input, true);
    Tensor device_output = createDeviceTensor(output, false);
    Tensor * deviceFilters = createDeviceTensors(filters, filterCount, true);

    //define dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(device_output.width/BLOCK_SIZE, device_output.height/BLOCK_SIZE);
    cudaDeviceSynchronize();

    // Initialize timer  
    initialize_timer();
    start_timer();

    Conv<<<dimGrid, dimBlock>>>(device_input, device_output, deviceFilters);
    cudaDeviceSynchronize();

    // Compute and return elapsed time 
    stop_timer();
    time = elapsed_time();

    // copy to host
    size_t size = output.width * output.height * output.depth * sizeof(double);
    cudaMemcpy(output.elements, device_output.elements, size, cudaMemcpyDeviceToHost);

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
    free(output.elements);
    freeHostTensors(filters, filterCount);

    cudaFree(device_input.elements);
    cudaFree(device_output.elements);
    freeDeviceTensors(deviceFilters, filterCount);
}
