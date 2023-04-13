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

int main(int argc, char ** argv) {
    // const int inChannels(3), inHeight(1024), inWidth(1024);
    // const int outChannels(64), outHeight(1024), outWidth(1024);
    // const int filterHeight(3), filterWidth(3);
    // const int filterDepth(inChannels), filterCount(outChannels);

    double time;
    // create tensors for input and output
    Tensor input = createHostTensor(inWidth, inHeight, inChannels);
    Tensor output = createHostTensor(outWidth, outHeight, outChannels);

    // create tensors for filters
    Tensor filters[filterCount];
    for (int k = 0; k < filterCount; ++k) {
        filters[k] = createHostTensor(filterWidth, filterHeight, filterDepth);
    }

    printf("Created all tensors\n");
    // initialize input tensor and filters with values
    double value;
    double testFillValue(1.0);
    printf("Filling input with %lf\n", testFillValue);
    for (int c = 0; c < input.depth; ++c) {
        for (int y = 0; y < input.height; ++y) {
            for (int x = 0; x < input.width; ++x) {
                // I[c, x, y] = c · (x + y)
                // value = c*double(x+y);
                // setCellValueHost(input, value, x, y, c);
                setCellValueHost(input, testFillValue, x, y, c);

            }
        }
    }

    printf("Filling filters\n");
    for (int k = 0; k < filterCount; ++k) {
        Tensor filter = filters[k];
        for (int c = 0; c < filter.depth; ++c) {
            for (int y = 0; y < filter.height; ++y) {
                for (int x = 0; x < filter.width; ++x) {

                    //F[k, c, i, j] = (c + k) · (i + j)
                    // value = (c+k)*double(x+y);
                    // setCellValueHost(filter, value, x, y, c);

                    setCellValueHost(filter, testFillValue, x, y, c);
                }
            }
        }
    }
    printf("Section of filter: ");
    printTensor(filters[0], 3, 3, 3);

    printf("Section of input: ");
    printTensor(input, 3, 3, 3);



    // create tensors on device
    Tensor device_input = createDeviceTensor(input, true);
    Tensor device_output = createDeviceTensor(output, false);

    // create tensors for filters on device
    Tensor device_filters[filterCount];
    for (int k = 0; k < filterCount; ++k) {
        device_filters[k] = createDeviceTensor(filters[k], true);
    }
    // create device array of tensors with device tensors
    Tensor * device_filters_device_array;
    cudaMalloc((void**)&device_filters_device_array, filterCount*sizeof(Tensor));
    cudaMemcpy(device_filters_device_array, device_filters, filterCount*sizeof(Tensor), cudaMemcpyHostToDevice);
    

    //define dimensions
    dim3 dimBlock(blockSize, blockSize);
    dim3 dimGrid(device_output.width/blockSize, device_output.height/blockSize);

    // Initialize timer  
    initialize_timer();
    start_timer();

    Conv<<<dimGrid, dimBlock>>>(device_input, device_output, device_filters_device_array);

    // Compute and return elapsed time 
    stop_timer();
    time = elapsed_time();

    // copy to host
    size_t size = output.width * output.height * output.depth * sizeof(double);
    cudaMemcpy(output.elements, device_output.elements, size, cudaMemcpyDeviceToHost);

    // check result
    int errors = 0;
    double expectedValue(9.0);
    for (int c = 0; c < output.depth; ++c) {
        for (int y = 0; y < output.height; ++y) {
            for (int x = 0; x < output.width; ++x) {
                if (x > 0 && y > 0 && x < output.width-1 && y < output.height-1) {
                    expectedValue = 27;
                } else if (x > 0 || x < output.width-1) {
                    expectedValue = 18;
                } else if (y > 0 || y < output.height-1) {
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

    // cleanup
    free(input.elements);
    free(output.elements);
    for (int c = 0; c < filterCount; ++c) {
        free(filters[c].elements);
    }

    cudaFree(device_input.elements);
    cudaFree(device_output.elements);
    for (int c = 0; c < filterCount; ++c) {
        cudaFree(device_filters[c].elements);
    }
}
