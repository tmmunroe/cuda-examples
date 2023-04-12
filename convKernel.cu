#include "convKernel.h"

__global__ void Conv(const Tensor input, Tensor output, const Tensor filters[filterCount]) {
    // int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    for (int out_z = 0; out_z < filterCount; ++out_z) {
        if (out_x < output.width && out_y < output.height) {
            double pixelValue = convolveWithFilter(input, filters[out_z], out_x, out_y);
            setCellValue(output, pixelValue, out_x, out_y, out_z);
        }
    }
}
