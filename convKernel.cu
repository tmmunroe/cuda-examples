#include "convKernel.h"
#include <stdio.h>

__device__ void setCellValue(Tensor target, double value, int x, int y, int z) {
    target.elements[x + y*target.stride + z*target.layerStride] = value;
}

__device__ double cellValue(const Tensor source, int x, int y, int z) {
    return source.elements[x + y*source.stride + z*source.layerStride];
}

__host__ void setCellValueHost(Tensor target, double value, int x, int y, int z) {
    target.elements[x + y*target.stride + z*target.layerStride] = value;
}

__host__ double cellValueHost(const Tensor source, int x, int y, int z) {
    return source.elements[x + y*source.stride + z*source.layerStride];
}


Tensor createDeviceTensor(const Tensor source, bool copy) {
  // Create a new matrix in device memory.
  Tensor tensor;
  tensor.width = source.width;
  tensor.height = source.height;
  tensor.depth = source.depth;
  tensor.stride = source.width;
  tensor.layerStride = source.layerStride;

  size_t size = source.width * source.height * source.depth * sizeof(double);
  cudaMalloc((void**) &tensor.elements, size);
  if (copy)
    cudaMemcpy(tensor.elements, source.elements, size, cudaMemcpyHostToDevice);

  return tensor;
}

Tensor * createHostTensors(const TensorDescriptor tensorDescriptor, int count) {
    Tensor * tensors = (Tensor*) malloc(sizeof(Tensor)*count);
    for (int i = 0; i<count; ++i) {
        tensors[i] = createHostTensor(tensorDescriptor);
    }
    return tensors;
}

void freeHostTensors(Tensor * hostTensors, int count) {
    for (int i = 0; i < count; ++i) {
        free(hostTensors[i].elements);
    }
}

Tensor * createDeviceTensors(const Tensor * sources, int count, bool copy) {
    // this is a two step procedure where we first create the device tensors in
    //  a host array; and then we create the device array containing those device tensors

    // create device tensors for filters
    Tensor * device_tensors = (Tensor*) malloc(sizeof(Tensor)*count);
    for (int k = 0; k < count; ++k) {
        device_tensors[k] = createDeviceTensor(sources[k], true);
    }

    // create device array of tensors with device tensors
    Tensor * device_tensors_device_array;
    cudaMalloc((void**)&device_tensors_device_array, count*sizeof(Tensor));
    cudaMemcpy(device_tensors_device_array, device_tensors, count*sizeof(Tensor), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // free host array
    free(device_tensors);

    return device_tensors_device_array;
}

void freeDeviceTensors(Tensor * tensors, int count) {
    // this is a four step procedure where we
    // 1) create a host array pointing to the device tensors
    // 2) iterate over the host array to free the device tensor element arrays
    // 3) free the device array that was passed in
    // 4) free the host array we used to access the device tensors
    size_t size = sizeof(Tensor)*count;

    Tensor * device_tensors_host_array = (Tensor*) malloc(size);
    cudaMemcpy(device_tensors_host_array, tensors, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < count; ++i) {
        cudaFree(device_tensors_host_array[i].elements);
    }

    cudaFree(tensors);
    free(device_tensors_host_array);
}


// Create a matrix in host memory.
Tensor createHostTensor(int width, int height, int depth){
  Tensor tensor;
  tensor.width = width;
  tensor.height = height;
  tensor.depth = depth;
  tensor.stride = width;
  tensor.layerStride = width*height;

  size_t size = width * height * depth * sizeof(double);
  // printf("Creating tensor with dims (%d, %d, %d) and size %zu\n", width, height, depth, size);
  tensor.elements = (double*)malloc(size);

  // printf("Created tensor with dims (%d, %d, %d) and size %zu\n", width, height, depth, size);

  return tensor;
}

Tensor createHostTensor(const TensorDescriptor tensorDescriptor) {
    return createHostTensor(tensorDescriptor.width, tensorDescriptor.height, tensorDescriptor.depth);
}

__device__ double convolveWithFilter(const Tensor input, const Tensor filter, int out_x, int out_y) {
    // using lecture notes as a basis for this function
    double pixelValue = 0.0;

    // note that z is the same for both the filter andand the input
    int start_x = out_x - (filter.width/2);
    int start_y = out_y - (filter.height/2);

    // note that z is the same for both the filter and the input
    for (int z = 0; z < filter.depth; ++z) {
        for(int dy = 0; dy < filter.height; ++dy) {
            for(int dx = 0; dx < filter.width; ++dx) {
                int in_x = start_x + dx;
                int in_y = start_y + dy;
                
                // Verify we are inside the boundaries width and height
                if(in_x > -1 && in_x < input.width
                    && in_y > -1 && in_y < input.height) {
                    //NOTE: we flip dy and dx when indexing into the filter in order to get the transpose of it
                    pixelValue += cellValue(input, in_x, in_y, z) * cellValue(filter, dy, dx, z);
                }
            }
        }
    }

    return pixelValue;
}

__global__ void Conv(const Tensor input, Tensor output, const Tensor * filters) {
    // int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int filterCount = output.depth;

    for (int out_z = 0; out_z < filterCount; ++out_z) {
        if (out_x < output.width && out_y < output.height) {
            double pixelValue = convolveWithFilter(input, filters[out_z], out_x, out_y);
            setCellValue(output, pixelValue, out_x, out_y, out_z);
        }
    }
}

__host__ void printTensor(const Tensor source, int x_lim, int y_lim, int z_lim) {
    for (int z=0; z < z_lim; ++z) {
        printf("\nDepth=%d", z);
        for (int y=0; y < y_lim; ++y) {
            printf("\n");
            for (int x=0; x < x_lim; ++x) {
                printf("%lf ", cellValueHost(source, x, y, z));
            }
        }
    }
    printf("\n");
}