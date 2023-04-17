#include "convKernel.h"
#include <stdio.h>


__host__ __device__ int dim(Tensor tensor, int dim) {
    return tensor.dims[dim];
}

__host__ __device__ int stride(Tensor tensor, int dim) {
    return tensor.strides[dim];
}

__host__ __device__ int offset(Tensor tensor, int d0, int d1) {
    return d0 + tensor.strides[0]*d1;
}

__host__ __device__ int offset(Tensor tensor, int d0, int d1, int d2) {
    return d0 + tensor.strides[0]*d1 + tensor.strides[1]*d2;
}

__host__ __device__ int offset(Tensor tensor, int d0, int d1, int d2, int d3) {
    return d0 + tensor.strides[0]*d1 + tensor.strides[1]*d2 + tensor.strides[2]*d3;
}

__host__ __device__ double cellValue(Tensor tensor, int d0, int d1, int d2) {
    return tensor.elements[offset(tensor, d0, d1, d2)];
}

__host__ __device__ double cellValue(Tensor tensor, int d0, int d1, int d2, int d3) {
    return tensor.elements[offset(tensor, d0, d1, d2, d3)];
}

__host__ __device__ void setCellValue(Tensor tensor, double value, int d0, int d1, int d2) {
    tensor.elements[offset(tensor, d0, d1, d2)] = value;
}

__host__ __device__ void setCellValue(Tensor tensor, double value, int d0, int d1, int d2, int d3) {
    tensor.elements[offset(tensor, d0, d1, d2, d3)] = value;
}

// __device__ Tensor cnnSubTensor(const Tensor source, int x, int y, int blockWidth, int blockHeight) {
//     Tensor sub;
//     sub.width = blockWidth;
//     sub.height = blockHeight;
//     sub.depth = source.depth;

//     sub.stride = source.stride;
//     sub.layerStride = source.layerStride;

//     sub.elements = &source.elements[source.stride * blockHeight * y + blockWidth * x];
//     return sub;
// }

__host__ __device__ Tensor tensorSubBlock(const Tensor source, int idx0, int dim0, int idx1, int dim1) {
    Tensor sub;
    sub.dim = 2;
    for (int d=0; d<source.dim; ++d) {
        sub.strides[d] = source.strides[d];
    }
    sub.dims[0] = dim0;
    sub.dims[1] = dim1;

    sub.elements = &source.elements[offset(source, idx0, idx1)];
    return sub;
};

__host__ __device__ Tensor tensorSubBlock(const Tensor source, int idx0, int dim0, int idx1, int dim1, int idx2, int dim2) {
    Tensor sub;
    sub.dim = 3;
    for (int d=0; d<source.dim; ++d) {
        sub.strides[d] = source.strides[d];
    }
    sub.dims[0] = dim0;
    sub.dims[1] = dim1;
    sub.dims[2] = dim2;

    sub.elements = &source.elements[offset(source, idx0, idx1, idx2)];
    return sub;
};

__host__ __device__ Tensor tensorSubBlock(const Tensor source,
    int idx0, int dim0,
    int idx1, int dim1,
    int idx2, int dim2,
    int idx3, int _dim3) {
    Tensor sub;
    sub.dim = 4;
    for (int d=0; d<source.dim; ++d) {
        sub.strides[d] = source.strides[d];
    }
    sub.dims[0] = dim0;
    sub.dims[1] = dim1;
    sub.dims[2] = dim2;
    sub.dims[3] = _dim3;
	
    //printf("\ntensorSubBlock: (%d, %d) (%d, %d) (%d, %d) (%d, %d), offset: %d\n\n",
		    //idx0, dim0, idx1, dim1, idx2, dim2, idx3, _dim3, offset(source, idx0, idx1, idx2, idx3)
		    //);
    sub.elements = &source.elements[offset(source, idx0, idx1, idx2, idx3)];
    //printf("\nfirst elements: %f, %f, %f\n", sub.elements[0], sub.elements[1], sub.elements[2]);
    return sub;
};

__host__ __device__ Tensor tensorLayer(const Tensor source, int dim, int idx) {
    if (dim < 1 || dim > source.dim) {
        return Tensor{};
    }

    if (dim == 1) {
        return tensorSubBlock(
            source,
            0, source.dims[0],
            idx, 1
        );
    } else if (dim == 2) {
        return tensorSubBlock(
            source,
            0, source.dims[0],
            0, source.dims[1],
            idx, 1
	);
    } else if (dim == 4) {
        return tensorSubBlock(
            source,
            0, source.dims[0],
            0, source.dims[1],
            0, source.dims[2],
            idx, 1
        );
    }
}

Tensor createDeviceTensor(const Tensor source, bool copy) {
  // Create a new matrix in device memory.
  Tensor tensor;
  tensor.dim = source.dim;
  for (int i=0; i<source.dim; ++i) {
    tensor.dims[i] = source.dims[i];
    tensor.strides[i] = source.strides[i];
  }

  size_t size = tensor.strides[tensor.dim-1] * sizeof(double);
  cudaMalloc((void**) &tensor.elements, size);
  if (copy)
    cudaMemcpy(tensor.elements, source.elements, size, cudaMemcpyHostToDevice);

  return tensor;
}

// Create a matrix in host memory.
Tensor createHostTensor(const TensorDescriptor tensorDescriptor){
  Tensor tensor;
  int stride = 1;

  tensor.dim = tensorDescriptor.dim;
  for (int i=0; i<tensorDescriptor.dim; ++i) {
    tensor.dims[i] = tensorDescriptor.dims[i];
    stride = stride * tensorDescriptor.dims[i];
    tensor.strides[i] = stride;
    printf("Stride %d: %d from %d\n", i, tensor.strides[i], tensorDescriptor.dims[i]);
  }

  size_t size = tensor.strides[tensor.dim-1] * sizeof(double);
  // printf("Creating tensor with dims (%d, %d, %d) and size %zu\n", width, height, depth, size);
  tensor.elements = (double*)malloc(size);

  // printf("Created tensor with dims (%d, %d, %d) and size %zu\n", width, height, depth, size);

  return tensor;
}

__device__ double convolveWithFilter(const Tensor input, const Tensor filter, int x, int y) {
    // using lecture notes as a basis for this function
    double pixelValue = 0.0;

    // note that z is the same for both the filter andand the input
    int width = filter.dims[0];
    int height = filter.dims[1];
    int depth = filter.dims[2];
    int input_width = input.dims[0];
    int input_height = input.dims[1];

    int start_x = x - (width/2);
    int start_y = y - (height/2);

    //printf("x: %d, y: %d, start_x: %d, start_y: %d\n", x, y, start_x, start_y);
    // note that z is the same for both the filter and the input
    for (int z = 0; z < depth; ++z) {
        for(int dy = 0; dy < height; ++dy) {
            for(int dx = 0; dx < width; ++dx) {
                int in_x = start_x + dx;
                int in_y = start_y + dy;
                
                // Verify we are inside the boundaries width and height
                if(in_x > -1 && in_x < input_width
                    && in_y > -1 && in_y < input_height) {
                    //NOTE: we flip dy and dx when indexing into the filter in order to get the transpose of it
                    pixelValue += cellValue(input, in_x, in_y, z) * cellValue(filter, dy, dx, z);
                }
            }
        }
    }

//printf("cellvalue: %lf, cellvalue: %lf\n", cellValue(input, x, y, 0), cellValue(filter, 1,1,1));
//printf("returning (%d, %d): %lf... (w: %d, h: %d, d: %d, iw: %d, ih: %d)\n", x, y, pixelValue, width, height, depth, input_width, input_height);
    return pixelValue;
}

// __global__ void ConvTiled(const Tensor input, Tensor output, const Tensor filters) {
//     // declare shared
//     __shared__ double filters[64][3][3][3];
//     __shared__ double shared_input[BLOCK_SIZE+1][BLOCK_SIZE+1][3];


//     int threadId = threadIdx.y * blockDim.x + threadIdx.x;
//     int out_x = blockIdx.x * blockDim.x + threadIdx.x;
//     int out_y = blockIdx.y * blockDim.y + threadIdx.y;
//     int filterCount = output.depth;

//     // copy filters and inputs to shared memory
//     for (int out_z = 0; out_z < filterCount; ++out_z) {
//         int k = threadId 
//     }

//     // 

//     // convolve for each filter
//     for (int out_z = 0; out_z < filterCount; ++out_z) {

//         if (out_x < output.width && out_y < output.height) {
//             double pixelValue = convolveWithFilter(input, filters[out_z], out_x, out_y);
//             setCellValue(output, pixelValue, out_x, out_y, out_z);
//         }
//     }

// }


__global__ void Conv(const Tensor input, Tensor output, const Tensor filters) {
    // int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int filterCount = output.dims[2];
    for (int out_z = 0; out_z < filterCount; ++out_z) {
        Tensor filter = tensorLayer(filters, 4, out_z);
	
	if (false && out_x == 0 && out_y == 0 && out_z == 1) {
	printf("Filter %d\n", out_z);
        Tensor filter = tensorLayer(filters, 4, out_z);
	printTensor(filter, 3, 3, 3);
	}
        if (out_x < output.dims[0] && out_y < output.dims[1]) {
            double pixelValue = convolveWithFilter(input, filter, out_x, out_y);
            setCellValue(output, pixelValue, out_x, out_y, out_z);
        }
    }
}

__host__ __device__ void printTensor(const Tensor source, int x_lim, int y_lim, int z_lim) {
    printf("Tensor Specs:\n");
    printf("Dim: %d\n", source.dim);
    printf("Dims: ");
    for (int i=0; i < source.dim; ++i) {
        printf("%d: %d, ", i, source.dims[i]);
    }
    printf("\nStrides: ");
    for (int i=0; i < source.dim; ++i) {
        printf("%d: %d, ", i, source.strides[i]);
    }
    printf("\n");

    for (int z=0; z < z_lim; ++z) {
        printf("\nDepth=%d", z);
        for (int y=0; y < y_lim; ++y) {
            printf("\n");
            for (int x=0; x < x_lim; ++x) {
                printf("%lf ", cellValue(source, x, y, z));
            }
        }
    }
    printf("\n");
}

__host__ void printTensorDescriptor(const TensorDescriptor source) {
    printf("TensorDescriptor Specs:\n");
    printf("Dim: %d\n", source.dim);
    printf("Dims: ");
    for (int i=0; i < source.dim; ++i) {
        printf("%d, ", source.dims[i]);
    }
    printf("\n");
}
