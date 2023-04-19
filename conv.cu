// Includes
#include <stdio.h>
#include <string>
#include <iostream>
#include "timer.h"
#include <cudnn.h>
#include <stdio.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#ifndef MAX_DIM
#define MAX_DIM 4
#endif

typedef struct {
  int dim;
  int dims[MAX_DIM];
} TensorDescriptor;

typedef struct {
  int dim;
  int dims[MAX_DIM];
  int strides[MAX_DIM];
  double* elements;
} Tensor;


__global__ void Conv(const Tensor input, Tensor output, const Tensor filters, int padding);
__global__ void ConvTiled(const Tensor input, Tensor output, const Tensor filters, int padding);

__host__ __device__ int dim(Tensor tensor, int dim);
__host__ __device__ int stride(Tensor tensor, int dim);
__host__ __device__ void setStridesToDims(Tensor tensor);
__host__ __device__ int elementsCount(Tensor tensor);
__host__ __device__ size_t sizeInBytes(Tensor tensor);

__host__ __device__ int offset(Tensor tensor, int d0, int d1);
__host__ __device__ int offset(Tensor tensor, int d0, int d1, int d2);
__host__ __device__ int offset(Tensor tensor, int d0, int d1, int d2, int d3);

__host__ __device__ double cellValue(Tensor tensor, int d0, int d1, int d2);
__host__ __device__ double cellValue(Tensor tensor, int d0, int d1, int d2, int d3);

__host__ __device__ void setCellValue(Tensor tensor, double value, int d0, int d1, int d2);
__host__ __device__ void setCellValue(Tensor tensor, double value, int d0, int d1, int d2, int d3);

__host__ __device__ Tensor tensorSubBlock(const Tensor source,
    int idx0, int dim0,
    int idx1, int dim1);
__host__ __device__ Tensor tensorSubBlock(const Tensor source,
    int idx0, int dim0,
    int idx1, int dim1,
    int idx2, int dim2);
__host__ __device__ Tensor tensorSubBlock(const Tensor source,
    int idx0, int dim0,
    int idx1, int dim1,
    int idx2, int dim2,
    int idx3, int dim3);
__host__ __device__ Tensor tensorView(const Tensor source);

__host__ __device__ Tensor tensorLayer(const Tensor source, int dim, int idx);


__device__ double convolveWithFilter(const Tensor input, const Tensor filter, int out_x, int out_y);

__host__ void printTensorDescriptor(const TensorDescriptor source);
__host__ __device__ void printTensor(const Tensor source, int x_lim, int y_lim, int z_lim);

Tensor createHostTensor(const TensorDescriptor tensorDescriptor);
Tensor createDeviceTensor(const Tensor source, bool copy);

__host__ __device__ int dim(Tensor tensor, int dim) {
    return tensor.dims[dim];
}

__host__ __device__ int stride(Tensor tensor, int dim) {
    return tensor.strides[dim];
}

__host__ __device__ int elementsCount(Tensor tensor) {
    int count = 1;
    for (int i=0; i<tensor.dim; ++i) {
        count *= tensor.dims[i];
    }
    return count;
}

__host__ __device__ size_t sizeInBytes(Tensor tensor) {
    return elementsCount(tensor) * sizeof(double);
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
	
    sub.elements = &source.elements[offset(source, idx0, idx1, idx2, idx3)];
    return sub;
};

__host__ __device__ Tensor tensorView(const Tensor source) {
    Tensor tensor;
    tensor.dim = source.dim;
    tensor.elements = source.elements;
    for (int i=0; i<source.dim; ++i) {
        tensor.dims[i] = source.dims[i];
        tensor.strides[i] = source.strides[i];
    }
    return tensor;
}

__host__ __device__ Tensor tensorLayer(const Tensor source, int dim, int idx) {
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
    return Tensor{};
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
  }

  size_t size = tensor.strides[tensor.dim-1] * sizeof(double);
  tensor.elements = (double*)malloc(size);

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

    // note that z is the same for both the filter and the input
    for (int z = 0; z < depth; ++z) {
        for(int dy = 0; dy < height; ++dy) {
            for(int dx = 0; dx < width; ++dx) {
                int in_x = start_x + dx;
                int in_y = start_y + dy;
                
                // Verify we are inside the boundaries width and height
                if(in_x > -1 && in_x < input_width
                    && in_y > -1 && in_y < input_height) {
                    pixelValue += cellValue(input, in_x, in_y, z) * cellValue(filter, width-1-dx, height-1-dy, z);
                }
            }
        }
    }

    return pixelValue;
}

__global__ void ConvTiled(const Tensor paddedInput, Tensor output, const Tensor filters, int padding) {
    // declare shared
    extern __shared__ double array[];

    double value;
    int sharedFilterCount = elementsCount(filters);
    int thread_x = threadIdx.x;
    int out_x = blockIdx.x * blockDim.x + thread_x;
    int out_y = blockIdx.y * blockDim.y;
    // if (out_x < output.dims[0] && out_y < output.dims[1]) {

    int threadCount = blockDim.x * blockDim.y;
    int block_dim_x = blockDim.x;
    int input_block_size_x = block_dim_x + (2*padding);
    int input_block_size_y = filters.dims[1];
    //printf("input_block_size_x: %d, input_block_size_y: %d\n", input_block_size_x, input_block_size_y);

    // transfer all filters to shared memory
    Tensor sharedFilter = tensorView(filters);
    sharedFilter.elements = array;
    for (int i=thread_x; i<sharedFilterCount; i+=threadCount) {
        sharedFilter.elements[i] = filters.elements[i];
    }

    // transfer BLOCK_SIZE * BLOCK_SIZE * input_depth size section of input into memory
    Tensor inputSubBlock = tensorSubBlock(paddedInput, 
        blockIdx.x * blockDim.x, input_block_size_x,
        blockIdx.y * blockDim.y, input_block_size_y,
        0, paddedInput.dims[2]);

    // create tensor to wrap shared input and copy input to shared input, resetting strides and pointing to shared memory
    Tensor sharedInput = tensorView(inputSubBlock);
    sharedInput.strides[0] = sharedInput.dims[0];
    for (int i=1; i<sharedInput.dim; ++i) {
        sharedInput.strides[i] = sharedInput.strides[i-1] * sharedInput.dims[i];
    }
    sharedInput.elements = &array[sharedFilterCount]; // start at end of shared memory for filters
//    printf("sharedInput: dims: %d, %d, %d, strides: %d, %d, %d\n", sharedInput.dims[0], sharedInput.dims[1],
		    // sharedInput.dims[2], sharedInput.strides[0], sharedInput.strides[1], sharedInput.strides[2]);

    // copy values over
    for (int z=0; z < paddedInput.dims[2]; ++z) {
        for (int dy=0; dy < input_block_size_y; ++dy) {
            for (int x=thread_x; x < input_block_size_x; x+=block_dim_x) {
                // copy from input to shared_input, keeping in mind that the sharedInput
                value = cellValue(inputSubBlock, x, dy, z);
                //setCellValue(sharedInput, value, x, out_y + dy, z);
		//printf("Setting  (%d, %d, %d), offset=%d\n", x, out_y+dy, z, offset(sharedInput, x, dy, z));
    		sharedInput.elements[offset(sharedInput, x, dy, z)] = value;
            }
        }
    }

    // sync threads
    __syncthreads();

    // run convolutions
    int filterCount = output.dims[2];
    for (int out_z = 0; out_z < filterCount; ++out_z) {
        Tensor filter = tensorLayer(sharedFilter, 4, out_z);
        if (out_x < output.dims[0] && out_y < output.dims[1]) {
            // remember, sharedInput pads borders, so we actually want x+padding and y+padding
            double pixelValue = convolveWithFilter(sharedInput, filter, thread_x+padding, padding);
            //setCellValue(output, pixelValue, out_x, out_y, out_z);
	    output.elements[offset(output, out_x, out_y, out_z)] = pixelValue;
        }
    }
    // }
}


__global__ void Conv(const Tensor paddedInput, Tensor output, const Tensor filters, int padding) {
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
            double pixelValue = convolveWithFilter(paddedInput, filter, out_x+padding, out_y+padding);
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
