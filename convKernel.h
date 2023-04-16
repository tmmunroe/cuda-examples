
#ifndef __CONVKERNEL__
#define __CONVKERNEL__

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


__global__ void Conv(const Tensor input, Tensor output, const Tensor filters);
__global__ void ConvTiled(const Tensor input, Tensor output, const Tensor filters);

__global__ int dim(Tensor tensor, int dim);
__global__ int stride(Tensor tensor, int dim);

__global__ int offset(Tensor tensor, int d0, int d1);
__global__ int offset(Tensor tensor, int d0, int d1, int d2);
__global__ int offset(Tensor tensor, int d0, int d1, int d2, int d3);

__global__ double cellValue(Tensor tensor, int d0, int d1, int d2);
__global__ double cellValue(Tensor tensor, int d0, int d1, int d2, int d3);

__global__ void setCellValue(Tensor tensor, double value, int d0, int d1, int d2);
__global__ void setCellValue(Tensor tensor, double value, int d0, int d1, int d2, int d3);



__device__ double convolveWithFilter(const Tensor input, const Tensor filter, int out_x, int out_y);
__device__ Tensor cnnSubTensor(const Tensor source, int x, int y, int z,
                            int blockWidth, int blockHeight);

__host__ void printTensor(const Tensor source, int x_lim, int y_lim, int z_lim);

Tensor createHostTensor(const TensorDescriptor tensorDescriptor);
Tensor createDeviceTensor(const Tensor source, bool copy);

#endif
