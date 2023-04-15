
#ifndef __CONVKERNEL__
#define __CONVKERNEL__

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif



typedef struct {
  int width;
  int height;
  int depth;
} TensorDescriptor;

typedef struct {
  int width;
  int height;
  int depth;
  int stride;
  int layerStride;
  double* elements;
} Tensor;

__global__ void Conv(const Tensor input, Tensor output, const Tensor * filters);
__global__ void ConvTiled(const Tensor input, Tensor output, const Tensor * filters);

__device__ double convolveWithFilter(const Tensor input, const Tensor filter, int out_x, int out_y);
__device__ void setCellValue(const Tensor target, double value, int x, int y, int z);
__device__ double cellValue(const Tensor source, int x, int y, int z);
__device__ Tensor cnnSubTensor(const Tensor source, int x, int y, int z,
                            int blockWidth, int blockHeight);

__host__ void setCellValueHost(const Tensor target, double value, int x, int y, int z);
__host__ double cellValueHost(const Tensor source, int x, int y, int z);
__host__ void printTensor(const Tensor source, int x_lim, int y_lim, int z_lim);

Tensor createHostTensor(int width, int height, int depth);
Tensor createHostTensor(const TensorDescriptor tensorDescriptor);
Tensor * createHostTensors(const TensorDescriptor tensorDescriptor, int count);
void freeHostTensors(Tensor * hostTensors, int count);

Tensor createDeviceTensor(const Tensor source, bool copy);
Tensor * createDeviceTensors(const Tensor * sources, int count, bool copy);
void freeDeviceTensors(Tensor * tensors, int count);

#endif
