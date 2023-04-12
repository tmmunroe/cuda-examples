
#ifndef __CONVKERNEL__
#define __CONVKERNEL__

typedef struct __align__(64) {
  int width;
  int height;
  int depth;
  int stride;
  int layerStride;
  double* elements;
} Tensor;

#define blockSize 32
#define inChannels 3
#define inHeight 1024
#define inWidth 1024
#define outChannels 64
#define outHeight 1024
#define outWidth 1024
#define filterHeight 3
#define filterWidth 3
#define filterDepth inChannels
#define filterCount outChannels

__global__ void Conv(const Tensor input, Tensor output, const Tensor filters[filterCount]);
__device__ double convolveWithFilter(const Tensor input, const Tensor filter, int out_x, int out_y);
__device__ void setCellValue(const Tensor target, double value, int x, int y, int z);
__device__ double cellValue(const Tensor source, int x, int y, int z);
__host__ void setCellValueHost(const Tensor target, double value, int x, int y, int z);
__host__ double cellValueHost(const Tensor source, int x, int y, int z);


Tensor createDeviceTensor(Tensor source, bool copy);

Tensor createHostTensor(int width, int height, int depth);

Tensor * createDeviceTensorArray(Tensor * sources, int sourcesCount, bool copy);

#endif
