///
/// matmultKernel00.cu
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// Modified by Wim Bohm and David Newman
/// Created: 2011-01-27
/// Last Modified: 2011-02-23 DVN
///
/// Multiplies two matrices using CUDA: A x B = C
///
/// Copy this file and modify the MatMultKernel device function for
/// each of your experiments. 
///

#include "matmultKernel.h"

// sub-block dimensions
// dim1 is the rows in A and cols in B
// dim2 is the cols in A and rows in B


// Define a gpu kernel to perform matrix multiplication
// of A x B = C.
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){
  // matrix blocks
  float *Asub, *Bsub, *Csub;

  // Putting these into registers speeds access.
  int threadId = threadIdx.y * blockDim.x + threadIdx.x;
  int zeroToEight = threadId / 32;
  int zeroToThirtyTwo = threadId % 32;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;
  int col_start = block_col * FOOTPRINT_SIZE;
  int row_start = block_row * FOOTPRINT_SIZE;

  // Copy ELEMENTS OF  ASub and Bsub into shared memory
  // EACH THREAD loads ONE ELEMENT of ASub and ONE of Bsub
  // Notice: it does not need to be the element it requires to
  //         compute its Cvalue, as long as all elements are 
  //         collaboratively read. 

  // Notice: every thread declares shared_A and shared_B in shared memory
  //         even though a thread block has only one shared_A and one shared_B
  __shared__ float shared_A[32][32];
  __shared__ float shared_B[32][32];

  // Each THREAD BLOCK computes one sub matrix Csub of C
  // EACH THREAD creates its own matrix descriptor Csub
  Csub = &C.elements[C.stride * row_start + col_start];

  // Each thread computes one element of Csub in its copy of CValue
  float c0 = 0;
  float c1 = 0;
  float c2 = 0;
  float c3 = 0;

  // Loop over all sub matrices in block_row of A and block_col of B
  // required to compute Csub. Block multiply each pair of sub matrices
  // and accumulate results
  for (int block_number = 0;  block_number < (A.width / FOOTPRINT_SIZE); ++block_number){
    // Get Asub and Bsub descriptors
    Asub = &A.elements[A.stride * row_start + 32 * block_number];
    Bsub = &B.elements[B.stride * 32 * block_number + col_start];
    
    // Each thread copies just four elements of shared_A and four elements of shared_B
    #pragma unroll
    for (int rowToCopy=zeroToEight; rowToCopy < 32; rowToCopy += 8) {
      shared_A[rowToCopy][zeroToThirtyTwo] = Asub[rowToCopy * A.stride + zeroToThirtyTwo];
      shared_B[rowToCopy][zeroToThirtyTwo] = Bsub[rowToCopy * B.stride + zeroToThirtyTwo];
    }

    // Synchronize to ensure all elements are read
    __syncthreads();

    // Do an inproduct of one row of shared_A and one col of shared_B
    // computing one Cvalue by accumulation
#pragma unroll
    for(int e=0; e<32; ++e) {
      c0 += shared_A[zeroToEight][e] * shared_B[e][zeroToThirtyTwo];
      c1 += shared_A[zeroToEight+8][e] * shared_B[e][zeroToThirtyTwo];
      c2 += shared_A[zeroToEight+16][e] * shared_B[e][zeroToThirtyTwo];
      c3 += shared_A[zeroToEight+24][e] * shared_B[e][zeroToThirtyTwo];
    }

    // Synchronize to ensure all Cvalues have been incremented
    // before reading in the next shared_A AND shared_B BLOCKS
    __syncthreads();
  }

  // Write Csub to GLOBAL memory.
  // Each thread writes its own cell value.
  Csub[zeroToEight * C.stride + zeroToThirtyTwo] = c0;
  Csub[zeroToEight+8 * C.stride + zeroToThirtyTwo] = c1;
  Csub[zeroToEight+16 * C.stride + zeroToThirtyTwo] = c2;
  Csub[zeroToEight+24 * C.stride + zeroToThirtyTwo] = c3;
}

