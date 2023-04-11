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
  int thread_row = threadId / 32;
  int thread_col = threadId % 32;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;
  int col_start = block_col * FOOTPRINT_SIZE;

  // Copy ELEMENTS OF  ASub and Bsub into shared memory
  // EACH THREAD loads ONE ELEMENT of ASub and ONE of Bsub
  // Notice: it does not need to be the element it requires to
  //         compute its Cvalue, as long as all elements are 
  //         collaboratively read. 

  // Notice: every thread declares shared_A and shared_B in shared memory
  //         even though a thread block has only one shared_A and one shared_B
  __shared__ float shared_A[8][32];
  __shared__ float shared_B[32][8];

  #pragma unroll
  for (int row_count = 0; row_count < 4; ++row_count) {
    int row_start = block_row * FOOTPRINT_SIZE + row_count * 8;

    // Each THREAD BLOCK computes one sub matrix Csub of C
    // EACH THREAD creates its own matrix descriptor Csub
    Csub = &C.elements[C.stride * row_start + col_start];

    // Each thread computes one element of Csub in its copy of CValue
    float Cvalue = 0;

    // Loop over all sub matrices in block_row of A and block_col of B
    // required to compute Csub. Block multiply each pair of sub matrices
    // and accumulate results
    for (int block_number = 0;  block_number < (A.width / 32); ++block_number){
      // Get Asub and Bsub descriptors
      Asub = &A.elements[A.stride * row_start + 32 * block_number];
      Bsub = &B.elements[B.stride * 32 * block_number + col_start];
      
      // Each thread copies just one element of shared_A and one element of shared_B
      shared_A[thread_row][thread_col] = Asub[thread_row * A.stride + thread_col];
      shared_B[thread_col][thread_row] = Bsub[thread_col * B.stride + thread_row];

      // Synchronize to ensure all elements are read
      __syncthreads();

      // Do an inproduct of one row of shared_A and one col of shared_B
      // computing one Cvalue by accumulation
  #pragma unroll
      for(int e=0; e<32; ++e)
        Cvalue += shared_A[thread_row][e] * shared_B[e][thread_col];

      // Synchronize to ensure all Cvalues have been incremented
      // before reading in the next shared_A AND shared_B BLOCKS
      __syncthreads();
    }

    // Write Csub to GLOBAL memory.
    // Each thread writes its own cell value.
    Csub[thread_row * C.stride + thread_col] = Cvalue;
    __syncthreads();
  }
}

