///
/// matmultKernel00.cu
/// Based on code from the CUDA Programming Guide
///
/// Multiplies two matrices using CUDA: A x B = C
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
  // store global thread id since the ordering induced by this is
  //  how CUDA partitions threads into warps
  int threadId = threadIdx.y * blockDim.x + threadIdx.x;

  // variables named based on value to make it a little easier to know what their
  //  range is.. these are used for indexing
  // In this kernel, the computation is partitioned into 4 blocks of (32 x 8) elements -
  // that is, 8 rows of 32 elements each. In total, that covers a (32 x 32) element block.
  // Each thread is responsible for 1 element in each of the subblocks.
  // By using the thread's global id modulo 32 as a column index, and the thread's global id
  // divided by 32 (as an integer), we can ensure that threads within a warp will access contiguous memory locations.
  int zeroToEight = threadId / 32;
  int zeroToThirtyTwo = threadId % 32;

  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  // Use col_start and row_start to get the beginning of the regioun of the array
  // that will be covered by this block.. Note that we use the FOOTPRINT_SIZE here, which is 32,
  // instead of the BLOCK_SIZE, which is 16, because we want to process (32x32) patches of the array
  //  which will enable us to coalesce reads and writes of each row
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

  // Each thread computes 4 elements of Csub in its copy of CValue
  // Since this thread block has (32x8) threads to compute (32x32)
  // entries in the matrix multiplication, each thread will calculate 
  // 4 values - one value for each of four rows. This enables reads and writes
  // to be fully coalesced since each warp will access a row of memory.
  int row0 = zeroToEight;
  int row1 = zeroToEight + 8;
  int row2 = zeroToEight + 16;
  int row3 = zeroToEight + 24;

  // registers for each value being computed, corresponding to the rows above, and the 
  //  unique column index denoted by zeroToThirtyTwo.
  float c0 = 0;
  float c1 = 0;
  float c2 = 0;
  float c3 = 0;

  // Loop over all sub matrices in block_row of A and block_col of B
  // required to compute Csub. Block multiply each pair of sub matrices
  // and accumulate results
  for (int block_number = 0;  block_number < (A.width / FOOTPRINT_SIZE); ++block_number){
    // Get Asub and Bsub descriptors
    Asub = &A.elements[A.stride * row_start + FOOTPRINT_SIZE * block_number];
    Bsub = &B.elements[B.stride * FOOTPRINT_SIZE * block_number + col_start];
    
    // Each thread copies just four elements of shared_A and four elements of shared_B
    // note that these reads are fully coalesced because threads in a given warp will
    // access contiguous blocks of memory.
    shared_A[row0][zeroToThirtyTwo] = Asub[row0 * A.stride + zeroToThirtyTwo];
    shared_A[row1][zeroToThirtyTwo] = Asub[row1 * A.stride + zeroToThirtyTwo];
    shared_A[row2][zeroToThirtyTwo] = Asub[row2 * A.stride + zeroToThirtyTwo];
    shared_A[row3][zeroToThirtyTwo] = Asub[row3 * A.stride + zeroToThirtyTwo];

    shared_B[row0][zeroToThirtyTwo] = Bsub[row0 * B.stride + zeroToThirtyTwo];
    shared_B[row1][zeroToThirtyTwo] = Bsub[row1 * B.stride + zeroToThirtyTwo];
    shared_B[row2][zeroToThirtyTwo] = Bsub[row2 * B.stride + zeroToThirtyTwo];
    shared_B[row3][zeroToThirtyTwo] = Bsub[row3 * B.stride + zeroToThirtyTwo];



    // Synchronize to ensure all elements are read
    __syncthreads();

    // Do an inproduct of one row of shared_A and one col of shared_B
    // computing one Cvalue by accumulation
#pragma unroll
    for(int e=0; e<32; ++e) {
      c0 += shared_A[row0][e] * shared_B[e][zeroToThirtyTwo];
      c1 += shared_A[row1][e] * shared_B[e][zeroToThirtyTwo];
      c2 += shared_A[row2][e] * shared_B[e][zeroToThirtyTwo];
      c3 += shared_A[row3][e] * shared_B[e][zeroToThirtyTwo];
    }

    // Synchronize to ensure all Cvalues have been incremented
    // before reading in the next shared_A AND shared_B BLOCKS
    __syncthreads();
  }

  // Write Csub to GLOBAL memory.
  // Each thread writes its own cell value.
  // Note again that these writes are fully coalesced because threads in a given warp
  // will access contigous indexes in memory.
  Csub[row0 * C.stride + zeroToThirtyTwo] = c0;
  Csub[row1 * C.stride + zeroToThirtyTwo] = c1;
  Csub[row2 * C.stride + zeroToThirtyTwo] = c2;
  Csub[row3 * C.stride + zeroToThirtyTwo] = c3;
  __syncthreads();
}

