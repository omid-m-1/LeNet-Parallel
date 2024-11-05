#include "kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define  part 8 //divide output columns to which size

// Subwarp reduction
template <int dim_wcount> //which number of subwarps
__device__ inline float subwarp_reduce(float val){
    for(int offset = 16/dim_wcount; offset > 0; offset /= 2)
        val+= __shfl_down_sync (FULL_WARP_MASK,val,offset, 32/dim_wcount);
    return val;
}

//matrix multiplication with data-reuse across i dimension
__global__ void mm4d_pr4(float* a, float* b, float* c,int d2a, int d2b, int d2c, int aSize, int bSize, int cSize) {
  int i, j;

    int tid = threadIdx.x; //thread id
  i = blockIdx.y* blockDim.y + threadIdx.y; //which mini-batch
    j = blockIdx.x*part; //which part of output result
    int j_upper = min(part, d2c-j); //last index of part

    //first element index of matrix A, matrix B and output
    int idx_a_base = i * d2a;
    int idx_b_base = j * d2b;
    int idx_c_base = i * d2c + j;

    //initialize variables to zero
    float a_value = 0.0f;
    float sum[part] = {0.0f};

    //repeat on shared dimension
    for (int kk = tid; kk < d2a; kk += 32) {
      int idx_a = idx_a_base + kk; //matrix A index
      if (idx_a < aSize) a_value = a[idx_a]; //save in temporary variable for data-reuse
      //repeat on part of output column
      for (int jj = 0; jj < j_upper; ++jj) {
        int idx_b = idx_b_base + jj * d2b + kk;
        if (idx_b < bSize) sum[jj] += a_value * b[idx_b];
      }
    }
    //reduce subwarp results and save to ouput matrix
    for (int jj = 0; jj < j_upper; ++jj) sum[jj] = subwarp_reduce<1>(sum[jj]);
    for (int jj = 0; jj < j_upper; ++jj) {if(tid == 0 && (idx_c_base + jj) < cSize) c[idx_c_base + jj] = sum[jj];}
}

//matrix multiplication and add bias with data-reuse across i dimension
__global__ void mm4d_pr4_bias(float* a, float* b, float* c, float* bias, int d2a, int d2b, int d2c, int aSize, int bSize, int cSize) {
  int i, j;

    int tid = threadIdx.x; //thread id
  i = blockIdx.y* blockDim.y + threadIdx.y; //which mini-batch
    j = blockIdx.x*part; //which part of output result
    int j_upper = min(part, d2c-j); //last index of part

    //first element index of matrix A, matrix B and output
    int idx_a_base = i * d2a;
    int idx_b_base = j * d2b;
    int idx_c_base = i * d2c + j;

    //initialize variables to zero
    float a_value = 0.0f;
    float sum[part] = {0.0f};

    //repeat on shared dimension
    for (int kk = tid; kk < d2a; kk += 32) {
      int idx_a = idx_a_base + kk; //matrix A index
      if (idx_a < aSize) a_value = a[idx_a]; //save in temporary variable for data-reuse

      //repeat on part of output column
      for (int jj = 0; jj < j_upper; ++jj) {
        int idx_b = idx_b_base + jj * d2b + kk;
        if (idx_b < bSize) sum[jj] += a_value * b[idx_b];
      }
    }

    //reduce subwarp results and save to ouput matrix
    for (int jj = 0; jj < j_upper; ++jj) sum[jj] = subwarp_reduce<1>(sum[jj]);
    for (int jj = 0; jj < j_upper; ++jj) {if(tid == 0 && (idx_c_base + jj) < cSize) c[idx_c_base + jj] = sum[jj] + bias[j + jj];}
}

//matrix multiplication with shared memory and data-reuse across i dimension
__global__ void mm4d_pr4_shared(float* a, float* b, float* c, int d2a, int d2b, int d2c, int aSize, int bSize, int cSize) {
  extern __shared__ float s_b[]; //shared memory for matrix B

  int i, j;

    int tid = threadIdx.x; //thread id
  i = blockIdx.y* blockDim.y + threadIdx.y; //which mini-batch
    j = blockIdx.x*part; //which part of output result
    int j_upper = min(part, d2c-j); //last index of part

    //first element index of matrix A, matrix B and output
    int idx_a_base = i * d2a;
    int idx_b_base = j * d2b;
    int idx_c_base = i * d2c + j;

    //initialize variables to zero
    float a_value = 0.0f;
    float sum[part] = {0.0f};

    // load data to shared memory
    for (int kk = tid; kk < d2a; kk += 32) {
        int idx_b = idx_b_base + threadIdx.y * d2b + kk;
        if (idx_b < bSize) s_b[kk + threadIdx.y*d2b] = b[idx_b];
    }
    __syncthreads();

    //repeat on shared dimension
    for (int kk = tid; kk < d2a; kk += 32) {
      int idx_a = idx_a_base + kk; //matrix A index
      if (idx_a < aSize) a_value = a[idx_a]; //save in temporary variable for data-reuse
      //repeat on part of output column
      for (int jj = 0; jj < j_upper; ++jj) {
        sum[jj] += a_value * s_b[kk + jj*d2b];
      }
    }

    //reduce subwarp results and save to ouput matrix
    for (int jj = 0; jj < j_upper; ++jj) sum[jj] = subwarp_reduce<1>(sum[jj]);
    for (int jj = 0; jj < j_upper; ++jj) {if(tid == 0 && (idx_c_base + jj) < cSize) c[idx_c_base + jj] = sum[jj];}
}

//matrix multiplication and add bias with shared memory and data-reuse across i dimension
__global__ void mm4d_pr4_bias_shared(float* a, float* b, float* c, float* bias, int d2a, int d2b, int d2c, int aSize, int bSize, int cSize) {
  extern __shared__ float s_b[]; //shared memory for matrix B

  int i, j;

    int tid = threadIdx.x; //thread id
  i = blockIdx.y* blockDim.y + threadIdx.y; //which mini-batch
    j = blockIdx.x*part; //which part of output result
    int j_upper = min(part, d2c-j); //last index of part

    //first element index of matrix A, matrix B and output
    int idx_a_base = i * d2a;
    int idx_b_base = j * d2b;
    int idx_c_base = i * d2c + j;

    //initialize variables to zero
    float a_value = 0.0f;
    float sum[part] = {0.0f};

    // load data to shared memory
    for (int kk = tid; kk < d2a; kk += 32) {
        int idx_b = idx_b_base + threadIdx.y * d2b + kk;
        if (idx_b < bSize) s_b[kk + threadIdx.y*d2b] = b[idx_b];
    }
    __syncthreads();

    //repeat on shared dimension
    for (int kk = tid; kk < d2a; kk += 32) {
      int idx_a = idx_a_base + kk; //matrix A index
      if (idx_a < aSize) a_value = a[idx_a]; //save in temporary variable for data-reuse
      //repeat on part of output column
      for (int jj = 0; jj < j_upper; ++jj) {
        sum[jj] += a_value * s_b[kk + jj*d2b];
      }
    }

    //reduce subwarp results and save to ouput matrix
    for (int jj = 0; jj < j_upper; ++jj) sum[jj] = subwarp_reduce<1>(sum[jj]);
    for (int jj = 0; jj < j_upper; ++jj) {if(tid == 0 && (idx_c_base + jj) < cSize) c[idx_c_base + jj] = sum[jj] + bias[j + jj];}
}

//CPU function for loading CUDA kernels
void linear_with_bias(array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output1, array1d_t<float>& bias, bool is_forward, bool is_grad_bias) {

  //data pointers and dimensions of input1, input2, and output matrix 
  float* bs = bias.data_ptr; float* a = input1.data_ptr; float* b = input2.data_ptr; float* c = output1.data_ptr;
  int d1c = output1.row_count, d2c = output1.col_count, d1a = input1.row_count, d2a = input1.col_count, d1b = input2.row_count, d2b = input2.col_count;
  int aSize = d1a*d2a, bSize = d1b*d2b, cSize = d1c*d2c;

  // block and gride size
  dim3 blocks(32, part);
  dim3 grids((d2c + part - 1)/part, (d1c + part - 1)/part);

  //run forward or backward kernel
  int sharedSize = part * d2b * sizeof(float);
  if (is_forward) mm4d_pr4_bias_shared<<<grids, blocks, sharedSize>>>(a, b, c, bs, d2a, d2b, d2c, aSize, bSize, cSize);
  else mm4d_pr4_shared<<<grids, blocks, sharedSize>>>(a, b, c, d2a, d2b, d2c, aSize, bSize, cSize);

  /*
  if (is_forward) mm4d_pr4_bias<<<grids, blocks>>>(a, b, c, bs, d2a, d2b, d2c, aSize, bSize, cSize);
  else mm4d_pr4<<<grids, blocks>>>(a, b, c, d2a, d2b, d2c, aSize, bSize, cSize);
  */

}
