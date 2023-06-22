#pragma once

namespace SNICIT_SDGC {

__global__ 
void snig_inferenceAug(
  const float* Y_0,
  const bool* is_nonzero_row_0,
  const size_t sec_size,
  const size_t num_sec,
  const size_t num_neurons,
  const bool* All32_0,
  bool* All32_1,
  const int* col_w,
  const int* row_w,
  const float* val_w,
  const float bias,
  bool* is_nonzero_row_1,
  float* Y_1
);

//-----------------------------------------------------------------------------
//Definition of kernel function
//-----------------------------------------------------------------------------

__global__ 
void snig_inferenceAug(
  const float* Y_0,
  const bool* is_nonzero_row_0,
  const size_t sec_size,
  const size_t num_secs,
  const size_t num_neurons,
  const bool* All32_0,
  bool* All32_1,
  const int* col_w,
  const int* row_w,
  const float* val_w,
  const float bias,
  bool* is_nonzero_row_1,
  float* Y_1
) {
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  __shared__ bool thisAll32, thisAll32Loc, nextAll32[2];
  //r = blockIdx.x
  //s_o = blockIdx.y
  int num_threads = blockDim.x * blockDim.y;
  //num_secs is small enough to compute by each single thread
  bool is_all_zero = true;
  for(size_t s_i = 0; s_i < num_secs; ++s_i) {
    is_all_zero &= !is_nonzero_row_0[blockIdx.x * num_secs + s_i];
  }

  if(is_all_zero) {
    //incremental memory resetting
    //avoid calling cudaMemset
    if(is_nonzero_row_1[blockIdx.x * num_secs + blockIdx.y]) {
      for(size_t j = tid; j < sec_size; j += num_threads) {
        Y_1[blockIdx.x * num_neurons + blockIdx.y * sec_size + j] = 0;
      }
      __syncthreads();
      if(tid == 0) {
        is_nonzero_row_1[blockIdx.x * num_secs + blockIdx.y] = false;
      } 
    }
    return;
  }

  //forward feeding
  extern __shared__ float results[];

  //set results to bias directly
  for(size_t k = tid; k < sec_size; k += num_threads) {
    results[k] = bias;  
  }

  //use bool array size of 2 (is_nonzero) in share memory to avoid synchronization
  //is_nonzero[1] represents whether this row is nonzero
  //if is_nonzero[1] is true, this row is nonzero
  __shared__ bool is_nonzero[2];
  if(tid == 0) {
    is_nonzero[1] = false;
    nextAll32[1] = false;
  }
  __syncthreads();

  for(size_t s_i = 0; s_i < num_secs; ++s_i) {
    if (tid == 0) {
      thisAll32Loc = All32_0[blockIdx.x*num_secs+s_i];
      if (s_i == blockIdx.y) {
        thisAll32 = thisAll32Loc;
      }
    }
    __syncthreads();
    if(!is_nonzero_row_0[blockIdx.x * num_secs + s_i]) {
      continue;
    }
    for(size_t j = threadIdx.y + s_i * sec_size; j < (s_i + 1) * sec_size; j += blockDim.y) {
      float valY;
      if (thisAll32Loc) {
        valY = 32.0;
      }
      else{
        valY = Y_0[blockIdx.x * num_neurons + j];
      }
      // valY = Y_0[blockIdx.x * num_neurons + j];
      if(valY == 0) {
        continue;
      }
      int beg_w = col_w[blockIdx.y * num_neurons + j] + threadIdx.x;
      int end_w = col_w[blockIdx.y * num_neurons + j + 1];
      for(int k = beg_w; k < end_w; k += blockDim.x) {
        int roww = row_w[k];
        float valw = val_w[k];
        atomicAdd(&results[roww - blockIdx.y * sec_size], valY * valw);
      }
    }
  }
  __syncthreads();
  for(size_t i = tid; i < sec_size; i += num_threads) {
    float v = min(float(32), max(results[i], float(0)));
    if (v != 32.0 || !thisAll32) { //  && !thisAll32
      Y_1[blockIdx.x * num_neurons + blockIdx.y * sec_size + i] = v;
    }

    is_nonzero[v != 0] = true;
    nextAll32[v != 32] = true;
  }

  //if one thread sets is_nonzero[1] to true
  //meaning this row is nonzero
  //toggle is_nonzero_row_1[this row] to true
  __syncthreads();
  if(tid == 0) {
    is_nonzero_row_1[blockIdx.x * num_secs + blockIdx.y] = is_nonzero[1];
    All32_1[blockIdx.x* num_secs+blockIdx.y] = !(nextAll32[1]);
  }
}

}// end of namespace snig ----------------------------------------------
