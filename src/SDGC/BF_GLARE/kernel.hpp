#pragma once

namespace GLARE{

__global__ 
void bf_inference_GLARE(
  const float* Y0,
  const size_t nerowsY,
  const int* rowsY0,
  int* rlenY0,
  const size_t COL_BLK,
  const size_t N_SLAB,
  const size_t num_neurons_per_layer,
  const bool* All32_0,
  bool* All32_1,
  const int* roffW,
  const int* colsW,
  const float* valsW,
  const float bias,
  float* Y1,
  int* rlenY1
);

//-----------------------------------------------------------------------------
//Definition of task function
//-----------------------------------------------------------------------------

__global__ 
void bf_inference_GLARE(
  const float* Y0,
  const size_t nerowsY,
  const int* rowsY0,
  int* rlenY0,
  const size_t COL_BLK,
  const size_t N_SLAB,
  const size_t num_neurons_per_layer,
  const bool* All32_0,
  bool* All32_1,
  const int* roffW,
  const int* colsW,
  const float* valsW,
  const float bias,
  float* Y1,
  int* rlenY1
) {

  if(blockIdx.x >= nerowsY) {
    return;
  }

  extern  __shared__ float shRow[];
  __shared__ bool thisAll32;

  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int rid = rowsY0[blockIdx.x];
  __syncthreads();
  if(tid == 0) {
    rlenY0[rid] = 0;
    rlenY1[rid] = 0;
  }


  for(size_t i = 0; i < N_SLAB; i++) {
    if (tid == 0) {
        thisAll32 = All32_0[rid*N_SLAB+i];
    }
    __syncthreads();
    for(size_t j = threadIdx.x; j < COL_BLK; j++) {
      shRow[j] = 0;  
    }
    __syncthreads();
    for(size_t j = threadIdx.y; j < num_neurons_per_layer; j += blockDim.y) {
      float valY;
      if (thisAll32) {
        valY = 32.0;
      }
      else {
        valY = Y0[rid * num_neurons_per_layer + j];
      }
       
      if(valY == 0) {
        continue;
      }
      int begOffW = roffW[i * num_neurons_per_layer + j] + threadIdx.x;
      int endOffW = roffW[i * num_neurons_per_layer + j + 1];
      for(int k = begOffW; k < endOffW; k += blockDim.x) {
        int colW = colsW[k];
        float valW = valsW[k];
        atomicAdd(&shRow[colW - i * COL_BLK], valY * valW);
      }
    }
    __syncthreads();
    int count = 0;
    int count32 = 0;

    for(size_t j = 0; j < COL_BLK; j += blockDim.x * blockDim.y) {
      int localcount32 = 0;
      float v = j + tid < COL_BLK ? shRow[j + tid] + bias : -1;
      count += __syncthreads_count(v > 0);
      localcount32 = __syncthreads_count(v >= 32);
      if(j + tid < COL_BLK) {
        if (v < 32 || !thisAll32) {
          Y1[rid * num_neurons_per_layer + i * COL_BLK + j + tid] =  min(float(32), max(float(0), v));
        }
        count32 += localcount32;
      }
    }
    if(tid == 0) {
      rlenY1[rid] += count;
      if (count32 >= COL_BLK)
        All32_1[rid*N_SLAB+i] = true;
    }
  }
}

}// end of namespace snig ----------------------------------------------
