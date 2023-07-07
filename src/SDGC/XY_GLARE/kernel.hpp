#pragma once

namespace GLARE{

#define OUT_CHANNEL 16
__global__ void n16384_l11_kernel_GLARE(
    float * __restrict__ A, 
    float * __restrict__ B, 
    float * __restrict__ C, 
    int* __restrict__ index, 
    int* __restrict__ active,
    bool* All32_0,
    bool* All32_1,
    int batch, 
    int neuron, 
    float bias,
    int partition
) {
    
    extern __shared__ float shared[];
    __shared__ bool thisAll32[64];

    for(int n = threadIdx.x; n < OUT_CHANNEL * 32; n += blockDim.x){
        shared[n] = B[(blockIdx.y * OUT_CHANNEL * 32) + n];
    }
    if(threadIdx.x < neuron / partition) {
        thisAll32[threadIdx.x] = All32_0[(neuron / partition)*blockIdx.x+threadIdx.x];
    }

    __syncthreads();

    if((blockIdx.x * blockDim.x + threadIdx.x) >= batch) return;
    int begin_idx = blockIdx.y * OUT_CHANNEL / 16 * 32;
    int count = 0;
    for(int o_r = 0; o_r < OUT_CHANNEL / 16; ++o_r) {
        float reduce[16] = {0.0};
        int idx = begin_idx + o_r * 32;
        for(int r = 0; r < 32; ++r) {
            int row_idx = index[idx + r];  // check every?
            float val;
            if (thisAll32[row_idx / partition]) { // thisAll32[row_idx / 1024]
                val = 32.0;
            }
            else 
            {
                val = A[row_idx * batch + blockIdx.x * blockDim.x + threadIdx.x];
            }
            for(int c = 0; c < 16; ++c) {
                reduce[c] += val * shared[o_r * 32 * 16 + r * 16 + c];
            }
        }
        for(int c = 0; c < 16; ++c) {
            float v = __ReLU(reduce[c] + bias);
            if (v != 0) {
                active[blockIdx.x * blockDim.x + threadIdx.x] = 1;
            }
            if (v!=32) { //  || !(thisAll32[(blockIdx.y * OUT_CHANNEL  + o_r * 16 + c) / 1024])
                C[(blockIdx.y * OUT_CHANNEL  + o_r * 16 + c) * batch + blockIdx.x * blockDim.x + threadIdx.x] = v;
            }
            count += (v == 32.0);
        }
    }
    if (count < OUT_CHANNEL) {
        All32_1[(neuron / partition)*blockIdx.x+(blockIdx.y*OUT_CHANNEL)/partition] = false;
    }
}

}