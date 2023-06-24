#pragma once
#define MINIBATCH 8
#define UNROLL 8
#define YSTARTOP 20

namespace SNICIT_SDGC {

__global__ void post_spMM_GLARE(
    float *A, 
    float *C, 
    bool *ne_record,
    int *centroid_map,
    int *rowsY,
    float * __restrict__ B, 
    int* __restrict__ index, 
    bool* All32_0,
    bool* All32_1,
    int batch,
    float bias,
    int neuron
) {
    extern __shared__ float shared[];
    __shared__ bool thisAll32[16];


    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int rid = rowsY[blockIdx.y];
    int begin_idx = blockIdx.x * OUT_CHANNEL / 16 *  32;

    if(threadIdx.x < neuron / 1024) {
        thisAll32[threadIdx.x] = All32_0[(neuron / 1024)*rid+threadIdx.x];
        All32_0[(neuron / 1024)*rid+threadIdx.x] = true;
    }
    __syncthreads();

    if (centroid_map[rid] == -1) {
        float result = 0;
        int idx = begin_idx;
        for(int r = 0; r < 32/OUT_CHANNEL; r++) {
            int row_idx = index[idx + r*OUT_CHANNEL+threadIdx.x];  // check every?
            float val;
            if (thisAll32[row_idx/1024])
                val =32.0;
            else
                val = A[rid*neuron+row_idx];
            shared[r*OUT_CHANNEL+threadIdx.x] = val; 
        }
        __syncthreads();
        for(int r = 0; r < 32; ++r) {
            float val = shared[r];
            if (val != 0)
                result += val * B[(blockIdx.x * OUT_CHANNEL * 32) + r*OUT_CHANNEL+threadIdx.x];
        }
        float v = __ReLU(result+bias);

        if (v != 32 || !(thisAll32[tid/1024]))
            C[rid*neuron+(tid)] = v;
        if (v!=32) All32_1[tid/1024] = false;
    }
    else {
        if(threadIdx.x < neuron / 1024) {
            thisAll32[threadIdx.x] = All32_0[(neuron / 1024)*centroid_map[rid]+threadIdx.x];
        }
        __syncthreads();
        float result = 0;
        int idx = begin_idx;
        for(int r = 0; r < 32/OUT_CHANNEL; r++) {
            int row_idx = index[idx + r*OUT_CHANNEL+threadIdx.x];  // check every?
            float val;
            if (thisAll32[row_idx/1024])
                val =32.0;
            else
                val = A[centroid_map[rid]*neuron+row_idx];
            val += A[rid*neuron+row_idx];
            shared[r*OUT_CHANNEL+threadIdx.x] = val; 
        }
        __syncthreads();
        for(int r = 0; r < 32; ++r) {
            float val = shared[r];
            if (val != 0)
                result += val * B[(blockIdx.x * OUT_CHANNEL * 32) + r*OUT_CHANNEL+threadIdx.x];
        }
        C[rid*neuron+(tid)] = __ReLU(result+bias);
    }
}

__global__ void post_minus_GLARE(
    float *  A, 
    float *  C, 
    bool *ne_record,
    int *centroid_map,
    int *rowsY,
    int neuron, 
    int batch
) {

    int rid = rowsY[blockIdx.x];
    if (centroid_map[rid] == -1) {
        return;
    }
    int count = 0;
    for (int i = threadIdx.x; i < neuron; i += blockDim.x) {
        float wy_centroid = A[centroid_map[rid]*neuron + i];
        float wdelta_y = A[rid*neuron + i];
        float val = wdelta_y-wy_centroid;
        int cnt = __syncthreads_count(val != 0);
        count += cnt;
        C[rid * neuron+ i] = val;
    }
    
    if (threadIdx.x == 0) {
        if (count == 0) ne_record[rid] = false;
        else ne_record[rid] = true;
    }
};


}
