#pragma once
#define MINIBATCH 8
#define UNROLL 8
#define YSTARTOP 20

namespace SNICIT_SDGC {

__global__ void post_spMM(
    float *A, 
    float *C, 
    int *rowsY,
    float * __restrict__ B, 
    int* __restrict__ index, 
    int batch,
    int neuron
) {
    extern __shared__ float shared[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int rid = rowsY[blockIdx.y];
    int begin_idx = blockIdx.x * OUT_CHANNEL / 16 *  32;
    
    float result = 0;
    int idx = begin_idx;
    for(int r = 0; r < 32/OUT_CHANNEL; r++) {
        int row_idx = index[idx + r*OUT_CHANNEL+threadIdx.x];  // check every?
        if (centroid_map[rid] == -1)
        shared[r*OUT_CHANNEL+threadIdx.x] = A[rid*neuron+row_idx]; // Y_star[row_idx];
        // result += 32 * B[(tid * 32) + r];
    }
    __syncthreads();
    for(int r = 0; r < 32; ++r) {
        float val = shared[r];
        if (val != 0)
            result += val * B[(blockIdx.x * OUT_CHANNEL * 32) + r*OUT_CHANNEL+threadIdx.x];
        // if (blockIdx.y == 1 && blockIdx.x * blockDim.x + threadIdx.x == 0)
        //     printf("\n");
    }
    C[rid*neuron+(tid)] = result;
}

__global__ void post_minus(
    float *  A, 
    float *  C, 
    bool *ne_record,
    int *centroid_map,
    int *rowsY,
    int neuron, 
    float bias,
    int batch
) {
    int rid = rowsY[blockIdx.x];
    if (centroid_map[rid] == -1) {
        for (int i = threadIdx.x; i < neuron; i += blockDim.x) {
            C[rid*neuron + i] = __ReLU(A[rid*neuron + i] + bias); //Y0[rid * neurons+tid];
        }
        ne_record[rid] = true;
        return;
    }
    int count = 0;
    for (int i = threadIdx.x; i < neuron; i += blockDim.x) {
        float wy_centroid = A[centroid_map[rid]*neuron + i];
        float wdelta_y = A[rid*neuron + i];
        float val = __ReLU(wy_centroid+bias+wdelta_y)-__ReLU(wy_centroid+bias);
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
