#pragma once
#define MINIBATCH 8
#define UNROLL 8
#define YSTARTOP 20

namespace GLARE {


__global__ void n16384l1_kernel_GC23(
	float * __restrict__ A, 
	float * __restrict__ B, 
	float * __restrict__ C, 
	int* __restrict__ index, 
    int* active,
	int batch, 
    int neuron, 
	float bias)     
{
	extern __shared__ float shared[];
	int start_idx = index[blockIdx.y];
	int col_gropu = threadIdx.x / 16;
	int last_load = ((neuron / 16) % 7) * 16 + 16;
	int load_num = (blockIdx.y + 1) == gridDim.y ? last_load : 128;
	for(int n = threadIdx.x; n < load_num; n += blockDim.x){
		for(int f = 0; f < MINIBATCH; ++f) {
			shared[f * 128 + n] = A[(blockIdx.x * MINIBATCH + f) * neuron + (start_idx + n) % neuron];
		}
	}
	__syncthreads();
	int last_thread = (neuron % 112);
	if(col_gropu == 7 || ((blockIdx.y + 1) == gridDim.y && threadIdx.x >= last_thread)) return;
    float res[MINIBATCH] = {0.0};
    for(int r = 0; r < 32; ++r) {
        float val = B[(blockIdx.y * 128 * 32) + r * 128 + threadIdx.x];
        int idx = col_gropu * 16 + r;
        for(int f = 0; f < MINIBATCH / UNROLL; ++f) {
            res[0 + f * UNROLL] += shared[(f * UNROLL + 0) * 128 + idx] * val;
            res[1 + f * UNROLL] += shared[(f * UNROLL + 1) * 128 + idx] * val;
            res[2 + f * UNROLL] += shared[(f * UNROLL + 2) * 128 + idx] * val;
            res[3 + f * UNROLL] += shared[(f * UNROLL + 3) * 128 + idx] * val;
            res[4 + f * UNROLL] += shared[(f * UNROLL + 4) * 128 + idx] * val;
            res[5 + f * UNROLL] += shared[(f * UNROLL + 5) * 128 + idx] * val;
            res[6 + f * UNROLL] += shared[(f * UNROLL + 6) * 128 + idx] * val;
            res[7 + f * UNROLL] += shared[(f * UNROLL + 7) * 128 + idx] * val;
        }
    }
    __syncthreads();
    for(int f = 0; f < MINIBATCH; ++f) {
        if(C[(blockIdx.x * MINIBATCH + f) * neuron + blockIdx.y * 112 + threadIdx.x] = __ReLU(res[f] + bias)) {
            active[blockIdx.x * MINIBATCH + f] = 1;
        }
    }
}


__global__ void _non_empty_rows_kernel(int *buf, int *res, int N, int neth) {
    int global_id = blockDim.x * blockIdx.x+threadIdx.x; 
    if (global_id < N) {
        if ((global_id == 0 && buf[global_id] > 0) ||
         (global_id > 0 && buf[global_id] != buf[global_id-1])) {
            res[buf[global_id]-1] = global_id;
        }
    }
    if (global_id >= neth) {
        res[global_id] = 0;
    }

}

__global__ void post_spMM_reduce(
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
    // for(int r = 0; r < 32/OUT_CHANNEL; r++) {
    //     int row_idx = index[idx + r*OUT_CHANNEL+threadIdx.x];  // check every?
    //     shared[r*OUT_CHANNEL+threadIdx.x] = A[rid*neuron+row_idx]; // Y_star[row_idx];
    //     // result += 32 * B[(tid * 32) + r];
    // }
    __syncthreads();
    for(int r = 0; r < 32; ++r) {
        float val = 32.0;
        result += val * B[(blockIdx.x * OUT_CHANNEL * 32) + r*OUT_CHANNEL+threadIdx.x];
        // if (blockIdx.y == 1 && blockIdx.x * blockDim.x + threadIdx.x == 0)
        //     printf("\n");
    }
    // C[rid*neuron+(tid)] = result;
}

__global__ void post_spMM_GC23(
    float *A, 
    float *C, 
    float * __restrict__ B, 
    int* __restrict__ index, 
    int batch,
    int neuron
) {
    extern __shared__ float shared[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int rid = blockIdx.y;
    int begin_idx = blockIdx.x * OUT_CHANNEL / 16 *  32;
    
    float result = 0;
    int idx = begin_idx;
    if (ne_record[rid]){
        for(int r = 0; r < 32/OUT_CHANNEL; r++) {
            int row_idx = index[idx + r*OUT_CHANNEL+threadIdx.x];  // check every?
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
}




__global__ void post_minus_GC23(
    float *  A, 
    float *  C, 
    bool *ne_record,
    int *centroid_map,
    int neuron, 
    float bias,
    int batch
) {
    int rid = blockIdx.x;
    if (ne_record[rid]){
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
    }
    else return;
};

}
