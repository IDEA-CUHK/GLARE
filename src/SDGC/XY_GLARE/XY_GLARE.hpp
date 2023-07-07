#pragma once

#include <string>
#include <utility.hpp>
#include <inspector/header.h>
#include <reorder/header.h>
#include <gpu_lib/header.h>
#include <XY/kernel.hpp>
#include <XY_GLARE/kernel.hpp>

#include <cuda.h>
#include <cstdio>
#include <cstdlib>

namespace GLARE {   

class XY_GLARE{

private:

    float _bias;
    std::string _weight_path;
    int _layer;
    int _neuron;
    int _threshold;
    std::vector<std::vector<float>> _input;

    void _dense_reorder(std::vector<std::vector<float>> &input, Reorder &reorder_class);

    int  _infer(    
        std::vector<std::vector<float>> &input,
        std::vector<std::vector<float>> &weight, 
        std::vector<std::vector<int>> &row_access, 
        int batch, 
        int neuron, 
        float bias,
        GpuEnv &env
    );

public:

    XY_GLARE(
        const std::string& weight_path,
        const float bias = -.3f,
        const size_t num_neurons_per_layer = 1024,
        const size_t num_layers = 120,
        const int threshold = 30
    );

    void infer(
        const std::string& input_path,
        const std::string& golden_path,
        const size_t num_inputs,
        const size_t batch
    );


};


XY_GLARE::XY_GLARE(
  const std::string& weight_path,
  const float bias,
  const size_t num_neurons_per_layer,
  const size_t num_layers,
  const int threshold
):
  _weight_path(weight_path), _bias(bias), _neuron(num_neurons_per_layer), _layer(num_layers), _threshold(threshold)
{
  std::cout<<("Constructing XY_GLARE method......")<<std::endl;
}


void XY_GLARE::_dense_reorder(std::vector<std::vector<float>> &input, Reorder &reorder_class) {
    for(int i = 0; i < input.size(); ++i) {
        std::vector<float> tmp(input[i].size());
        for(int j = 0; j < input[i].size(); ++j) {
            auto new_j = reorder_class.reorder(j);
            tmp[new_j] = input[i][j];
        }
        input[i] = tmp;
    }
}


void XY_GLARE::infer(
  const std::string& input_path,
  const std::string& golden_path,
  const size_t num_inputs,
  const size_t batch
) {
    std::vector<std::vector<float>> _input(batch, std::vector<float>(_neuron));
    std::vector<std::vector<float>> weight; 
    std::vector<std::vector<int>> row_access; 

    int feature = 0;

    std::map<int, int> hash_map = {
        {65536, 4096},
        {16384, 1024},
        {4096, 256},
        {1024, 64}
    };

    std::map<int, float> type_1 = {
        {65536, 12},
        {16384, 10},
        {4096, 8},
        {1024, 6}
    };

    std::map <int, int> feature_golden = {
        {1024, 1812},
        {4096, 1801},
        {16384, 1918},
        {65536, 1994}
    };


    for (int offset = 0; offset < 60000; offset += batch) {
        std::cout << "[BEGIN for round "<< std::to_string(offset / batch) <<"]..." << std::endl;
        read_input_xy(input_path, _input, _neuron, batch, offset);
        std::cout << "--------------------" << std::endl;
        HashReorder hash_reorder_t(hash_map[_neuron], _neuron);
        _dense_reorder(_input, hash_reorder_t);
        if (offset == 0) {
            std::filesystem::path filepath = std::string("../scheduled_bm/neuron"+std::to_string(_neuron)+"/");
            bool filepathExists = std::filesystem::is_directory(filepath);
            if (filepathExists == false && _layer == 1920) {
                for(int l = 0; l < _layer; ++l) {
                    std::filesystem::create_directory("../scheduled_bm/neuron"+std::to_string(_neuron)+"/");
                    std::string weight_file = _weight_path;
                    weight_file += "/n" + std::to_string(_neuron) + "-l"
                        + std::to_string(l + 1) + ".tsv";
                    COOMatrix coo(weight_file, 1, false);
                    if (l % 100 == 0)
                        std::cout << "["<< weight_file << "] to COO success!" << std::endl;
                    coo.reorder(hash_reorder_t);
                    CSRCSCMatrix csr_csc(coo);
                    csr_csc.transpose();
                    BlockContainer blocks(csr_csc, SparseMatrixBlockGen::naive_method);
                    MaxInReuseBSchedule schedule(blocks);
                    if(l == 0) {
                        schedule.schedule(16, 7);
                    }
                    else if(l < type_1[_neuron]) {
                        schedule.schedule_output_parallel(128, 1, false);
                    }        
                    else {
                        schedule.schedule(128, 1);
                    }
                    auto data = schedule.get_data(_neuron);
                    weight.push_back(data.value);
                    row_access.push_back(data.row_access);
                    std::ofstream weight_output_file("../scheduled_bm/neuron"+std::to_string(_neuron)+"/n"+std::to_string(_neuron)+"-weight-l"+std::to_string(l)+".tsv");
                    for (const auto &e : data.value) weight_output_file << e << "\n";

                    std::ofstream row_output_file("../scheduled_bm/neuron"+std::to_string(_neuron)+"/n"+std::to_string(_neuron)+"-row-l"+std::to_string(l)+".tsv");
                    for (const auto &e : data.row_access) row_output_file << e << "\n";
                }
            }
            else {
                for(int l = 0; l < _layer; ++l) {
                    std::ifstream weight_input_file("../scheduled_bm/neuron"+std::to_string(_neuron)+"/n"+std::to_string(_neuron)+"-weight-l"+std::to_string(l)+".tsv");
                    std::ifstream row_input_file("../scheduled_bm/neuron"+std::to_string(_neuron)+"/n"+std::to_string(_neuron)+"-row-l"+std::to_string(l)+".tsv");
                    if(!weight_input_file || !row_input_file) {
                        std::cout << "File:" << "n"+std::to_string(_neuron)+"-l"+std::to_string(l)+".tsv" << " does not exists.\n";
                        exit(-1);
                    }
                    float val;
                    int row_idx;
                    std::vector<float> weight_layer; 
                    std::vector<int> row_access_layer;
                    while(weight_input_file >> val) {
                        weight_layer.push_back(val);
                    }
                    while(row_input_file >> row_idx) {
                        row_access_layer.push_back(row_idx);
                    }
                    weight.push_back(weight_layer);
                    row_access.push_back(row_access_layer);
                    if (l % 100 == 0)
                        std::cout << "layer " + std::to_string(l) + " read succ" << std::endl;
                }
            }
        }
        GpuEnv env(0);
        std::cout << "==========[Champion XY_GLARE-2021]============ " << std::endl;
        feature += _infer(_input, weight, row_access, batch, _neuron, _bias, env);
        std::cout << "[END for round "<< std::to_string(offset / batch) <<"]..." << std::endl;

        for (int destr_iter = 0; destr_iter < batch; destr_iter++) {
            std::fill(_input[destr_iter].begin(), _input[destr_iter].end(), 0);
        }
        
    }

    if (feature == feature_golden[_neuron]) {
        std::cout << "CHALLENGE PASSED\n";
    }
    else {
        std::cout << "CHALLENGE FAILED\n";
    }

}

int XY_GLARE::_infer(
    std::vector<std::vector<float>> &input,
    std::vector<std::vector<float>> &weight, 
    std::vector<std::vector<int>> &row_access, 
    int batch, 
    int neuron, 
    float bias,
    GpuEnv &env
) {

	float *A;
    float *A_d;
    
    float *A_T;

	float *C;
    float *C_d;

    float **B;
    float **B_d;
	int **index;
    int **index_d;

    int *category;
    int *active;
    int *old_to_new_map;
    int *category_d;
    int *active_d;
    int *old_to_new_map_d;
    std::map<int, int> partition_map = {
        {1024, 1024},
        {4096, 4096}, // 4096
        {16384, 2048},
        {65536, 1024}
    };
    int partition = partition_map[neuron];
    int this_round_batch = batch;
    int layer = weight.size();

    A = (float*)malloc(sizeof(float) * neuron * batch);
    C = (float*)malloc(sizeof(float) * neuron * batch);
    memset(C, 0, sizeof(float) * neuron * batch);

    for(int l = 0; l < input.size(); ++l) {
        for(int i = 0; i < input[l].size(); ++i) {
            A[l * neuron + i] = input[l][i];
        }
    }

    B = (float**) malloc(sizeof(float*) * weight.size());
    B_d = (float**) malloc(sizeof(float*) * weight.size());
    for(int l = 0; l < weight.size(); ++l) {
        B[l] = (float*) malloc(sizeof(float*) * weight[l].size());
        for(int i = 0; i < weight[l].size(); ++i) {
            B[l][i] = weight[l][i];
        }
    }

    index = (int**) malloc(sizeof(int*) * row_access.size());
    index_d = (int**) malloc(sizeof(int*) * row_access.size());
    for(int l = 0; l < row_access.size(); ++l) {
        index[l] = (int*) malloc(sizeof(int*) * row_access[l].size());
        for(int i = 0; i < row_access[l].size(); ++i) {
            index[l][i] = row_access[l][i];
        }
    }

    category = (int*) malloc(sizeof(int*) * batch);
    for(int i = 0; i < batch; ++i) {
        category[i] = i;
    }

    old_to_new_map = (int*) malloc(sizeof(int*) * batch);
    for(int i = 0; i < batch; ++i) {
        old_to_new_map[i] = -1;
    }

    active = (int*) malloc(sizeof(int*) * batch);
    for(int i = 0; i < batch; ++i){
        active[i] = 0;
    }

    Safe_Call(cudaMalloc((void**)&A_d, sizeof(float) * neuron * batch));
    Safe_Call(cudaMemcpy(A_d, A, sizeof(float) * neuron * batch, cudaMemcpyHostToDevice));


    Safe_Call(cudaMalloc((void**)&A_T, sizeof(float) * neuron * batch));
    Safe_Call(cudaMemset(A_T, 0, sizeof(float) * neuron * batch));

    Safe_Call(cudaMalloc((void**)&C_d, sizeof(float) * neuron * batch));
    Safe_Call(cudaMemset(C_d, 0, sizeof(float) * neuron * batch));

    Safe_Call(cudaMalloc((void**)&active_d, sizeof(int) * batch));
    Safe_Call(cudaMalloc((void**)&category_d, sizeof(int) * batch));
    Safe_Call(cudaMalloc((void**)&old_to_new_map_d, sizeof(int) * batch));

    bool *All32_last;
    bool *All32_next;

    for(int l = 0; l < layer; ++l) {
        Safe_Call(cudaMalloc((void**)&(B_d[l]), sizeof(float) * weight[l].size()));
        Safe_Call(cudaMemcpy(B_d[l], B[l], sizeof(float) * weight[l].size(), cudaMemcpyHostToDevice));

        Safe_Call(cudaMalloc((void**)&(index_d[l]), sizeof(float) * row_access[l].size()));
        Safe_Call(cudaMemcpy(index_d[l], index[l], sizeof(float) * row_access[l].size(), cudaMemcpyHostToDevice));
    }

    float all_time = 0;
    float stat_time = 0;
    int post_conv_cnt = 0;
    env.add_event("XY_GLARE-2021");
    

    std::map<int, int> neuron_map = {
        {1024, 6},
        {4096, 8},
        {16384, 10},
        {65536, 12}
    };
    std::map<int, int> stride_map = {
        {1, 16},
        {2, 32},
        {3, 64},
        {4, 128},
        {5, 256},
        {6, 512},
        {7, 1024},
        {8, 2048},
        {9, 4096},
        {10, 8192},
        {11, 16384},
        {12, 32768}
    };



    bool now_transpose = false;
    int last_feature = batch;
    int transpose_batch = 0;
    int feature = 0;
    for(int l = 0; l < layer; ++l) {
        auto stream = env.get_stream("XY_GLARE-2021");
        env.event_start_record("XY_GLARE-2021");
        Safe_Call(cudaMemsetAsync(active_d, 0, sizeof(int) * batch, stream));

        if(l == 0) {
            int blocksize = 128;
            dim3 block(blocksize);
            dim3 grid((this_round_batch + MINIBATCH - 1)/ MINIBATCH, (neuron + 112 - 1) / 112);
            n16384l1_kernel<<<grid, block, sizeof(float) * (MINIBATCH * (128 + 16)), stream>>>(
                A_d, B_d[l], C_d, index_d[l], category_d, active_d, this_round_batch, neuron, bias
            );
            cudaError_t err = cudaGetLastError();        
            if (err != cudaSuccess) {
                printf("what CUDA Error: %s\n", cudaGetErrorString(err));
                exit(-1);
            }
        }
        else if(l <= neuron_map[neuron] - 1){
            int blocksize = 128;
            dim3 block(blocksize);
            dim3 grid((this_round_batch + MINIBATCH - 1)/ MINIBATCH, (neuron + blocksize - 1) / blocksize);
            int stride = stride_map[l + 1];
            int load_num = stride > blocksize ? 32 * (blocksize / 16) : stride + 16 * (blocksize / 16);
            int shared_size = ((load_num + 31) / 32) * 32;
        	n16384_l2_l11_kernel<<<grid, block, sizeof(float) * (MINIBATCH * shared_size), stream>>>(
                A_d, B_d[l], C_d, category_d, active_d, stride, this_round_batch, neuron, bias
            );
            cudaError_t err = cudaGetLastError();        
            if (err != cudaSuccess) {
                printf("what CUDA Error: %s\n", cudaGetErrorString(err));
                exit(-1);
            }
        }
        else {
            if(!now_transpose) {
                transpose_batch = last_feature;
                now_transpose = true;
                dim3 grid((neuron + TILE_DIM - 1) / TILE_DIM, (transpose_batch +  TILE_DIM - 1) / TILE_DIM);
                dim3 block(TILE_DIM, BLOCK_ROWS);
                matrix_transpose<<<grid, block, sizeof(float) * (TILE_DIM * TILE_DIM + TILE_DIM), 
                    stream>>>(
                        A_T, A_d, neuron, transpose_batch
                );
                cudaError_t err = cudaGetLastError();        
   	            if (err != cudaSuccess) {
		            printf("what CUDA Error: %s\n", cudaGetErrorString(err));
      	            exit(-1);
   	            }
            }
            if(l == 22) {
                dim3 grid((transpose_batch + TILE_DIM - 1) / TILE_DIM, (neuron +  TILE_DIM - 1) / TILE_DIM);
                dim3 block(TILE_DIM, BLOCK_ROWS);
                matrix_re_transpose_and_delete<<<grid, block, sizeof(float) * (TILE_DIM * TILE_DIM + TILE_DIM), 
                    stream>>>(
                        A_d, A_T, old_to_new_map_d, neuron, transpose_batch
                );
                Safe_Call(cudaStreamSynchronize(stream));
                cudaError_t err = cudaGetLastError();        
   	            if (err != cudaSuccess) {
		            printf("what CUDA Error: %s\n", cudaGetErrorString(err));
      	            exit(-1);
   	            }

                dim3 grid2((neuron + TILE_DIM - 1) / TILE_DIM, (this_round_batch +  TILE_DIM - 1) / TILE_DIM);
                dim3 block2(TILE_DIM, BLOCK_ROWS);
                matrix_transpose<<<grid2, block2, sizeof(float) * (TILE_DIM * TILE_DIM + TILE_DIM), 
                    stream>>>(
                        A_T, A_d, neuron, this_round_batch
                );
                // A_T = A_d;
                Safe_Call(cudaStreamSynchronize(stream));
                err = cudaGetLastError();        
                if (err != cudaSuccess) {
                 printf("what CUDA Error: %s\n", cudaGetErrorString(err));
                   exit(-1);
                }
                transpose_batch = this_round_batch;

                Safe_Call(cudaMallocManaged((void**)&All32_last, sizeof(bool) * ((transpose_batch + 256 - 1) / 256) * (neuron/partition)));
                Safe_Call(cudaMallocManaged((void**)&All32_next, sizeof(bool) * ((transpose_batch + 256 - 1) / 256) * (neuron/partition)));
                Safe_Call(cudaMemset(All32_last, 0, sizeof(bool) * ((transpose_batch + 256 - 1) / 256) * (neuron/partition)));
                Safe_Call(cudaMemset(All32_next, 1, sizeof(bool) * ((transpose_batch + 256 - 1) / 256) * (neuron/partition)));

                Safe_Call(cudaMemcpy(C_d, A_T, sizeof(float) * transpose_batch * neuron, cudaMemcpyDeviceToDevice));
            }
            int blocksizex = 256;
            int blocksizey = 1;
            dim3 block(blocksizex);
            dim3 grid((transpose_batch + blocksizex - 1) / blocksizex,  neuron / (OUT_CHANNEL));
            if (l >= 22) {
                n16384_l11_kernel_GLARE<<<grid, block, sizeof(float) * (OUT_CHANNEL * 32), stream>>>(
                    A_T, B_d[l], C_d, index_d[l], active_d, All32_last, All32_next, transpose_batch, neuron, bias, partition
                );
                Safe_Call(cudaStreamSynchronize(stream));
                bool *tmp = All32_last;
                All32_last = All32_next;
                All32_next = tmp;
                Safe_Call(cudaMemset(All32_next, 1, sizeof(bool) * ((transpose_batch + 256 - 1) / 256) * (neuron/partition)));
                // n16384_l11_kernel<<<grid, block, sizeof(float) * (OUT_CHANNEL * 32), stream>>>(
                //     A_T, B_d[l], C_d, index_d[l], active_d, transpose_batch, neuron, bias
                // );
                // Safe_Call(cudaStreamSynchronize(stream));
                // printf("\n=============================\n");
            }
            else {
                n16384_l11_kernel<<<grid, block, sizeof(float) * (OUT_CHANNEL * 32), stream>>>(
                    A_T, B_d[l], C_d, index_d[l], active_d, transpose_batch, neuron, bias
                );
                Safe_Call(cudaStreamSynchronize(stream));
            }
            // n16384_l11_kernel_Aug<<<grid, block, sizeof(float) * (OUT_CHANNEL * 32), stream>>>(
            //     A_T, B_d[l], C_d, index_d[l], active_d, All32_last, All32_next, transpose_batch, neuron, bias
            // );
            // Safe_Call(cudaStreamSynchronize(stream));
            cudaError_t err = cudaGetLastError();        
            if (err != cudaSuccess) {
                printf("what CUDA Error: %s\n", cudaGetErrorString(err));
                exit(-1);
            }

        }

        if(l > neuron_map[neuron] - 1) {
            Safe_Call(cudaMemcpyAsync(active, active_d, sizeof(int) * transpose_batch, cudaMemcpyDeviceToHost, stream));
        }
        else {
            Safe_Call(cudaMemcpyAsync(active, active_d, sizeof(int) * this_round_batch, cudaMemcpyDeviceToHost, stream));
        }
        Safe_Call(cudaStreamSynchronize(stream));

        feature = 0;
        if(l <= neuron_map[neuron] - 1) { 
            for(int k = 0; k < this_round_batch; ++k) {
                if(active[k]) {
                    // category[feature] = category[k];
                    category[feature] = k;
                    feature++;
                }
            }
            float* tmp = A_d;
            A_d = C_d;
            C_d = tmp;
        }   
        else if(l == 21) {
            int neg_1 = 0;
            int have_v = 0;
            for(int k = 0; k < transpose_batch; ++k) {
                if(active[k]) {
                    old_to_new_map[k] = feature;
                    feature++;
                    have_v++;
                }
                else {
                    old_to_new_map[k] = -1;
                    neg_1++;
                }
            }
            float* tmp = A_T;
            A_T = C_d;
            C_d = tmp;
        }   
        else {
            for(int k = 0; k < batch; ++k) {
                if(active[k]) {
                    // category[feature] = category[k];
                    category[feature] = k;
                    feature++;
                }
            }
            float* tmp = A_T;
            A_T = C_d;
            C_d = tmp;
        }

        for(int i = 0; i < batch; ++i){
            active[i] = 0;
        }
        
        last_feature = this_round_batch;
        this_round_batch = feature;
        // printf("layer=%d, feature=%d\n", l, feature);

        Safe_Call(cudaMemcpyAsync(category_d, category, sizeof(int) * feature, cudaMemcpyHostToDevice, stream));

        if(l == 21)
            Safe_Call(cudaMemcpyAsync(old_to_new_map_d, old_to_new_map, sizeof(int) * transpose_batch, cudaMemcpyHostToDevice, stream));
        env.event_stop_record("XY_GLARE-2021-kernel");
        float time = env.get_event_time("XY_GLARE-2021-kernel"); 
        if (l >= _threshold) {
            stat_time += time;
            post_conv_cnt++;
        }
        all_time += time;
    }

	Safe_Call(cudaMemcpy(C, C_d, sizeof(float) * neuron * batch, cudaMemcpyDeviceToHost));
	std::cout << "XY_GLARE info: runtime " << all_time <<  "ms" << 
    " avgpost " << stat_time/post_conv_cnt <<  "ms" <<std::endl;

    delete [] A;
    Safe_Call(cudaFree(A_d));
    Safe_Call(cudaFree(A_T));
    delete [] C;
    Safe_Call(cudaFree(C_d));


    for(int l = 0; l < weight.size(); ++l) {
        delete [] B[l];
        delete [] index[l];
        Safe_Call(cudaFree(B_d[l]));
        Safe_Call(cudaFree(index_d[l]));
    }
    delete [] B;
    delete [] B_d;
    delete [] index;
    delete [] index_d;
    delete [] category;
    delete [] old_to_new_map;
    delete [] active;

    Safe_Call(cudaFree(active_d));
    Safe_Call(cudaFree(category_d));
    Safe_Call(cudaFree(old_to_new_map_d));

    return feature;
}


}