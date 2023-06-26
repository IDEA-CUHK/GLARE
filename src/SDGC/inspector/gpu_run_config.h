#pragma once

namespace GLARE {
    class GpuRunConfig {
    public:
        int block_num;
        int thread_num;
        int shared_memory_size;
    };

};