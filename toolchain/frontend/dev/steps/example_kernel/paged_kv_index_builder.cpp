/**
 * @file paged_kv_index_builder.cpp
 * @brief PagedKVIndexBuilder 各类计算内核的实现
 * 
 * 该内核接收 KVCacheData，提取其中的 K 缓存（L×D 矩阵），
 * 每 page_size_ 个 token 构成一页，对页内的 Key 向量进行平均池化，
 * 再使用权重矩阵进行线性投影。将各页生成的索引向量按页排列，
 * 构成 FlatIndexMemory。
 */

#include "../build_memory.h"
#include <iostream>
#include <cmath>

namespace heteromm {
namespace step {

void PagedKVIndexBuilder::run_test_kernel(
    const data_type::KVCacheData<float>& raw_data,
    data_type::FlatIndexMemory<float>& memory
) {
    // 从 KVCacheData 中提取 K 缓存：形状为 L×D，L 是上下文长度，D 是注意力头维度。
    const auto& kv_cache = raw_data.export_data();
    const auto& keys = kv_cache.keys;  // L x D
    
    size_t context_length = keys.size();

    size_t head_dim = keys.empty() ? 0 : keys[0].size();
    
    if (context_length == 0 || head_dim == 0) {
        std::clog << "[PagedKVIndexBuilder] Warning: Empty KV cache data." << std::endl;
        memory.set_data({});
        return;
    }
    
    // 根据权重矩阵确定投影后的输出维度。
    size_t output_dim = weight_.size();  // weight_ 的形状为 output_dim×head_dim。
    if (output_dim == 0 || weight_[0].size() != head_dim) {
        std::clog << "[PagedKVIndexBuilder] Warning: Weight matrix dimension mismatch. "
                  << "Expected weight[?][" << head_dim << "], got weight[" 
                  << output_dim << "][" << (weight_.empty() ? 0 : weight_[0].size()) << "]" << std::endl;
        memory.set_data({});
        return;
    }
    
    // 计算页数：每页包含 page_size_ 个 token，不足一页时向上取整。
    size_t num_pages = (context_length + page_size_ - 1) / page_size_;
    
    std::vector<std::vector<float>> result;
    result.reserve(num_pages);
    
    for (size_t page_idx = 0; page_idx < num_pages; ++page_idx) {
        size_t start_token = page_idx * page_size_;
        size_t end_token = std::min(start_token + page_size_, context_length);
        size_t tokens_in_page = end_token - start_token;
        
        // 步骤 1：平均池化，将本页所有 token 的 Key 合并为一个向量。
        std::vector<float> pooled_token(head_dim, 0.0f);
        for (size_t token_idx = start_token; token_idx < end_token; ++token_idx) {
            for (size_t d = 0; d < head_dim; ++d) {
                pooled_token[d] += keys[token_idx][d];
            }
        }
        // 除以本页实际的 token 数量，得到平均值。
        for (size_t d = 0; d < head_dim; ++d) {
            pooled_token[d] /= static_cast<float>(tokens_in_page);
        }
        
        // 步骤 2：使用权重矩阵进行线性投影。
        // output = weight_ @ pooled_token，即矩阵与向量相乘。
        // weight_ 的形状为 output_dim×head_dim，pooled_token 的长度为 head_dim。
        // 投影结果的长度为 output_dim。
        std::vector<float> projected_token(output_dim, 0.0f);
        for (size_t o = 0; o < output_dim; ++o) {
            for (size_t d = 0; d < head_dim; ++d) {
                projected_token[o] += weight_[o][d] * pooled_token[d];
            }
        }
        
        result.push_back(projected_token);
    }
    
    // 将按页排列的投影向量写入 FlatIndexMemory。
    memory.set_data(result);
}

void PagedKVIndexBuilder::run_cpu_kernel(
    const data_type::KVCacheData<float>& raw_data,
    data_type::FlatIndexMemory<float>& memory
) {
    // TODO：实现 CPU 内核；当前复用参考实现。
    run_test_kernel(raw_data, memory);
    return;
}

void PagedKVIndexBuilder::run_gpu_kernel(
    const data_type::KVCacheData<float>& raw_data,
    data_type::FlatIndexMemory<float>& memory
) {
    // TODO：实现 GPU 内核。
    std::clog << "[PagedKVIndexBuilder] GPU kernel not implemented yet." << std::endl;
    return;
}

void PagedKVIndexBuilder::run_fpga_kernel(
    const data_type::KVCacheData<float>& raw_data,
    data_type::FlatIndexMemory<float>& memory
) {
    // TODO：实现 FPGA 内核。
    std::clog << "[PagedKVIndexBuilder] FPGA kernel not implemented yet." << std::endl;
    return;
}

}  // namespace step
}  // namespace heteromm
