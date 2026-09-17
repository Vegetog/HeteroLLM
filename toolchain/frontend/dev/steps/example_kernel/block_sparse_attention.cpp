/**
 * @file block_sparse_attention.cpp
 * @brief BlockSparseAttention 各类 kernel 的实现
 * 
 * 该文件实现分块稀疏注意力的参考计算过程：
 * 1. 接收 KV Cache、Query 和 Top-K 块编号
 * 2. 根据块编号找到被选中块内的 Key 和 Value
 * 3. 仅使用被选中的 Key 计算注意力分数 Q @ K^T
 * 4. 对这些分数计算 softmax
 * 5. 使用注意力权重对被选中的 Value 加权求和
 *
 * 这是容易理解和验证的标量参考实现，并非高性能 GPU/FPGA kernel。
 */

#include "../apply_memory.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace heteromm {
namespace step {

void BlockSparseAttention::run_test_kernel(
    const data_type::KVCacheData<float>& retrieved_data,
    const data_type::TopKIndex& index,
    const data_type::VectorInputOutputData<float>& input,
    data_type::VectorInputOutputData<float>& output
) {
    // 取出原始 KV Cache。每个 token 分别对应一个 Key 向量和 Value 向量。
    const auto& kv_cache = retrieved_data.export_data();
    const auto& keys = kv_cache.keys;      // 形状：[context_length, head_dim]
    const auto& values = kv_cache.values;  // 形状：[context_length, head_dim]
    
    // 取出上一阶段选中的 Top-K 块编号，这里的编号还不是 token 编号。
    const auto& topk_indices = index.export_data();
    
    // 取出用于计算最终 Attention 的 Query 向量。
    const auto& query = input.export_data();  // 形状：[head_dim]
    
    size_t head_dim = query.size();
    size_t context_length = keys.size();
    size_t num_blocks = topk_indices.size();
    
    if (context_length == 0 || head_dim == 0 || num_blocks == 0) {
        // 任一必要输入为空时，返回与 Query 等长的零向量。
        std::vector<float> zero_output(head_dim, 0.0f);
        output.set_data(zero_output);
        return;
    }

    // 标准 Scaled Dot-Product Attention 的缩放系数 1/sqrt(d_k)，防止维度较大时内积过大。
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    // 将 Top-K 块编号展开为具体 token 编号。
    // 若 page_size_=16，块 b 对应 token [b*16, b*16+15]。
    std::vector<int> selected_token_indices;
    for (int block_idx : topk_indices) {
        // 每个块对应 page_size_ 个连续 token；最后一块可能不满，需用 context_length 截断。
        int start_token = block_idx * static_cast<int>(page_size_);
        int end_token = std::min(start_token + static_cast<int>(page_size_), 
                                  static_cast<int>(context_length));
        
        for (int t = start_token; t < end_token; ++t) {
            selected_token_indices.push_back(t);
        }
    }
    
    size_t num_selected_tokens = selected_token_indices.size();
    
    if (num_selected_tokens == 0) {
        std::vector<float> zero_output(head_dim, 0.0f);
        output.set_data(zero_output);
        return;
    }
    
    // 步骤 1：仅计算 Query 与被选中 token 的 Key 之间的注意力分数。
    // scores[i] = Q @ K[selected_token_indices[i]]^T * scale
    std::vector<float> attention_scores(num_selected_tokens);
    
    for (size_t i = 0; i < num_selected_tokens; ++i) {
        int token_idx = selected_token_indices[i];
        const auto& key_vec = keys[token_idx];
        
        float dot_product = 0.0f;
        size_t dim = std::min(head_dim, key_vec.size());
        for (size_t d = 0; d < dim; ++d) {
            dot_product += query[d] * key_vec[d];
        }
        attention_scores[i] = dot_product * scale;
    }
    
    // 步骤 2：在所有被选中 token 之间计算 softmax。
    // softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    
    // 先减去最大值以提高数值稳定性，避免 exp 溢出。
    float max_score = *std::max_element(attention_scores.begin(), attention_scores.end());
    
    // 计算 exp(score-max_score) 及其总和，用作 softmax 分母。
    std::vector<float> exp_scores(num_selected_tokens);
    float sum_exp = 0.0f;
    
    for (size_t i = 0; i < num_selected_tokens; ++i) {
        exp_scores[i] = std::exp(attention_scores[i] - max_score);
        sum_exp += exp_scores[i];
    }
    
    // 归一化后得到注意力权重，所有权重之和为 1。
    std::vector<float> attention_weights(num_selected_tokens);
    for (size_t i = 0; i < num_selected_tokens; ++i) {
        attention_weights[i] = exp_scores[i] / sum_exp;
    }
    
    // 步骤 3：使用注意力权重对对应的 Value 向量加权求和。
    // output = sum(attention_weights[i] * V[selected_token_indices[i]])
    std::vector<float> result(head_dim, 0.0f);
    
    for (size_t i = 0; i < num_selected_tokens; ++i) {
        int token_idx = selected_token_indices[i];
        const auto& value_vec = values[token_idx];
        float weight = attention_weights[i];
        
        size_t dim = std::min(head_dim, value_vec.size());
        for (size_t d = 0; d < dim; ++d) {
            result[d] += weight * value_vec[d];
        }
    }
    
    output.set_data(result);
}

void BlockSparseAttention::run_cpu_kernel(
    const data_type::KVCacheData<float>& retrieved_data,
    const data_type::TopKIndex& index,
    const data_type::VectorInputOutputData<float>& input,
    data_type::VectorInputOutputData<float>& output
) {
    // TODO：实现优化的 CPU kernel。当前直接复用标量参考实现。
    run_test_kernel(retrieved_data, index, input, output);
    return;
}

void BlockSparseAttention::run_gpu_kernel(
    const data_type::KVCacheData<float>& retrieved_data,
    const data_type::TopKIndex& index,
    const data_type::VectorInputOutputData<float>& input,
    data_type::VectorInputOutputData<float>& output
) {
    // TODO：实现 GPU kernel。当前路径不会产生有效输出。
    std::clog << "[BlockSparseAttention] GPU kernel not implemented. exiting." << std::endl;
    return;
}

void BlockSparseAttention::run_fpga_kernel(
    const data_type::KVCacheData<float>& retrieved_data,
    const data_type::TopKIndex& index,
    const data_type::VectorInputOutputData<float>& input,
    data_type::VectorInputOutputData<float>& output
) {
    // TODO：实现 FPGA kernel。当前路径不会产生有效输出。
    std::clog << "[BlockSparseAttention] FPGA kernel not implemented. exiting." << std::endl;
    return;
}

}  // namespace step
}  // namespace heteromm
