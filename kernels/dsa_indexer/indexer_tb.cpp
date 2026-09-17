#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <set>

#include <gflags/gflags.h>
#include <tapa.h>

#include "indexer_vanilla.h"

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");

// 软件参考实现
void indexer_top_ref(
    const int L,
    const std::vector<std::vector<ap_int<16>>>& query_vecs,  // [NUM_INDEX_HEAD][HEAD_DIM]
    const std::vector<std::vector<ap_int<16>>>& key_vecs,    // [L][HEAD_DIM]
    const std::vector<float>& weights,                        // [NUM_INDEX_HEAD]
    std::vector<int>& topk_ids)                              // [TOP_K]
{
    // 计算加权索引分数
    std::vector<std::pair<float, int>> scores(L);
    
    for (int k = 0; k < L; k++) {
        float total_score = 0.0f;
        
        for (int h = 0; h < NUM_INDEX_HEAD; h++) {
            // 计算 qk 点积（使用 int64_t 累加以避免溢出）
            int64_t qk = 0;
            for (int d = 0; d < HEAD_DIM; d++) {
                qk += query_vecs[h][d].to_int() * key_vecs[k][d].to_int();
            }
            
            // 应用 ReLU 激活
            if (qk < 0) {
                qk = 0;
            }
            
            // 加权并累加
            total_score += weights[h] * (float)qk;
        }

        scores[k] = std::make_pair(total_score, k);
    }
    
    // 找出分数最高的 K 个索引
    std::sort(scores.begin(), scores.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    topk_ids.resize(TOP_K);
    for (int i = 0; i < TOP_K; i++) {
        topk_ids[i] = scores[i].second;
    }
}

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    int L = 4096;  // Key 向量的数量
    if (argc > 1) {
        L = std::atoi(argv[1]);
    }
    
    std::cout << "Indexer Testbench" << std::endl;
    std::cout << "L (key vectors): " << L << std::endl;
    std::cout << "NUM_INDEX_HEAD: " << NUM_INDEX_HEAD << std::endl;
    std::cout << "HEAD_DIM: " << HEAD_DIM << std::endl;
    std::cout << "TOP_K: " << TOP_K << std::endl;
    
    // 初始化随机数生成器
    std::mt19937 gen(42);  // 使用固定种子，保证结果可复现
    std::uniform_int_distribution<int> dis_int(-128, 127);  // 用于生成 ap_int<16> 测试数据的取值范围
    std::uniform_real_distribution<float> dis_float(0.0f, 2.0f);
    
    // 生成整数类型的随机 Query 向量（16 个头，每个头 128 维）
    std::vector<std::vector<ap_int<16>>> query_vecs(NUM_INDEX_HEAD, std::vector<ap_int<16>>(HEAD_DIM));
    for (int h = 0; h < NUM_INDEX_HEAD; h++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            query_vecs[h][d] = ap_int<16>(dis_int(gen));
        }
    }
    
    // 生成整数类型的随机 Key 向量（默认 4096 个向量，每个向量 128 维）
    std::vector<std::vector<ap_int<16>>> key_vecs(L, std::vector<ap_int<16>>(HEAD_DIM));
    for (int k = 0; k < L; k++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            key_vecs[k][d] = ap_int<16>(dis_int(gen));
        }
    }
    
    // 生成随机权重（16 个头对应 16 个权重）
    std::vector<float> weights(NUM_INDEX_HEAD);
    for (int h = 0; h < NUM_INDEX_HEAD; h++) {
        weights[h] = dis_float(gen);  // 使用正权重
    }
    
    // 准备硬件格式的输入数据
    // qk_vec_mem 的内存布局： 
    // - 首先：沿 NUM_INDEX_HEAD 维度打包 Query 向量
    //   对于 HEAD_DIM 中的每个维度，将 16 个头的对应元素打包到一个 vec_t 中
    // - 然后：沿 L 维度打包 Key 向量
    //   对于 HEAD_DIM 中的每个维度，将连续 16 个 Key 的对应元素打包到一个 vec_t 中
    const int query_lines = HEAD_DIM;  // 共 HEAD_DIM 行，每行包含 16 个头的对应元素
    const int key_lines = L * HEAD_DIM / 16;  // L 个向量 × HEAD_DIM 维 ÷ 每个 vec_t 的 16 个元素
    const int total_qk_lines = query_lines + key_lines;
    
    // 为 8 个通道分配内存
    std::vector<std::vector<tapa::vec_t<ap_int<16>, 16>>> qk_vec_hw(8);
    for (int ch = 0; ch < 8; ch++) {
        qk_vec_hw[ch].resize(total_qk_lines / 8);
    }
    
    // 打包 Query 向量：转置为 [HEAD_DIM][NUM_INDEX_HEAD]
    for (int d = 0; d < HEAD_DIM; d++) {
        tapa::vec_t<ap_int<16>, 16> packed;
        for (int h = 0; h < NUM_INDEX_HEAD; h++) {
            packed[h] = query_vecs[h][d];
        }
        int ch = d % 8;
        int idx = d / 8;
        qk_vec_hw[ch][idx] = packed;
    }
    
    // 打包 Key 向量：对每个维度，将连续 16 个 Key 的对应元素打包
    for (int k = 0; k < L; k += 16) {
        for (int d = 0; d < HEAD_DIM; d++) {
            tapa::vec_t<ap_int<16>, 16> packed;
            for (int i = 0; i < 16; i++) {
                packed[i] = key_vecs[k + i][d];
            }
            int global_line = query_lines + (k / 16) * HEAD_DIM + d;
            int ch = global_line % 8;
            int idx = global_line / 8;
            qk_vec_hw[ch][idx] = packed;
        }
    }
    
    // 将权重打包为硬件所需的格式
    std::vector<tapa::vec_t<float, 16>> weight_hw(1);
    for (int i = 0; i < NUM_INDEX_HEAD; i++) {
        weight_hw[0][i] = weights[i];
    }
    
    // 分配输出内存
    std::vector<tapa::vec_t<int, 16>> topk_id_hw(TOP_K / 16);
    
    std::cout << "\nRunning hardware kernel..." << std::endl;
    
    // 调用内核
    int64_t kernel_time_ns = 0;
    kernel_time_ns = tapa::invoke(indexer_top, FLAGS_bitstream,
                 L,
                 tapa::read_only_mmaps<tapa::vec_t<ap_int<16>, 16>, 8>(qk_vec_hw),
                 tapa::read_only_mmap<tapa::vec_t<float, 16>>(weight_hw),
                 tapa::write_only_mmap<tapa::vec_t<int, 16>>(topk_id_hw));
    
    std::cout << "Hardware kernel completed." << std::endl;
    std::clog << "kernel time: " << kernel_time_ns * 1e-9 << " s" << std::endl;

    
    // 提取内核输出结果
    std::vector<int> topk_ids_hw(TOP_K);
    for (int i = 0; i < TOP_K / 16; i++) {
        for (int j = 0; j < 16; j++) {
            topk_ids_hw[i * 16 + j] = topk_id_hw[i][j];
        }
    }
    
    // 计算软件参考结果（使用与内核相同的整数输入进行比较）
    std::cout << "\nRunning software reference..." << std::endl;
    
    std::vector<int> topk_ids_sw;
    indexer_top_ref(L, query_vecs, key_vecs, weights, topk_ids_sw);
    
    std::cout << "Software reference completed." << std::endl;
    
    // 比较结果
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Hardware Top-64 IDs:" << std::endl;
    for (int i = 0; i < std::min(64, TOP_K); i++) {
        std::cout << topk_ids_hw[i] << " ";
    }
    std::cout << "..." << std::endl;
    
    std::cout << "\nSoftware Top-64 IDs:" << std::endl;
    for (int i = 0; i < std::min(64, TOP_K); i++) {
        std::cout << topk_ids_sw[i] << " ";
    }
    std::cout << "..." << std::endl;
    
    // 检查正确性
    // 由于分数相同时 Top-K 的顺序可能略有不同，这里检查两个索引集合的重合程度
    std::set<int> hw_set(topk_ids_hw.begin(), topk_ids_hw.end());
    std::set<int> sw_set(topk_ids_sw.begin(), topk_ids_sw.end());
    
    int matches = 0;
    for (int id : hw_set) {
        if (sw_set.count(id)) {
            matches++;
        }
    }
    
    float overlap_ratio = static_cast<float>(matches) / TOP_K;
    std::cout << "\nOverlap: " << matches << "/" << TOP_K << " (" << (overlap_ratio * 100) << "%)" << std::endl;
    
    // 检查索引及其顺序是否完全一致
    bool exact_match = true;
    for (int i = 0; i < TOP_K; i++) {
        if (topk_ids_hw[i] != topk_ids_sw[i]) {
            exact_match = false;
            break;
        }
    }
    
    if (exact_match) {
        std::cout << "\n✓ PASSED: Exact match between hardware and software!" << std::endl;
        return 0;
    } else if (overlap_ratio >= 0.95) {
        std::cout << "\n✓ PASSED: High overlap (>95%) between hardware and software!" << std::endl;
        std::cout << "  (Small differences may be due to floating point precision and tie-breaking)" << std::endl;
        return 0;
    } else {
        std::cout << "\n✗ FAILED: Significant mismatch between hardware and software!" << std::endl;
        return 1;
    }
}
