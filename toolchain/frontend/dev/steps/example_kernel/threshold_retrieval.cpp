/**
 * @file threshold_retrieval.cpp
 * @brief ThresholdRetrieval 各类计算内核的实现
 */

#include "../memory_retrieval.h"

namespace heteromm {
namespace step {

void ThresholdRetrieval::run_test_kernel(
    const data_type::VectorScore<float>& score,
    data_type::ThresholdBitmapIndex& index
) {
    const auto& scores = score.export_data();
    size_t n = scores.size();
    
    // 计算位图所需的 64 位存储单元数量
    size_t num_words = (n + 63) / 64;
    std::vector<unsigned long long> bitmap(num_words, 0);

    // 将分数大于等于阈值的条目所对应的位设为 1
    for (size_t i = 0; i < n; ++i) {
        if (scores[i] >= threshold_) {
            size_t word_index = i / 64;
            size_t bit_index = i % 64;
            bitmap[word_index] |= (1ULL << bit_index);
        }
    }

    index = data_type::ThresholdBitmapIndex(bitmap, n);
}

void ThresholdRetrieval::run_cpu_kernel(
    const data_type::VectorScore<float>& score,
    data_type::ThresholdBitmapIndex& index
) {
    // TODO：实现优化的 CPU 内核
    run_test_kernel(score, index);
    return;
}

void ThresholdRetrieval::run_gpu_kernel(
    const data_type::VectorScore<float>& score,
    data_type::ThresholdBitmapIndex& index
) {
    // TODO：实现 GPU 内核
    std::clog << "[ThresholdRetrieval] GPU kernel not implemented. exiting." << std::endl;
    return;
}

void ThresholdRetrieval::run_fpga_kernel(
    const data_type::VectorScore<float>& score,
    data_type::ThresholdBitmapIndex& index
) {
    // TODO：实现 FPGA 内核
    std::clog << "[ThresholdRetrieval] FPGA kernel not implemented. exiting." << std::endl;
    return;
}

}  // namespace step
}  // namespace heteromm
