/**
 * @file memory_retrieval.h
 * @brief MemoryRetrieval 步骤：根据分数选出前 k 个条目
 * 
 * 本步骤负责根据已计算的相似度分数，
 * 选出相关性最高的记忆条目。
 */

#ifndef HETEROMM_DEV_STEPS_MEMORY_RETRIEVAL_H_
#define HETEROMM_DEV_STEPS_MEMORY_RETRIEVAL_H_

#include "util.h"
#include "../types/score.h"
#include "../types/retrieved_index.h"
#include <type_traits>
#include <memory>
#include <iostream>

namespace heteromm {
namespace step {

/**
 * @brief MemoryRetrieval 步骤的抽象基类
 * 
 * 数据流： Score -> MemoryRetrieval -> RetrievedIndex
 * 
 * 本步骤接收已计算的分数，选出相关性最高的
 * k 个记忆条目的索引。
 * 
 * 用法示例：
 * @code
 *   class TopKRetrieval : public MemoryRetrieval<MyScore, MyIndices> {
 *       void run_cpu_kernel(const MyScore& s, MyIndices& idx) override {
 *           // 选出前 k 个条目
 *       }
 *   };
 * @endcode
 */
template<typename ScoreType, typename IndexType>
class MemoryRetrieval {

static_assert(std::is_base_of<data_type::Score<typename ScoreType::content_type>, ScoreType>::value,
              "ScoreType must derive from data_type::Score");
static_assert(std::is_base_of<data_type::RetrievedIndex<typename IndexType::content_type>, IndexType>::value,
              "IndexType must derive from data_type::RetrievedIndex");

public:
    MemoryRetrieval() = default;
    virtual ~MemoryRetrieval() = default;

    int execute(
        const ScoreType& score,
        IndexType& index, // 进行功能测试时，在此传入预期的正确结果
        bool run_functional_test = false,
        bool verbose = false
    ) {
        if(run_functional_test) {
            if(verbose) {
                std::clog << "[MemoryRetrieval] Running functional test kernel." << std::endl;
                std::clog << "  Score Type: " << score.type_name() << std::endl;
                std::clog << "  Index Type: " << index.type_name() << std::endl;
            }
            // 深拷贝预期索引，供后续结果校验使用
            IndexType original_index = index;

            run_test_kernel(score, index);

            if(verbose) {
                std::clog << "[MemoryRetrieval] Functional test kernel completed." << std::endl;
            }
            if(index.is_equal(original_index)) {
                if(verbose) {
                    std::clog << "[MemoryRetrieval] Functional test passed: computed index matches original." << std::endl;
                }
            } else {
                if(verbose) {
                    std::clog << "[MemoryRetrieval] Functional test failed: computed index does not match original." << std::endl;
                }
                return 1; 
            }
            
            return 0;
        }

        // 先按照静态调度配置选择执行分支
        switch (current_kernel_) {
            case KernelType::CPU:
                if(verbose) {
                    std::clog << "[MemoryRetrieval] Running CPU kernel." << std::endl;
                }
                run_cpu_kernel(score, index);
                if(verbose) {
                    std::clog << "[MemoryRetrieval] CPU kernel completed." << std::endl;
                }
                break;
            case KernelType::GPU:
                if(verbose) {
                    std::clog << "[MemoryRetrieval] Running GPU kernel." << std::endl;
                }
                run_gpu_kernel(score, index);
                if(verbose) {
                    std::clog << "[MemoryRetrieval] GPU kernel completed." << std::endl;
                }
                break;
            case KernelType::FPGA:
                if(verbose) {
                    std::clog << "[MemoryRetrieval] Running FPGA kernel." << std::endl;
                }
                run_fpga_kernel(score, index);
                if(verbose) {
                    std::clog << "[MemoryRetrieval] FPGA kernel completed." << std::endl;
                }
                break;
            default:
                break;
        }
        return 0;
    }

    KernelType current_kernel() const {
        return current_kernel_;
    }

    void set_current_kernel(KernelType type) {
        current_kernel_ = type;
    }

protected:
    virtual void run_test_kernel(
        const ScoreType& score,
        IndexType& index
    ) = 0;

    virtual void run_gpu_kernel(
        const ScoreType& score,
        IndexType& index
    ) = 0;

    virtual void run_fpga_kernel(
        const ScoreType& score,
        IndexType& index
    ) = 0;

    virtual void run_cpu_kernel(
        const ScoreType& score,
        IndexType& index
    ) = 0;

private:
    KernelType current_kernel_ = KernelType::CPU;

};

/**
 * @brief Top-K 检索：选出分数最高的 k 个条目
 * 
 * 返回分数最高的 k 个条目的索引。
 */
class TopKRetrieval : public MemoryRetrieval<
    data_type::VectorScore<float>,
    data_type::TopKIndex
> {
public:
    explicit TopKRetrieval(size_t k) : k_(k) {}
    ~TopKRetrieval() override = default;

    size_t k() const { return k_; }
    void set_k(size_t k) { k_ = k; }

protected:
    void run_test_kernel(
        const data_type::VectorScore<float>& score,
        data_type::TopKIndex& index
    ) override;

    void run_cpu_kernel(
        const data_type::VectorScore<float>& score,
        data_type::TopKIndex& index
    ) override;

    void run_gpu_kernel(
        const data_type::VectorScore<float>& score,
        data_type::TopKIndex& index
    ) override;

    void run_fpga_kernel(
        const data_type::VectorScore<float>& score,
        data_type::TopKIndex& index
    ) override;

private:
    size_t k_;
};

/**
 * @brief 阈值检索：选出分数超过阈值的条目
 * 
 * 返回位图，标记哪些条目的分数超过阈值。
 */
class ThresholdRetrieval : public MemoryRetrieval<
    data_type::VectorScore<float>,
    data_type::ThresholdBitmapIndex
> {
public:
    explicit ThresholdRetrieval(float threshold) : threshold_(threshold) {}
    ~ThresholdRetrieval() override = default;

    float threshold() const { return threshold_; }
    void set_threshold(float threshold) { threshold_ = threshold; }

protected:
    void run_test_kernel(
        const data_type::VectorScore<float>& score,
        data_type::ThresholdBitmapIndex& index
    ) override;

    void run_cpu_kernel(
        const data_type::VectorScore<float>& score,
        data_type::ThresholdBitmapIndex& index
    ) override;

    void run_gpu_kernel(
        const data_type::VectorScore<float>& score,
        data_type::ThresholdBitmapIndex& index
    ) override;

    void run_fpga_kernel(
        const data_type::VectorScore<float>& score,
        data_type::ThresholdBitmapIndex& index
    ) override;

private:
    float threshold_;
};

}  // namespace step
}  // namespace heteromm

#endif  // HETEROMM_DEV_STEPS_MEMORY_RETRIEVAL_H_
