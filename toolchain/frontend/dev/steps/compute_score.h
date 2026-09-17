/**
 * @file compute_score.h
 * @brief ComputeScore 步骤：计算查询与记忆之间的相似度分数
 * 
 * 本步骤负责计算查询向量与记忆条目之间的
 * 相似度或相关性分数。
 */

#ifndef HETEROMM_DEV_STEPS_COMPUTE_SCORE_H_
#define HETEROMM_DEV_STEPS_COMPUTE_SCORE_H_

#include "util.h"
#include "../types/memory.h"
#include "../types/query.h"
#include "../types/score.h"
#include <type_traits>
#include <memory>
#include <tuple>
#include <assert.h>
#include <iostream>

namespace heteromm {
namespace step {

/**
 * @brief ComputeScore 步骤的抽象基类
 * 
 * 数据流： (Memory, Query) -> ComputeScore -> Score
 * 
 * 本步骤计算查询向量与记忆条目之间的相似度分数，
 * 利用这些分数识别相关性最高的记忆条目。
 * 
 * 用法示例：
 * @code
 *   class DotProductScore : public ComputeScoreStep<MyIndex, MyQuery, MyScore> {
 *       StepStatus run_cpu_kernel(const MyIndex& mem, const MyQuery& q, MyScore& s) override {
 *           // 计算内积
 *       }
 *   };
 * @endcode
 */
template<typename MemoryType, typename QueryType, typename ScoreType>
class ComputeScore {

static_assert(std::is_base_of<data_type::Memory<typename MemoryType::content_type>, MemoryType>::value,
              "MemoryType must derive from data_type::Memory");
static_assert(std::is_base_of<data_type::Query<typename QueryType::content_type>, QueryType>::value,
              "QueryType must derive from data_type::Query");
static_assert(std::is_base_of<data_type::Score<typename ScoreType::content_type>, ScoreType>::value,
              "ScoreType must derive from data_type::Score");

public:
    ComputeScore() = default;
    virtual ~ComputeScore() = default;

    int execute(
        const MemoryType& memory,
        const QueryType& query,
        ScoreType& score, // 进行功能测试时，在此传入预期的正确结果
        bool run_functional_test = false,
        bool verbose = false
    ) {
        if(run_functional_test) {
            if(verbose) {
                std::clog << "[ComputeScore] Running functional test kernel." << std::endl;
                std::clog << "  Memory Type: " << memory.type_name() << std::endl;
                std::clog << "  Query Type: " << query.type_name() << std::endl;
                std::clog << "  Score Type: " << score.type_name() << std::endl;
            }
            // 深拷贝预期分数，供后续结果校验使用
            ScoreType original_score = score;

            run_test_kernel(memory, query, score);

            if(verbose) {
                std::clog << "[ComputeScore] Functional test kernel completed." << std::endl;
            }
            if(score.is_equal(original_score)) {
                if(verbose) {
                    std::clog << "[ComputeScore] Functional test passed: computed score matches original." << std::endl;
                }
            } else {
                if(verbose) {
                    std::clog << "[ComputeScore] Functional test failed: computed score does not match original." << std::endl;
                }
                return 1; 
            }
            
            return 0;
        }

        // 先按照静态调度配置选择执行分支
        switch (current_kernel_) {
            case KernelType::CPU:
                if(verbose) {
                    std::clog << "[ComputeScore] Running CPU kernel." << std::endl;
                }
                run_cpu_kernel(memory, query, score);
                if(verbose) {
                    std::clog << "[ComputeScore] CPU kernel completed." << std::endl;
                }
                break;
            case KernelType::GPU:
                if(verbose) {
                    std::clog << "[ComputeScore] Running GPU kernel." << std::endl;
                }
                run_gpu_kernel(memory, query, score);
                if(verbose) {
                    std::clog << "[ComputeScore] GPU kernel completed." << std::endl;
                }
                break;
            case KernelType::FPGA:
                if(verbose) {
                    std::clog << "[ComputeScore] Running FPGA kernel." << std::endl;
                }
                run_fpga_kernel(memory, query, score);
                if(verbose) {
                    std::clog << "[ComputeScore] FPGA kernel completed." << std::endl;
                }
                break;
            default:
                break;
            }
        // 动态调度：调用后端
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
        const MemoryType& memory,
        const QueryType& query,
        ScoreType& score
    ) = 0;

    virtual void run_gpu_kernel(
        const MemoryType& memory,
        const QueryType& query,
        ScoreType& score
    ) = 0;

    virtual void run_fpga_kernel(
        const MemoryType& memory,
        const QueryType& query,
        ScoreType& score
    ) = 0;

    virtual void run_cpu_kernel(
        const MemoryType& memory,
        const QueryType& query,
        ScoreType& score
    ) = 0;

private:
    KernelType current_kernel_ = KernelType::CPU;

};

/**
 * @brief 计算查询向量与记忆向量之间的内积（点积）
 * 
 * 计算 VectorQuery 与 FlatIndexMemory 中各向量的内积相似度分数。
 */
class InnerProductCompute : public ComputeScore<
    data_type::FlatIndexMemory<float>,
    data_type::VectorQuery<float>,
    data_type::VectorScore<float>
> {
public:
    InnerProductCompute() = default;
    ~InnerProductCompute() override = default;

protected:
    void run_test_kernel(
        const data_type::FlatIndexMemory<float>& memory,
        const data_type::VectorQuery<float>& query,
        data_type::VectorScore<float>& score
    ) override;

    void run_cpu_kernel(
        const data_type::FlatIndexMemory<float>& memory,
        const data_type::VectorQuery<float>& query,
        data_type::VectorScore<float>& score
    ) override;

    void run_gpu_kernel(
        const data_type::FlatIndexMemory<float>& memory,
        const data_type::VectorQuery<float>& query,
        data_type::VectorScore<float>& score
    ) override;

    void run_fpga_kernel(
        const data_type::FlatIndexMemory<float>& memory,
        const data_type::VectorQuery<float>& query,
        data_type::VectorScore<float>& score
    ) override;
};

}  // namespace step
}  // namespace heteromm

#endif  // HETEROMM_DEV_STEPS_COMPUTE_SCORE_H_
