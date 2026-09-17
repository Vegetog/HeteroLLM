/**
 * @file apply_memory.h
 * @brief ApplyMemory 步骤：应用检索到的记忆，生成输出
 * 
 * 本步骤将检索到的记忆条目与
 * 输入数据结合，生成最终输出。
 */

#ifndef HETEROMM_DEV_STEPS_APPLY_MEMORY_H_
#define HETEROMM_DEV_STEPS_APPLY_MEMORY_H_

#include "util.h"
#include "../types/retrieved_data.h"
#include "../types/retrieved_index.h"
#include "../types/target_data.h"
#include <type_traits>
#include <memory>
#include <vector>
#include <string>
#include <tuple>
#include <assert.h>
#include <iostream>

namespace heteromm {
namespace step {

/**
 * @brief ApplyMemory 步骤的抽象基类
 * 
 * 数据流： (RawData, RetrievedIndex, ArbitraryInput) -> ApplyMemory -> OutputData
 * 
 * 本步骤结合原始数据、检索得到的记忆索引
 * 以及可选的额外输入，生成最终输出。
 * 
 * 用法示例：
 * @code
 *   class MemoryAugmentation : public ApplyMemoryStep<MyRaw, MyIdx, MyInput, MyOutput> {
 *       StepStatus run_cpu_kernel(const MyRaw& raw, const MyIdx& idx, 
 *                                  const MyInput& in, MyOutput& out) override {
 *           // 应用记忆增强
 *       }
 *   };
 * @endcode
 */

template<typename RetDataType, typename IndexType, typename InputType, typename OutputType>
class ApplyMemory {
static_assert(std::is_base_of<data_type::RetrievedData<typename RetDataType::content_type>, RetDataType>::value,
              "RetDataType must derive from data_type::RetrievedData");
static_assert(std::is_base_of<data_type::RetrievedIndex<typename IndexType::content_type>, IndexType>::value,
              "IndexType must derive from data_type::RetrievedIndex");
static_assert(std::is_base_of<data_type::TargetData<typename InputType::content_type>, InputType>::value,
              "InputType must derive from data_type::TargetData");
static_assert(std::is_base_of<data_type::TargetData<typename OutputType::content_type>, OutputType>::value,
              "OutputType must derive from data_type::TargetData");
public:
    ApplyMemory() = default;
    virtual ~ApplyMemory() = default;

    int execute(
        const RetDataType& retrieved_data,
        const IndexType& index,
        const InputType& input,
        OutputType& output, // 进行功能测试时，在此传入预期的正确结果
        bool run_functional_test = false,
        bool verbose = false
    ) {
        if(run_functional_test) {
            if(verbose) {
                std::clog << "[ApplyMemory] Running functional test kernel." << std::endl;
                std::clog << "  RetrievedData Type: " << retrieved_data.type_name() << std::endl;
                std::clog << "  Index Type: " << index.type_name() << std::endl;
                std::clog << "  Input Type: " << input.type_name() << std::endl;
                std::clog << "  Output Type: " << output.type_name() << std::endl;
            }

            OutputType original_output = output;

            run_test_kernel(retrieved_data, index, input, output);
            
            if(!output.is_equal(original_output)) {
                std::clog << "[ApplyMemory] Functional test failed: output does not match expected." << std::endl;
                return 1;
            } else {
                if(verbose) {
                    std::clog << "[ApplyMemory] Functional test passed." << std::endl;
                }
            }
            return 0;
        } 

        // 先按照静态调度配置选择执行分支
        switch (current_kernel_) { 
            case KernelType::CPU:
                if(verbose) {
                    std::clog << "[ApplyMemory] Running CPU kernel." << std::endl;
                }
                run_cpu_kernel(retrieved_data, index, input, output);
                if(verbose) {
                    std::clog << "[ApplyMemory] CPU kernel completed." << std::endl;
                }
                break;
            case KernelType::GPU:
                if(verbose) {
                    std::clog << "[ApplyMemory] Running GPU kernel." << std::endl;
                }
                run_gpu_kernel(retrieved_data, index, input, output);
                if(verbose) {
                    std::clog << "[ApplyMemory] GPU kernel completed." << std::endl;
                }
                break;
            case KernelType::FPGA:
                if(verbose) {
                    std::clog << "[ApplyMemory] Running FPGA kernel." << std::endl;
                }
                run_fpga_kernel(retrieved_data, index, input, output);
                if(verbose) {
                    std::clog << "[ApplyMemory] FPGA kernel completed." << std::endl;
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
        const RetDataType& retrieved_data,
        const IndexType& index,
        const InputType& input,
        OutputType& output
    ) = 0;

    virtual void run_cpu_kernel(
        const RetDataType& retrieved_data,
        const IndexType& index,
        const InputType& input,
        OutputType& output
    ) = 0;

    virtual void run_gpu_kernel(
        const RetDataType& retrieved_data,
        const IndexType& index,
        const InputType& input,
        OutputType& output
    ) = 0;

    virtual void run_fpga_kernel(
        const RetDataType& retrieved_data,
        const IndexType& index,
        const InputType& input,
        OutputType& output
    ) = 0;

private:
    KernelType current_kernel_ = KernelType::CPU;

};

class BlockSparseAttention : public ApplyMemory<
    data_type::KVCacheData<float>,
    data_type::TopKIndex,
    data_type::VectorInputOutputData<float>,
    data_type::VectorInputOutputData<float>
> {

public:
    BlockSparseAttention() = default;
    BlockSparseAttention(size_t page_size) : page_size_(page_size) {}
    ~BlockSparseAttention() override = default;

protected:
    void run_test_kernel(
        const data_type::KVCacheData<float>& retrieved_data,
        const data_type::TopKIndex& index,
        const data_type::VectorInputOutputData<float>& input,
        data_type::VectorInputOutputData<float>& output
    ) override;

    void run_cpu_kernel(
        const data_type::KVCacheData<float>& retrieved_data,
        const data_type::TopKIndex& index,
        const data_type::VectorInputOutputData<float>& input,
        data_type::VectorInputOutputData<float>& output
    ) override;

    void run_gpu_kernel(
        const data_type::KVCacheData<float>& retrieved_data,
        const data_type::TopKIndex& index,
        const data_type::VectorInputOutputData<float>& input,
        data_type::VectorInputOutputData<float>& output
    ) override;

    void run_fpga_kernel(
        const data_type::KVCacheData<float>& retrieved_data,
        const data_type::TopKIndex& index,
        const data_type::VectorInputOutputData<float>& input,
        data_type::VectorInputOutputData<float>& output
    ) override;

private:
    size_t page_size_ = 0;

};

/**
 * @brief RAG 记忆应用：将检索到的文档与查询拼接
 * 
 * 本步骤实现 RAG 流水线的“应用记忆”阶段。
 * 根据检索得到的 Top-K 文档索引和文本数据库，
 * 通过以下操作构建增强后的提示词：
 * 1. 根据索引查找文档正文
 * 2. 根据文档构建上下文字符串
 * 3. 将上下文与原始查询拼接
 * 4. 对完整的 RAG 提示词进行分词
 * 
 * 在 rag_pipeline.py 流程中，这对应以下操作：
 * - BM25Retriever.get_documents() 查找文档正文
 * - RAGGenerator._build_context_string() 格式化上下文
 * - RAGGenerator._build_rag_prompt() 构建完整提示词
 * 
 * 数据流： (TextDBData, TopKIndex, TextInputOutputData<int>) -> RAGApplyMemory -> TextInputOutputData<int>
 */
class RAGApplyMemory : public ApplyMemory<
    data_type::TextDBData,
    data_type::TopKIndex,
    data_type::TextInputOutputData<int>,
    data_type::TextInputOutputData<int>
> {
public:
    RAGApplyMemory(
        const std::string& python_module_path = ".",
        const std::string& model_name = "meta-llama/Llama-3.2-1B-Instruct"
    ) : python_module_path_(python_module_path),
        model_name_(model_name) {}

    ~RAGApplyMemory() override = default;

protected:
    void run_gpu_kernel(
        const data_type::TextDBData& retrieved_data,
        const data_type::TopKIndex& index,
        const data_type::TextInputOutputData<int>& input,
        data_type::TextInputOutputData<int>& output
    ) override;

    void run_cpu_kernel(
        const data_type::TextDBData& retrieved_data,
        const data_type::TopKIndex& index,
        const data_type::TextInputOutputData<int>& input,
        data_type::TextInputOutputData<int>& output
    ) override;

    void run_fpga_kernel(
        const data_type::TextDBData& retrieved_data,
        const data_type::TopKIndex& index,
        const data_type::TextInputOutputData<int>& input,
        data_type::TextInputOutputData<int>& output
    ) override;

    void run_test_kernel(
        const data_type::TextDBData& retrieved_data,
        const data_type::TopKIndex& index,
        const data_type::TextInputOutputData<int>& input,
        data_type::TextInputOutputData<int>& output
    ) override;

private:
    std::string python_module_path_;
    std::string model_name_;
};

}  // namespace steps
}  // namespace heteromm

#endif  // HETEROMM_DEV_STEPS_APPLY_MEMORY_H_
