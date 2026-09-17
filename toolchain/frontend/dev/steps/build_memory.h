/**
 * @file build_memory.h
 * @brief BuildMemory 步骤：将原始数据 RawData 转换为记忆结构 Memory
 * 
 * 本步骤负责根据原始输入数据构建记忆结构。
 * 例如构建向量索引、创建哈希表，或
 * 对嵌入向量进行量化。
 */

#ifndef HETEROMM_DEV_STEPS_BUILD_MEMORY_H_
#define HETEROMM_DEV_STEPS_BUILD_MEMORY_H_

#include "util.h"
#include "../types/retrieved_data.h"
#include "../types/memory.h"
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
 * @brief BuildMemory 步骤的抽象基类
 * 
 * 数据流： RawData -> BuildMemory -> Memory
 * 
 * 本步骤处理原始输入数据，构建带索引的
 * 记忆结构，以便高效检索。
 * 
 * 用法示例：
 * @code
 *   class MyIndexBuilder : public BuildMemoryStep<MyRawData, MyIndex> {
 *       StepStatus run_cpu_kernel(const MyRawData& raw, MyIndex& memory) override {
 *           // 在 CPU/GPU 上构建索引
 *       }
 *   };
 * @endcode
 */
template<typename RetDataType, typename MemoryType>
class BuildMemory {
static_assert(std::is_base_of<data_type::RetrievedData<typename RetDataType::content_type>, RetDataType>::value,
              "RetDataType must derive from data_type::RetrievedData");
static_assert(std::is_base_of<data_type::Memory<typename MemoryType::content_type>, MemoryType>::value,
              "MemoryType must derive from data_type::Memory");

public:
    BuildMemory() = default;
    virtual ~BuildMemory() = default;

    int execute(
        const RetDataType& raw_data,
        MemoryType& memory, // 进行功能测试时，在此传入预期的正确结果
        bool run_functional_test = false,
        bool verbose = false
    ) {
        if(run_functional_test) {
            if(verbose) {
                std::clog << "[BuildMemory] Running functional test kernel." << std::endl;
                std::clog << "  RawData Type: " << raw_data.type_name() << std::endl;
                std::clog << "  Memory Type: " << memory.type_name() << std::endl;
            }

            MemoryType original_memory = memory;

            run_test_kernel(raw_data, memory);
            
            if(!memory.is_equal(original_memory)) {
                std::clog << "[BuildMemory] Functional test failed: output memory does not match expected." << std::endl;
                return 1;
            } else {
                if(verbose) {
                    std::clog << "[BuildMemory] Functional test passed." << std::endl;
                }
            }
            return 0;
        }

        // 先按照静态调度配置选择执行分支
        switch (current_kernel_) {
            case KernelType::CPU:
                if(verbose) {
                    std::clog << "[BuildMemory] Running CPU kernel." << std::endl;
                }
                run_cpu_kernel(raw_data, memory);
                if(verbose) {
                    std::clog << "[BuildMemory] CPU kernel completed." << std::endl;
                }
                break;
            case KernelType::GPU:
                if(verbose) {
                    std::clog << "[BuildMemory] Running GPU kernel." << std::endl;
                }
                run_gpu_kernel(raw_data, memory);
                if(verbose) {
                    std::clog << "[BuildMemory] GPU kernel completed." << std::endl;
                }
                break;
            case KernelType::FPGA:
                if(verbose) {
                    std::clog << "[BuildMemory] Running FPGA kernel." << std::endl;
                }
                run_fpga_kernel(raw_data, memory);
                if(verbose) {
                    std::clog << "[BuildMemory] FPGA kernel completed." << std::endl;
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
        const RetDataType& raw_data,
        MemoryType& memory
    ) = 0;

    virtual void run_cpu_kernel(
        const RetDataType& raw_data,
        MemoryType& memory
    ) = 0;

    virtual void run_gpu_kernel(
        const RetDataType& raw_data,
        MemoryType& memory
    ) = 0;

    virtual void run_fpga_kernel(
        const RetDataType& raw_data,
        MemoryType& memory
    ) = 0;

private:
    KernelType current_kernel_ = KernelType::CPU;      
};

class PagedKVIndexBuilder : public BuildMemory<data_type::KVCacheData<float>, data_type::FlatIndexMemory<float>> {
public:
    PagedKVIndexBuilder() = default;
    PagedKVIndexBuilder(const std::vector<std::vector<float>>& weight, int page_size)
        : weight_(weight), page_size_(page_size) {}
    ~PagedKVIndexBuilder() override = default;
protected:
    void run_test_kernel(
        const data_type::KVCacheData<float>& raw_data,
        data_type::FlatIndexMemory<float>& memory
    ) override;

    void run_cpu_kernel(
        const data_type::KVCacheData<float>& raw_data,
        data_type::FlatIndexMemory<float>& memory
    ) override;

    void run_gpu_kernel(
        const data_type::KVCacheData<float>& raw_data,
        data_type::FlatIndexMemory<float>& memory
    ) override;

    void run_fpga_kernel(
        const data_type::KVCacheData<float>& raw_data,
        data_type::FlatIndexMemory<float>& memory
    ) override;

private:
    std::vector<std::vector<float>> weight_;
    int page_size_;
};

/**
 * @brief BM25 数据集构建器：根据文本语料构建 BM25 索引记忆
 * 
 * 本步骤加载语料、进行分词、构建 BM25 索引，
 * 并为融合检索步骤准备 FPGA 缓冲区。
 * 
 * CPU 内核通过以下 Python 函数完成相关操作：
 * - bm25_loader_xrt.load_document_frequency_mmap: 加载文档频率
 * - bm25_loader_xrt.load_term_frequencies_mmap: 加载词频
 * - bm25_loader_xrt.pack_documents_for_hw: 将文档打包为 FPGA 所需的格式
 * - launch_bm25.fpga_retriever_setup: 初始化 FPGA 设备及缓冲区
 * 
 * 数据流： TextDBData -> BM25DatasetBuilder -> BM25IndexMemory
 */
class BM25DatasetBuilder : public BuildMemory<data_type::TextDBData, data_type::BM25IndexMemory> {
public:
    BM25DatasetBuilder(
        const std::string& bitstream_path = "../indexer_bm25.xclbin",
        const std::string& export_dir = "./export",
        const std::string& python_module_path = "."
    ) : bitstream_path_(bitstream_path),
        export_dir_(export_dir),
        python_module_path_(python_module_path),
        py_initialized_(false),
        fpga_setup_object_(nullptr) {}

    ~BM25DatasetBuilder() override;

    /**
     * @brief 获取已保存的 FPGA 初始化结果 Python 对象
     * 融合检索步骤使用该对象启动 FPGA 内核。
     */
    void* get_fpga_setup_object() const { return fpga_setup_object_; }

protected:
    void run_cpu_kernel(
        const data_type::TextDBData& raw_data,
        data_type::BM25IndexMemory& memory
    ) override;

    void run_gpu_kernel(
        const data_type::TextDBData& raw_data,
        data_type::BM25IndexMemory& memory
    ) override;

    void run_fpga_kernel(
        const data_type::TextDBData& raw_data,
        data_type::BM25IndexMemory& memory
    ) override;

    void run_test_kernel(
        const data_type::TextDBData& raw_data,
        data_type::BM25IndexMemory& memory
    ) override;

private:
    void ensure_python_initialized();

    std::string bitstream_path_;
    std::string export_dir_;
    std::string python_module_path_;
    bool py_initialized_;
    void* fpga_setup_object_;
};

}  // namespace step
}  // namespace heteromm

#endif  // HETEROMM_DEV_STEPS_BUILD_MEMORY_H_
