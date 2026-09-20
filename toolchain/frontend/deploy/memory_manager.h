/**
 * @file memory_manager.h
 * @brief 用于组织流水线执行的 MemoryManager 基类
 * 
 * 提供构建记忆、管理检索和应用记忆的主要接口，
 * 将这些操作组织到统一的工作流程中。
 */

#ifndef HETEROMM_DEPLOY_MEMORY_MANAGER_H_
#define HETEROMM_DEPLOY_MEMORY_MANAGER_H_

// 引入 dev 目录中的步骤定义
#include "../dev/steps/util.h"
#include "../dev/steps/build_memory.h"
#include "../dev/steps/compute_score.h"
#include "../dev/steps/memory_retrieval.h"
#include "../dev/steps/apply_memory.h"

// 引入 dev 目录中的数据类型定义
#include "../dev/types/base_types.h"
#include "../dev/types/memory.h"
#include "../dev/types/query.h"
#include "../dev/types/score.h"
#include "../dev/types/retrieved_index.h"
#include "../dev/types/retrieved_data.h"
#include "../dev/types/target_data.h"

#include "utils.h"

#include <memory>
#include <stdexcept>
#include <chrono>
#include <iostream>

namespace heteromm {
namespace deploy {

/**
 * @brief 流水线各步骤的内核类型配置
 */
struct PipelineKernelConfig {
    step::KernelType build_memory = step::KernelType::CPU;
    step::KernelType compute_score = step::KernelType::CPU;
    step::KernelType memory_retrieval = step::KernelType::CPU;
    step::KernelType apply_memory = step::KernelType::CPU;
    
    PipelineKernelConfig() = default;
    
    PipelineKernelConfig(step::KernelType build, step::KernelType compute,
                         step::KernelType retrieval, step::KernelType apply)
        : build_memory(build), compute_score(compute),
          memory_retrieval(retrieval), apply_memory(apply) {}
};

/**
 * @brief 记忆管理器操作的执行结果
 */
struct ExecutionResult {
    bool success = false;
    std::string error_message;
    double total_time_ms = 0.0;
    
    // 各步骤耗时明细
    double build_memory_time_ms = 0.0;
    double compute_score_time_ms = 0.0;
    double retrieval_time_ms = 0.0;
    double apply_memory_time_ms = 0.0;
    
    static ExecutionResult Success(double time_ms = 0.0) {
        ExecutionResult r;
        r.success = true;
        r.total_time_ms = time_ms;
        return r;
    }
    
    static ExecutionResult Failure(const std::string& msg) {
        ExecutionResult r;
        r.success = false;
        r.error_message = msg;
        return r;
    }
};

/**
 * @brief 记忆管理器基类
 * 
 * 本类负责组织完整的检索增强流水线：
 * 1. build_memory: RetrievedData -> Memory
 * 2. compute_score: (Memory, Query) -> Score
 * 3. memory_retrieval: Score -> RetrievedIndex
 * 4. apply_memory: (RetrievedData, RetrievedIndex, TargetData) -> TargetData
 * 
 * 提供三个主要入口：
 * - build_memory(): 仅构建记忆结构
 * - manage_memory_and_apply(): 使用已构建的记忆，执行从查询到输出的流程
 * - build_and_apply_memory(): 执行从原始数据到输出的完整流水线
 * 
 * 用户可以通过继承本基类创建新类，以实现：
 * - 提供自定义的步骤实现对象
 * - 配置每个步骤的内核类型
 * - 添加日志、性能分析或其他自定义功能
 * 
 * 三个主要函数供调用使用，不应重新实现。
 * 自定义功能应通过步骤实现对象和工厂方法完成。
 * 
 * @tparam RetDataT 派生自 data_type::RetrievedData 的类型
 * @tparam MemoryT 派生自 data_type::Memory 的类型
 * @tparam QueryT 派生自 data_type::Query 的类型
 * @tparam ScoreT 派生自 data_type::Score 的类型
 * @tparam IndexT 派生自 data_type::RetrievedIndex 的类型
 * @tparam InputT 派生自 data_type::TargetData 的类型，作为 apply_memory 的输入
 * @tparam OutputT 派生自 data_type::TargetData 的类型，作为 apply_memory 的输出
 */
template<typename RetDataT, typename MemoryT, typename QueryT,
         typename ScoreT, typename IndexT, typename InputT, typename OutputT>
class MemoryManager {
public:
    // 为便于使用而定义的类型别名
    using RetrievedData = RetDataT;
    using Memory = MemoryT;
    using Query = QueryT;
    using Score = ScoreT;
    using Index = IndexT;
    using Input = InputT;
    using Output = OutputT;
    
    // 使用 dev 目录中的步骤类定义各阶段的类型
    using BuildMemoryStepT = step::BuildMemory<RetDataT, MemoryT>;
    using ComputeScoreStepT = step::ComputeScore<MemoryT, QueryT, ScoreT>;
    using MemoryRetrievalStepT = step::MemoryRetrieval<ScoreT, IndexT>;
    using ApplyMemoryStepT = step::ApplyMemory<RetDataT, IndexT, InputT, OutputT>;
    
    /**
     * @brief 默认构造函数：所有步骤的内核类型初始化为 CPU
     * @param schedule_path 调度配置 JSON 文件的路径（可选）
     */
    explicit MemoryManager(const std::string& schedule_path = "") 
        : kernel_config_(){
        if (!schedule_path.empty()) {
            schedule_config_ = ScheduleConfig::load_from_file(schedule_path);
        }
    }
    
    /**
     * @brief 接收内核配置的构造函数
     * @param schedule_path 调度配置 JSON 文件的路径
     * @param config 流水线每个步骤的内核配置
     */
    MemoryManager(const std::string& schedule_path, const PipelineKernelConfig& config) 
        : kernel_config_(config){
        if (!schedule_path.empty()) {
            schedule_config_ = ScheduleConfig::load_from_file(schedule_path);
        }
    }
    
    virtual ~MemoryManager() = default;
    
    // ===== 内核配置设置函数 =====
    
    /**
     * @brief 设置所有步骤的内核配置
     * @param config 内核配置
     */
    void set_kernel_config(const PipelineKernelConfig& config) {
        kernel_config_ = config;
    }
    
    /**
     * @brief 设置 build_memory 步骤的内核类型
     * @param type 内核类型（CPU、GPU 或 FPGA）
     */
    void set_build_memory_kernel(step::KernelType type) {
        kernel_config_.build_memory = type;
    }
    
    /**
     * @brief 设置 compute_score 步骤的内核类型
     * @param type 内核类型（CPU、GPU 或 FPGA）
     */
    void set_compute_score_kernel(step::KernelType type) {
        kernel_config_.compute_score = type;
    }
    
    /**
     * @brief 设置 memory_retrieval 步骤的内核类型
     * @param type 内核类型（CPU、GPU 或 FPGA）
     */
    void set_memory_retrieval_kernel(step::KernelType type) {
        kernel_config_.memory_retrieval = type;
    }
    
    /**
     * @brief 设置 apply_memory 步骤的内核类型
     * @param type 内核类型（CPU、GPU 或 FPGA）
     */
    void set_apply_memory_kernel(step::KernelType type) {
        kernel_config_.apply_memory = type;
    }
    
    /**
     * @brief 获取当前内核配置
     * @return 当前内核配置
     */
    const PipelineKernelConfig& get_kernel_config() const {
        return kernel_config_;
    }
    
    /**
     * @brief 获取调度配置
     * @return 调度配置
     */
    const ScheduleConfig& get_schedule_config() const {
        return schedule_config_;
    }
    
    // ===== 步骤实现对象的设置函数（用于自定义） =====
    
    /**
     * @brief 设置构建记忆步骤的实现对象
     */
    void set_build_memory_step(std::shared_ptr<BuildMemoryStepT> step) {
        build_memory_step_ = std::move(step);
    }
    
    /**
     * @brief 设置相关性计算步骤的实现对象
     */
    void set_compute_score_step(std::shared_ptr<ComputeScoreStepT> step) {
        compute_score_step_ = std::move(step);
    }
    
    /**
     * @brief 设置记忆检索步骤的实现对象
     */
    void set_memory_retrieval_step(std::shared_ptr<MemoryRetrievalStepT> step) {
        memory_retrieval_step_ = std::move(step);
    }
    
    /**
     * @brief 设置记忆应用步骤的实现对象
     */
    void set_apply_memory_step(std::shared_ptr<ApplyMemoryStepT> step) {
        apply_memory_step_ = std::move(step);
    }
    
    // ===== 工厂方法（通过重写提供自定义步骤实现对象） =====
    
    /**
     * @brief 创建构建记忆步骤的实现对象
     * 重写此方法以提供自定义实现。
     */
    virtual std::shared_ptr<BuildMemoryStepT> create_build_memory_step() {
        return nullptr;  // 必须重写此方法，或手动设置步骤实现对象
    }
    
    /**
     * @brief 创建相关性计算步骤的实现对象
     */
    virtual std::shared_ptr<ComputeScoreStepT> create_compute_score_step() {
        return nullptr;
    }
    
    /**
     * @brief 创建记忆检索步骤的实现对象
     */
    virtual std::shared_ptr<MemoryRetrievalStepT> create_memory_retrieval_step() {
        return nullptr;
    }
    
    /**
     * @brief 创建记忆应用步骤的实现对象
     */
    virtual std::shared_ptr<ApplyMemoryStepT> create_apply_memory_step() {
        return nullptr;
    }

    virtual int ret_data_size(RetDataT& retrieved_data) = 0;
    virtual int memory_size(MemoryT& memory) = 0;
    
    // ===== 主要 API 函数（供调用使用，不应重新实现） =====
    
    /**
     * @brief 根据检索数据构建记忆结构
     * 
     * 本函数仅执行 build_memory 步骤。
     * 当记忆构建是独立的过程，
     * 并且可能离线执行时，使用此函数。
     * 
     * 注意：调用本函数前，应通过 set_build_memory_kernel()
     * 或 set_kernel_config() 配置内核类型。
     * 
     * @param retrieved_data 输入的检索数据
     * @param memory 输出的记忆结构
     * @param verbose 是否启用详细日志
     * @return 执行结果
     */
    ExecutionResult build_memory(
            const RetDataT& retrieved_data,
            MemoryT& memory,
            bool verbose = false) {

        int ret_data_size_val = ret_data_size(retrieved_data);
        int memory_size_val = memory_size(memory);
        auto config = schedule_config_.get_build_memory_config(ret_data_size_val, memory_size_val);
        set_build_memory_kernel(step::string_to_kernel_type(config));
        
        auto step = get_or_create_build_memory_step();
        if (!step) {
            return ExecutionResult::Failure("BuildMemory step not configured");
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        int status = step->execute(retrieved_data, memory, false, verbose);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        if (status != 0) {
            return ExecutionResult::Failure("BuildMemory execution failed");
        }
        
        ExecutionResult result = ExecutionResult::Success(time_ms);
        result.build_memory_time_ms = time_ms;
        return result;
    }
    
    /**
     * @brief 管理并应用记忆，生成输出
     * 
     * 本函数执行以下步骤：
     * - compute_score: (memory, query) -> scores
     * - memory_retrieval: scores -> indices
     * - apply_memory: (retrieved_data, indices, input) -> output
     * 
     * 当记忆已经构建完成时，使用此函数。
     * 
     * 注意：调用本函数前，应通过 set_*_kernel()
     * 或 set_kernel_config() 配置内核类型。
     * 
     * @param retrieved_data 原始检索数据，供 apply_memory 使用
     * @param memory 预先构建的记忆结构
     * @param query 用于检索的查询
     * @param input 额外输入数据
     * @param output 输出结果
     * @param verbose 是否启用详细日志
     * @return 执行结果
     */
    ExecutionResult manage_memory_and_apply(
            const RetDataT& retrieved_data,
            const MemoryT& memory,
            const QueryT& query,
            const InputT& input,
            OutputT& output,
            bool verbose = false) {
        
        ExecutionResult result;
        result.success = true;

        // 获取配置
        int ret_data_size_val = ret_data_size(retrieved_data);
        int memory_size_val = memory_size(memory);
        auto config = schedule_config_.get_manage_memory_and_apply_config(ret_data_size_val, memory_size_val);
        set_compute_score_kernel(step::string_to_kernel_type(config[0]));
        set_memory_retrieval_kernel(step::string_to_kernel_type(config[1]));
        set_apply_memory_kernel(step::string_to_kernel_type(config[2]));
        
        // 本函数的第 1 步：计算相关性分数（完整流水线的第 2 步）
        auto score_step = get_or_create_compute_score_step();
        if (!score_step) {
            return ExecutionResult::Failure("ComputeScore step not configured");
        }
        
        ScoreT scores;
        auto score_start = std::chrono::high_resolution_clock::now();
        int score_status = score_step->execute(memory, query, scores, false, verbose);
        auto score_end = std::chrono::high_resolution_clock::now();
        
        if (score_status != 0) {
            return ExecutionResult::Failure("ComputeScore execution failed");
        }
        result.compute_score_time_ms = std::chrono::duration<double, std::milli>(
            score_end - score_start).count();
        
        // 本函数的第 2 步：检索记忆（完整流水线的第 3 步）
        auto retrieval_step = get_or_create_memory_retrieval_step();
        if (!retrieval_step) {
            return ExecutionResult::Failure("MemoryRetrieval step not configured");
        }
        
        IndexT indices;
        auto retr_start = std::chrono::high_resolution_clock::now();
        int retr_status = retrieval_step->execute(scores, indices, false, verbose);
        auto retr_end = std::chrono::high_resolution_clock::now();
        
        if (retr_status != 0) {
            return ExecutionResult::Failure("MemoryRetrieval execution failed");
        }
        result.retrieval_time_ms = std::chrono::duration<double, std::milli>(
            retr_end - retr_start).count();
        
        // 本函数的第 3 步：应用记忆（完整流水线的第 4 步）
        auto apply_step = get_or_create_apply_memory_step();
        if (!apply_step) {
            return ExecutionResult::Failure("ApplyMemory step not configured");
        }
        
        auto apply_start = std::chrono::high_resolution_clock::now();
        int apply_status = apply_step->execute(retrieved_data, indices, input, output, false, verbose);
        auto apply_end = std::chrono::high_resolution_clock::now();
        
        if (apply_status != 0) {
            return ExecutionResult::Failure("ApplyMemory execution failed");
        }
        result.apply_memory_time_ms = std::chrono::duration<double, std::milli>(
            apply_end - apply_start).count();
        
        result.total_time_ms = result.compute_score_time_ms + 
                               result.retrieval_time_ms + 
                               result.apply_memory_time_ms;
        
        return result;
    }
    
    /**
     * @brief 一次调用完成记忆构建与应用
     * 
     * 本函数执行完整流水线：
     * - build_memory: retrieved_data -> memory
     * - compute_score: (memory, query) -> scores
     * - memory_retrieval: scores -> indices
     * - apply_memory: (retrieved_data, indices, input) -> output
     * 
     * 注意：调用本函数前，应通过 set_*_kernel()
     * 或 set_kernel_config() 配置内核类型。
     * 
     * @param retrieved_data 输入的检索数据
     * @param query 用于检索的查询
     * @param input 额外输入数据
     * @param output 输出结果
     * @param verbose 是否启用详细日志
     * @return 执行结果
     */
    ExecutionResult build_and_apply_memory(
            const RetDataT& retrieved_data,
            const QueryT& query,
            const InputT& input,
            OutputT& output,
            bool verbose = false) {

        int ret_data_size_val = ret_data_size(retrieved_data);
        int memory_size_val = memory_size(memory);
        auto config = schedule_config_.get_build_memory_and_apply_config(ret_data_size_val, memory_size_val);
        set_build_memory_kernel(step::string_to_kernel_type(config[0]));
        set_compute_score_kernel(step::string_to_kernel_type(config[1]));
        set_memory_retrieval_kernel(step::string_to_kernel_type(config[2]));
        set_apply_memory_kernel(step::string_to_kernel_type(config[3]));
        
        // 构建记忆
        MemoryT memory;
        auto build_result = build_memory(retrieved_data, memory, verbose);
        if (!build_result.success) {
            return build_result;
        }
        
        // 执行其余步骤
        auto apply_result = manage_memory_and_apply(
            retrieved_data, memory, query, input, output, verbose);
        
        if (!apply_result.success) {
            return apply_result;
        }
        
        // 合并执行结果
        ExecutionResult result = apply_result;
        result.build_memory_time_ms = build_result.build_memory_time_ms;
        result.total_time_ms = build_result.total_time_ms + apply_result.total_time_ms;
        
        return result;
    }
    
    /**
     * @brief 对所有已配置的步骤执行功能测试
     * 
     * 使用各步骤的测试内核进行计算，并与预期的正确结果比较。
     * 
     * @param retrieved_data 测试输入数据
     * @param expected_memory 预期的记忆输出
     * @param query 测试查询
     * @param expected_scores 预期的分数输出
     * @param expected_indices 预期的索引输出
     * @param input apply_memory 的测试输入
     * @param expected_output 预期的最终输出
     * @param verbose 是否启用详细日志
     * @return 所有测试通过时返回 true
     */
    bool run_functional_tests(
            const RetDataT& retrieved_data,
            MemoryT& expected_memory,
            const QueryT& query,
            ScoreT& expected_scores,
            IndexT& expected_indices,
            const InputT& input,
            OutputT& expected_output,
            bool verbose = false) {
        
        bool all_passed = true;
        
        // 测试 build_memory
        auto build_step = get_or_create_build_memory_step();
        if (build_step) {
            int status = build_step->execute(retrieved_data, expected_memory, true, verbose);
            if (status != 0) {
                if (verbose) std::clog << "[MemoryManager] BuildMemory test FAILED" << std::endl;
                all_passed = false;
            } else if (verbose) {
                std::clog << "[MemoryManager] BuildMemory test PASSED" << std::endl;
            }
        }
        
        // 测试 compute_score
        auto score_step = get_or_create_compute_score_step();
        if (score_step) {
            int status = score_step->execute(expected_memory, query, expected_scores, true, verbose);
            if (status != 0) {
                if (verbose) std::clog << "[MemoryManager] ComputeScore test FAILED" << std::endl;
                all_passed = false;
            } else if (verbose) {
                std::clog << "[MemoryManager] ComputeScore test PASSED" << std::endl;
            }
        }
        
        // 测试 memory_retrieval
        auto retrieval_step = get_or_create_memory_retrieval_step();
        if (retrieval_step) {
            int status = retrieval_step->execute(expected_scores, expected_indices, true, verbose);
            if (status != 0) {
                if (verbose) std::clog << "[MemoryManager] MemoryRetrieval test FAILED" << std::endl;
                all_passed = false;
            } else if (verbose) {
                std::clog << "[MemoryManager] MemoryRetrieval test PASSED" << std::endl;
            }
        }
        
        // 测试 apply_memory
        auto apply_step = get_or_create_apply_memory_step();
        if (apply_step) {
            int status = apply_step->execute(retrieved_data, expected_indices, input, 
                                              expected_output, true, verbose);
            if (status != 0) {
                if (verbose) std::clog << "[MemoryManager] ApplyMemory test FAILED" << std::endl;
                all_passed = false;
            } else if (verbose) {
                std::clog << "[MemoryManager] ApplyMemory test PASSED" << std::endl;
            }
        }
        
        return all_passed;
    }

protected:
    // 各步骤的实现对象
    std::shared_ptr<BuildMemoryStepT> build_memory_step_;
    std::shared_ptr<ComputeScoreStepT> compute_score_step_;
    std::shared_ptr<MemoryRetrievalStepT> memory_retrieval_step_;
    std::shared_ptr<ApplyMemoryStepT> apply_memory_step_;
    
    // 内核配置
    PipelineKernelConfig kernel_config_;
    
    // 调度配置
    ScheduleConfig schedule_config_;
    
    // 辅助函数：获取或创建步骤实现对象，并设置其内核配置
    std::shared_ptr<BuildMemoryStepT> get_or_create_build_memory_step() {
        if (!build_memory_step_) {
            build_memory_step_ = create_build_memory_step();
        }
        if (build_memory_step_) {
            build_memory_step_->set_current_kernel(kernel_config_.build_memory);
        }
        return build_memory_step_;
    }
    
    std::shared_ptr<ComputeScoreStepT> get_or_create_compute_score_step() {
        if (!compute_score_step_) {
            compute_score_step_ = create_compute_score_step();
        }
        if (compute_score_step_) {
            compute_score_step_->set_current_kernel(kernel_config_.compute_score);
        }
        return compute_score_step_;
    }
    
    std::shared_ptr<MemoryRetrievalStepT> get_or_create_memory_retrieval_step() {
        if (!memory_retrieval_step_) {
            memory_retrieval_step_ = create_memory_retrieval_step();
        }
        if (memory_retrieval_step_) {
            memory_retrieval_step_->set_current_kernel(kernel_config_.memory_retrieval);
        }
        return memory_retrieval_step_;
    }
    
    std::shared_ptr<ApplyMemoryStepT> get_or_create_apply_memory_step() {
        if (!apply_memory_step_) {
            apply_memory_step_ = create_apply_memory_step();
        }
        if (apply_memory_step_) {
            apply_memory_step_->set_current_kernel(kernel_config_.apply_memory);
        }
        return apply_memory_step_;
    }
};

}  // namespace deploy
}  // namespace heteromm

#endif  // HETEROMM_DEPLOY_MEMORY_MANAGER_H_
