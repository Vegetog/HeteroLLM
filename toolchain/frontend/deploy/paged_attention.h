/**
 * @file paged_attention.h
 * @brief 用于分块稀疏注意力的分页注意力内存管理器
 * 
 * 该文件提供一个内存管理器，将下列步骤串联起来：
 * PagedKVIndexBuilder -> InnerProductCompute -> TopKRetrieval -> BlockSparseAttention
 * 
 * 这条流水线实现分块稀疏注意力：
 * 1. 将 KV Cache 按页切分，并为每一页生成轻量索引
 * 2. 通过内积计算 Query 与各页索引的相关性
 * 3. 选出相关性最高的 Top-K 个页
 * 4. 仅对被选中页内的 K/V 计算最终注意力
 *
 * 注意：页索引的内积只用于粗筛选，不是最终的 QK^T 注意力分数。
 */

#ifndef HETEROMM_DEPLOY_PAGED_ATTENTION_H_
#define HETEROMM_DEPLOY_PAGED_ATTENTION_H_

#include "memory_manager.h"
#include <cmath>

namespace heteromm {
namespace deploy {

/**
 * @brief 分页注意力内存管理器的类型别名
 * 
 * 数据流：
 * - RawData (RetrievedData): KVCacheData<float> - 原始 KV Cache
 * - Memory: FlatIndexMemory<float> - 由每个页的 Key 生成的分页索引
 * - Query: VectorQuery<float> - 用于页级粗筛选的查询向量
 * - Score: VectorScore<float> - Query 与每个页索引的相似度
 * - Index: TopKIndex - 最相关的 Top-K 个页编号
 * - Input: VectorInputOutputData<float> - 计算最终注意力时使用的 Query
 * - Output: VectorInputOutputData<float> - 稀疏注意力输出
 */
using PagedAttentionManager = MemoryManager<
    data_type::KVCacheData<float>,           // RetrievedData：原始 KV Cache
    data_type::FlatIndexMemory<float>,       // Memory：分页 Key 索引
    data_type::VectorQuery<float>,           // Query：页级粗筛选查询向量
    data_type::VectorScore<float>,           // Score：每个页的相关性分数
    data_type::TopKIndex,                    // Index：Top-K 页编号
    data_type::VectorInputOutputData<float>, // Input：最终注意力查询向量
    data_type::VectorInputOutputData<float>  // Output：稀疏注意力结果
>;

/**
 * @brief 带默认步骤实现的分页注意力管理器
 * 
 * 该类将四个步骤组装成一条完整流水线，并允许配置
 * 页大小、投影权重和 Top-K 数量。
 * 
 * 使用示例：
 * @code
 *   // 创建 page_size=16、top_k=8 的管理器
 *   auto weight = create_random_projection_weight(head_dim, proj_dim);
 *   PagedAttention attn("schedule.json", weight, 16, 8);
 *   
 *   // 从 KV Cache 构建分页索引，通常可在 Query 到来前完成
 *   KVCacheData<float> kv_cache = ...;
 *   FlatIndexMemory<float> index;
 *   attn.build_memory(kv_cache, index);
 *   
 *   // 运行页级打分、Top-K 检索和最终稀疏注意力
 *   VectorQuery<float> query = ...;
 *   VectorInputOutputData<float> input = ...;
 *   VectorInputOutputData<float> output;
 *   attn.manage_memory_and_apply(kv_cache, index, query, input, output);
 * @endcode
 */
class PagedAttention : public PagedAttentionManager {
public:
    /**
     * @brief 构造并完整配置一个分页注意力管理器
     * 
     * @param schedule_path 调度规则 JSON 文件路径
     * @param projection_weight 将池化后 Key 投影到索引空间的权重矩阵 [proj_dim x head_dim]
     * @param page_size 每个页包含的 token 数，也是平均池化的粒度
     * @param top_k 稀疏注意力需要选取的页数
     */
    PagedAttention(
        const std::string& schedule_path,
        const std::vector<std::vector<float>>& projection_weight,
        size_t page_size,
        size_t top_k
    ) : PagedAttentionManager(schedule_path),
        projection_weight_(projection_weight),
        page_size_(page_size),
        top_k_(top_k) {}
    
    /**
     * @brief 不初始化具体参数的默认构造函数
     * 使用前需通过各 set_* 方法补齐参数。
     * @param schedule_path 调度规则 JSON 文件路径（可选）
     */
    explicit PagedAttention(const std::string& schedule_path = "") 
        : PagedAttentionManager(schedule_path) {}
    
    ~PagedAttention() override = default;
    
    // ===== 参数访问与修改 =====
    
    void set_projection_weight(const std::vector<std::vector<float>>& weight) {
        projection_weight_ = weight;
        // 建索引步骤内部持有投影权重，需重建才能使用新权重。
        build_memory_step_.reset();
    }
    
    void set_page_size(size_t page_size) {
        page_size_ = page_size;
        // 建索引和展开 Top-K 页都依赖 page_size，因此两个步骤都需重建。
        build_memory_step_.reset();
        apply_memory_step_.reset();
    }
    
    void set_top_k(size_t top_k) {
        top_k_ = top_k;
        // Top-K 是检索步骤的构造参数，修改后需重建该步骤。
        memory_retrieval_step_.reset();
    }
    
    size_t page_size() const { return page_size_; }
    size_t top_k() const { return top_k_; }
    const std::vector<std::vector<float>>& projection_weight() const { 
        return projection_weight_; 
    }

    int ret_data_size(
        const data_type::KVCacheData<float>& retrieved_data
    ) {
        return retrieved_data.get_context_length();
    }

    int memory_size(
        const data_type::FlatIndexMemory<float>& memory
    ) {
        return memory.get_num_entries();
    }

protected:
    // ===== 工厂方法：为四阶段流水线创建具体步骤 =====
    
    /**
     * @brief 创建 PagedKVIndexBuilder（Prepare Memory）
     * 对每个页内的 Key 做平均池化和线性投影，生成一个轻量页索引。
     */
    std::shared_ptr<BuildMemoryStepT> create_build_memory_step() override {
        return std::make_shared<step::PagedKVIndexBuilder>(
            projection_weight_, 
            static_cast<int>(page_size_)
        );
    }
    
    /**
     * @brief 创建 InnerProductCompute（Compute Relevancy）
     * 计算 Query 与所有页索引的内积，为每个页产生一个粗筛选分数。
     */
    std::shared_ptr<ComputeScoreStepT> create_compute_score_step() override {
        return std::make_shared<step::InnerProductCompute>();
    }
    
    /**
     * @brief 创建 TopKRetrieval（Retrieval）
     * 从页级分数中选出最高的 Top-K 个页编号。
     */
    std::shared_ptr<MemoryRetrievalStepT> create_memory_retrieval_step() override {
        return std::make_shared<step::TopKRetrieval>(top_k_);
    }
    
    /**
     * @brief 创建 BlockSparseAttention（Apply Memory）
     * 只展开 Top-K 页内的 token，并使用它们的 K/V 计算最终注意力。
     */
    std::shared_ptr<ApplyMemoryStepT> create_apply_memory_step() override {
        return std::make_shared<step::BlockSparseAttention>(page_size_);
    }

private:
    std::vector<std::vector<float>> projection_weight_;
    size_t page_size_ = 16;
    size_t top_k_ = 8;
};

/**
 * @brief 用于创建已配置分页注意力管理器的 Builder
 * 
 * 提供链式调用接口，将调度文件、投影权重、页大小和 Top-K
 * 参数集中后再构造 PagedAttention。
 * 
 * 使用示例：
 * @code
 *   auto attn = PagedAttentionBuilder()
 *       .with_schedule_path("schedule.json")
 *       .with_projection_weight(weight)
 *       .with_page_size(32)
 *       .with_top_k(16)
 *       .build();
 * @endcode
 */
class PagedAttentionBuilder {
public:
    PagedAttentionBuilder() = default;
    
    /**
     * @brief 设置调度规则文件路径
     */
    PagedAttentionBuilder& with_schedule_path(const std::string& path) {
        schedule_path_ = path;
        return *this;
    }
    
    /**
     * @brief 设置页索引的投影权重矩阵
     */
    PagedAttentionBuilder& with_projection_weight(
        const std::vector<std::vector<float>>& weight
    ) {
        projection_weight_ = weight;
        return *this;
    }
    
    /**
     * @brief 设置页大小（每页 token 数）
     */
    PagedAttentionBuilder& with_page_size(size_t page_size) {
        page_size_ = page_size;
        return *this;
    }
    
    /**
     * @brief 设置需要选取的 Top-K 页数
     */
    PagedAttentionBuilder& with_top_k(size_t top_k) {
        top_k_ = top_k;
        return *this;
    }
    
    /**
     * @brief 根据当前累积的参数构造并返回管理器
     */
    std::shared_ptr<PagedAttention> build() {
        return std::make_shared<PagedAttention>(
            schedule_path_,
            projection_weight_,
            page_size_,
            top_k_
        );
    }

private:
    std::string schedule_path_;
    std::vector<std::vector<float>> projection_weight_;
    size_t page_size_ = 16;
    size_t top_k_ = 8;
};

/**
 * @brief 创建一个分页注意力 Builder
 */
inline PagedAttentionBuilder create_paged_attention() {
    return PagedAttentionBuilder();
}

/**
 * @brief 创建随机投影权重矩阵的辅助函数
 * 
 * @param head_dim 输入维度（Attention head 维度）
 * @param proj_dim 输出维度（页索引的投影维度）
 * @param seed 用于复现结果的随机种子
 * @return 形状为 [proj_dim x head_dim] 的权重矩阵
 */
inline std::vector<std::vector<float>> create_random_projection_weight(
    size_t head_dim,
    size_t proj_dim,
    unsigned int seed = 42
) {
    std::vector<std::vector<float>> weight(proj_dim, std::vector<float>(head_dim));
    
    // 使用简单的 LCG 伪随机数生成器，保证给定 seed 时结果可复现。
    unsigned int state = seed;
    auto next_random = [&state]() -> float {
        state = state * 1103515245 + 12345;
        return static_cast<float>((state >> 16) & 0x7FFF) / 32767.0f - 0.5f;
    };
    
    // Xavier/Glorot 初始化缩放系数，用于控制投影权重的数值范围。
    float scale = std::sqrt(2.0f / (head_dim + proj_dim));
    
    for (size_t i = 0; i < proj_dim; ++i) {
        for (size_t j = 0; j < head_dim; ++j) {
            weight[i][j] = next_random() * scale;
        }
    }
    
    return weight;
}

}  // namespace deploy
}  // namespace heteromm

#endif  // HETEROMM_DEPLOY_PAGED_ATTENTION_H_
