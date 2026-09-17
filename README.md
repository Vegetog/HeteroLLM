# HeteroLLM

**开发中（WIP）：之后会补充更加详细、准确的文档。**

HeteroLLM 是一个面向 **LLM 内存类工作负载（memory workloads）服务**的 AI 基础设施研究框架，可跨 **CPU、GPU 和 FPGA** 执行。

该仓库整合了：

* 强类型 C++ 流水线前端
* 基于调度策略的部署接口
* FPGA / GPU Kernel 实现
* Roofline 与设备放置建模
* RAG 实验
* FPGA-GPU 点对点（Peer-to-Peer，P2P）通信路径

其核心抽象是一条**异构内存管理流水线**：

```text
RetrievedData -> BuildMemory -> Memory
Memory + Query -> ComputeScore -> Score
Score -> MemoryRetrieval -> RetrievedIndex
RetrievedData + RetrievedIndex + TargetData -> ApplyMemory -> TargetData
```

这套抽象能够自然映射到需要以下功能的 LLM 系统：

* 检索（Retrieval）
* 稀疏注意力（Sparse Attention）
* KV Cache 索引
* BM25 / RAG 查询
* 专家路由（Expert Routing）

同时，每个流水线阶段都可以运行在最适合它的执行设备上。

## 项目结构

```text
.
├── toolchain/
│   ├── frontend/              # HeteroMM C++ API、部署管理器、Pass、测试
│   ├── backend/               # Profiling、Roofline 建模、任务分配分析
│   └── rag_test/              # BM25/RAG 流水线及 FPGA 支持的 BM25 Loader
├── kernels/
│   ├── bm25/                  # BM25 top-k 索引器 Kernel 和 Bitstream
│   ├── seerattention/         # SeerAttention 索引器的不同实现
│   ├── lserve/                # LServe 索引器的不同实现
│   ├── moe/                   # DeepSeek 风格的 MoE FPGA/GPU Kernel
│   ├── deepseek_engram/       # Engram GPU-FPGA 实验
│   └── dsa_indexer_lut/       # 基于 LUT 的 DSA 索引器
├── p2p_comm/
│   ├── u55c_rocm_p2p/         # Xilinx U55C + AMD MI210 P2P 示例
│   └── python_api/            # 面向 Python 工作流的 pybind11 P2P API
├── aws_ec2_ena/               # AWS FPGA Preview 实验产物
└── README.md
```

## 架构

HeteroLLM 被组织成一个**完整的框架栈（framework stack）**，而不是单一 Benchmark。

| 层级     | 作用                                                                                             |
| ------ | ---------------------------------------------------------------------------------------------- |
| 前端类型   | 为 Memory、Query、Score、Retrieved Index、Retrieved Data 和 Target Data 提供 C++ 数据抽象。                 |
| 流水线步骤  | 提供 `BuildMemory`、`ComputeScore`、`MemoryRetrieval` 和 `ApplyMemory` 接口，并带有 CPU/GPU/FPGA 调度 Hook。 |
| 部署管理器  | `MemoryManager` 负责组织完整流水线，并根据 JSON Schedule 选择执行设备。                                            |
| Pass   | 源代码级 Python Dispatch Pass 会检测 `PY_FUNC` 注解，并生成 Python 流水线 Wrapper。                             |
| 后端建模   | 包括 Roofline 分析、设计空间 Profiling、Kernel/Data 放置，以及 PCIe 数据传输建模。                                   |
| Kernel | 包括 TAPA/Vitis HLS FPGA Kernel、HIP/Torch GPU Kernel，以及针对特定流程预编译的 `.xclbin`。                     |
| 通信     | 使用 XRT + ROCm 管理 P2P Buffer，实现从 FPGA HBM 到 GPU 显存的通信路径。                                        |

## 主要能力

* 面向异构 LLM Serving 的强类型 C++ Memory Pipeline。
* 每个流水线阶段都可以采用静态或基于 Schedule 的 CPU/GPU/FPGA 调度。
* 内置以下示例：

  * Paged KV 索引
  * 内积打分
  * Top-k 检索
  * 阈值检索
  * Block-Sparse Attention
  * BM25 检索
  * RAG Prompt 应用
* 提供针对以下任务的 FPGA Kernel：

  * BM25
  * SeerAttention
  * LServe
  * MoE
  * Engram 风格 GPU-FPGA 异构执行
* 提供 Python 代码生成 Pass，将 C++ Pipeline Step 定义连接到 Python Runtime Pipeline。
* 使用 Roofline 和 Placement 模型帮助选择 Kernel / Device 分配方案。
* 提供 ROCm/XRT P2P Demo 和 pybind11 API，实现 FPGA-GPU 数据传输。

## 硬件与软件要求

如果只运行纯软件版本的前端测试，那么只需要：

* 支持 C++17 的编译器
* `make`

完整的异构执行栈主要围绕以下环境构建：

* Xilinx Alveo U55C，平台：

```text
xilinx_u55c_gen3x16_xdma_3_202210_1
```

* XRT
* Vitis / Vitis HLS
* TAPA
* AMD ROCm / HIP

仓库中的 ROCm/HIP 部分主要针对类似 **MI210** 的系统进行了测试。

根据具体工作流，Python 3 环境可能需要：

* `numpy`
* `pybind11`
* `bm25s`
* `datasets`
* `transformers`
* `torch`
* `vllm`
* `pyxrt`

等软件包。

部分 Makefile 默认假设工具安装在类似以下路径：

```text
/opt/xilinx/xrt
/opt/xilinx/Vitis/2024.2
/opt/rocm
```

或者假设存在 RapidStream/TAPA 环境。

如果在不同机器上构建，需要提前调整对应的环境变量。

## 快速开始

### 1. 前端纯软件测试

如果你正在开发 HeteroMM C++ Pipeline API，并且暂时不需要 FPGA/GPU 硬件，可以使用：

```bash
cd toolchain/frontend/dev/unittest
make test
```

该命令会构建并运行基于 doctest 的测试，包括：

* 内积打分
* Top-k 检索
* 阈值检索
* Paged KV Index 构建
* Block-Sparse Attention

### 2. 构建一个 C++ Pipeline Step

```cpp
#include "dev/dev.h"

using namespace heteromm;

int main() {
    std::vector<std::vector<float>> mem_data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    data_type::FlatIndexMemory<float> memory(mem_data);

    std::vector<float> query_data = {1.0f, 1.0f};
    data_type::VectorQuery<float> query(query_data);

    data_type::VectorScore<float> score({});
    step::InnerProductCompute compute;
    compute.set_current_kernel(step::KernelType::CPU);
    compute.execute(memory, query, score);

    data_type::TopKIndex indices({});
    step::TopKRetrieval retrieval(1);
    retrieval.execute(score, indices);
}
```

### 3. 运行 RAG 原型

```bash
cd toolchain/rag_test
pip install -r requirements.txt
python rag_pipeline.py --mode simple --question "What is machine learning?"
```

RAG 流程使用：

* **BM25S** 进行检索
* HuggingFace / vLLM 进行生成

如果系统中具有 XRT/PyXRT 和兼容的 Bitstream，那么 BM25 阶段可以由 FPGA 执行。

### 4. 编译并运行 BM25 Loader 测试

```bash
cd toolchain/rag_test
make
make csim
```

如果运行 FPGA 版本：

```bash
make run_xrt
```

硬件路径需要：

* 一个 BM25 `.xclbin`
* 导出的 BM25 数据

数据应位于：

```text
toolchain/rag_test/export
```

### 5. 尝试 FPGA-GPU P2P 数据传输

```bash
cd p2p_comm/u55c_rocm_p2p
source env.sh
make simple
./p2p_simple --fpga 81:00.1 --gpu 0 --size 64
```

该流程会：

1. 在 FPGA 上创建 XRT P2P Buffer。
2. 将这些 Buffer 注册到 ROCm。
3. 使 GPU Kernel 能够通过 PCIe：

   * 从 FPGA HBM 读取数据
   * 或向 FPGA HBM 写入数据

### 6. 安装 Python P2P API

```bash
cd p2p_comm/python_api
pip install -e .
python examples/basic_transfer.py --fpga-bdf 81:00.1
```

该 Python 包通过 `heteromem_p2p` 模块暴露以下能力：

* `FPGADevice`
* `GPUDevice`
* P2P Buffer 操作

## 基于 Schedule 的部署

文件：

```text
toolchain/frontend/deploy/memory_manager.h
```

提供了一个可复用的 `MemoryManager` 模板，并暴露三个公共入口：

* `build_memory(...)`

  从原始 Retrieved Data 构建 Index 或 Memory 数据结构。

* `manage_memory_and_apply(...)`

  在已有 Memory 上执行 Query、Retrieval 和 Apply 阶段。

* `build_and_apply_memory(...)`

  执行完整流水线。

Schedule 使用 JSON 规则，将不同问题规模映射到不同设备选择。

例如：

```json
{
  "manage_memory_and_apply": [
    {
      "retrieved_data": 4096,
      "memory": 4096,
      "query": 1,
      "output": 1,
      "config": ["fpga", "fpga", "gpu"]
    }
  ]
}
```

仓库内置的示例位于：

```text
toolchain/frontend/deploy/schedule.json
```

## Python Dispatch Pass

前端中包含一个源代码级 Pass，可以检测类似下面的注解：

```cpp
PY_FUNC("launch_bm25.fpga_retriver_launch")
void FusedBM25Retrieval::run_fpga_kernel(...);
```

它可以生成 Python Pipeline Wrapper。

对于带有注解的 Step，Wrapper 会调用原生 Python 函数；

对于没有注解的 Step，则调用通过 pybind11 导出的 C++ 模块。

构建：

```bash
cd toolchain/frontend/dev/passes
make
```

生成结果会写入：

```text
toolchain/frontend/dev/passes/generated/
```

## 建模与 Profiling

后端工具可以对异构执行方案进行估算和搜索：

```bash
cd toolchain/backend/modeling
python roofline_analyzer.py --profile-target profile_innerproduct --json
python optimal_assignment.py
```

相关输入数据位于：

```text
toolchain/backend/modeling/fpga_config/
toolchain/backend/modeling/kernel_profile_results/
toolchain/backend/modeling/pcie_config/
toolchain/backend/profiling/design_space.json
```

## Kernel 类型

| 目录                        | 描述                                                              |
| ------------------------- | --------------------------------------------------------------- |
| `kernels/bm25`            | FPGA BM25 Top-k Indexer 及 Testbench 相关文件。                       |
| `kernels/seerattention`   | SeerAttention Threshold Indexer 和 Token-Budget Indexer。         |
| `kernels/lserve`          | LServe Indexer 的不同版本，包括长上下文版本。                                  |
| `kernels/moe`             | DeepSeek 风格的 MoE FPGA Kernel、INT8 Decode 版本，以及 TileLang GPU 实验。 |
| `kernels/deepseek_engram` | Engram 风格 GPU-FPGA 异构执行及 DeepSeek V3 Benchmark。                 |
| `kernels/dsa_indexer_lut` | 基于 LUT 的 DSA Indexer 实现。                                        |

大多数 FPGA Kernel 目录使用 TAPA/Vitis 构建目标，例如：

```text
csim
hls
xclbin
```

硬件综合目标可能需要数小时才能完成。

## P2P 通信

`p2p_comm/u55c_rocm_p2p` 展示了：

* GPU 从 FPGA HBM 中读取数据
* GPU 向 FPGA P2P Buffer 写入数据
* GPU 上运行的 SpMV Demo，直接消费 FPGA 生成的 BM25 Index
* 使用 FPGA 端验证 Kernel 检查 GPU 写入的数据

常见系统检查命令：

```bash
xbutil examine -d 81:00.1 --report platform
rocm-smi --showbus
lspci | grep -i xilinx
```

如果真正的 P2P Buffer 注册失败，Demo 可以退化为使用 Host Staging 的 Mapped Memory。

这种方式仍然能够正常工作，但带宽会更低。

## 开发说明

* 对纯软件部分进行修改时，应确保有：

```text
toolchain/frontend/dev/unittest
```

中的测试覆盖。

* 修改硬件代码后，在运行耗时较长的综合任务之前，应先运行 C Simulation。

* 不要假设默认工具路径具有可移植性。

  大多数 Makefile 都暴露了类似以下变量：

```text
XILINX_XRT
XILINX_VITIS
ROCM_PATH
```

以及其他相关变量。

* 仓库中提交了一些 `.xclbin` 文件作为实验产物。

  如果需要重新编译这些文件，必须使用匹配的 FPGA Platform 和 Toolchain 版本。

* 当前该仓库尚未指定顶层 License。

## 更多文档

* `toolchain/frontend/README.md`

  HeteroMM 前端 API 详细说明。

* `toolchain/frontend/dev/passes/README.md`

  Python Dispatch Pass 的设计说明。

* `toolchain/rag_test/README.md`

  BM25 / RAG 实验的使用方式。

* `p2p_comm/u55c_rocm_p2p/README.md`

  FPGA-GPU P2P 环境配置及故障排查。

* `p2p_comm/python_api/README.md`

  pybind11 P2P API 使用说明。
