/**
 * @file bm25_dataset_builder.cpp
 * @brief BM25DatasetBuilder 各类计算内核的实现
 * 
 * 通过嵌入式 Python 解释器（Python.h）调用
 * bm25_loader_xrt 和 launch_bm25 中的 Python 函数，
 * 完成数据集准备和 FPGA 设备初始化。
 *
 * PY_FUNC 属性放在此处而不是头文件中，因为
 * 不同用户可能会改用纯 C++ 实现。
 * PythonDispatchPass 会扫描这些 .cpp 文件，识别其中的 Python 调用。
 */

#include "../build_memory.h"
#include <Python.h>
#include <stdexcept>
#include <iostream>
#include <sstream>

#ifndef PY_FUNC
#define PY_FUNC(func_name) __attribute__((annotate("py_func=" func_name)))
#endif

namespace heteromm {
namespace step {

BM25DatasetBuilder::~BM25DatasetBuilder() {
    if (fpga_setup_object_) {
        Py_XDECREF(static_cast<PyObject*>(fpga_setup_object_));
        fpga_setup_object_ = nullptr;
    }
}

void BM25DatasetBuilder::ensure_python_initialized() {
    if (py_initialized_) return;

    if (!Py_IsInitialized()) {
        Py_Initialize();
    }

    // 将模块路径加入 sys.path，以便导入 bm25_loader_xrt 和 launch_bm25
    PyObject* sys_path = PySys_GetObject("path");
    if (sys_path) {
        PyObject* path_str = PyUnicode_FromString(python_module_path_.c_str());
        PyList_Append(sys_path, path_str);
        Py_DECREF(path_str);
    }

    py_initialized_ = true;
}

PY_FUNC("bm25_loader_xrt.fpga_retriever_setup")
void BM25DatasetBuilder::run_cpu_kernel(
    const data_type::TextDBData& raw_data,
    data_type::BM25IndexMemory& memory
) {
    ensure_python_initialized();

    // 导入 launch_bm25 模块（该模块重新导出 bm25_loader_xrt 中的功能）
    PyObject* py_module = PyImport_ImportModule("launch_bm25");
    if (!py_module) {
        PyErr_Print();
        throw std::runtime_error("[BM25DatasetBuilder] Failed to import launch_bm25 module");
    }

    // 调用 fpga_retriever_setup(bitstream, export_dir)
    PyObject* py_setup_func = PyObject_GetAttrString(py_module, "fpga_retriever_setup");
    if (!py_setup_func || !PyCallable_Check(py_setup_func)) {
        Py_XDECREF(py_setup_func);
        Py_DECREF(py_module);
        PyErr_Print();
        throw std::runtime_error("[BM25DatasetBuilder] fpga_retriever_setup not found or not callable");
    }

    // 构造参数：(bitstream_path, export_dir)
    PyObject* py_args = PyTuple_New(2);
    PyTuple_SetItem(py_args, 0, PyUnicode_FromString(bitstream_path_.c_str()));
    PyTuple_SetItem(py_args, 1, PyUnicode_FromString(export_dir_.c_str()));

    // 调用该函数
    PyObject* py_result = PyObject_CallObject(py_setup_func, py_args);
    Py_DECREF(py_args);
    Py_DECREF(py_setup_func);

    if (!py_result) {
        Py_DECREF(py_module);
        PyErr_Print();
        throw std::runtime_error("[BM25DatasetBuilder] fpga_retriever_setup() call failed");
    }

    if (py_result == Py_None) {
        Py_DECREF(py_result);
        Py_DECREF(py_module);
        throw std::runtime_error("[BM25DatasetBuilder] fpga_retriever_setup() returned None - FPGA setup failed");
    }

    // 保存 FPGA 初始化结果元组，供融合检索步骤使用
    // 元组包含： (kernel, L, L_doc_total, bo_query_bitmap, bo_df_buffer,
    //                      bo_inst_mem, bo_doc_mem_0..3, bo_topk_id, buf_topk)
    if (fpga_setup_object_) {
        Py_XDECREF(static_cast<PyObject*>(fpga_setup_object_));
    }
    fpga_setup_object_ = py_result;  // 接管对象引用
    Py_INCREF(static_cast<PyObject*>(fpga_setup_object_));

    // 同时根据 raw_data 构建供 C++ 侧使用的 BM25IndexMemory
    const auto& documents = raw_data.export_data();
    data_type::BM25IndexData index_data;

    for (size_t doc_idx = 0; doc_idx < documents.size(); ++doc_idx) {
        std::unordered_map<int, int> doc_freq;
        for (int token_id : documents[doc_idx]) {
            doc_freq[token_id]++;
        }
        index_data.doc_freqs.push_back(doc_freq);
        for (const auto& [token_id, freq] : doc_freq) {
            index_data.df_map[token_id]++;
        }
    }

    memory = data_type::BM25IndexMemory(index_data);

    Py_DECREF(py_result);
    Py_DECREF(py_module);

    std::clog << "[BM25DatasetBuilder] CPU kernel completed. "
              << "Loaded " << documents.size() << " documents, "
              << "FPGA setup stored." << std::endl;
}

void BM25DatasetBuilder::run_gpu_kernel(
    const data_type::TextDBData& raw_data,
    data_type::BM25IndexMemory& memory
) {
    // RAG 流水线的 build_memory 步骤不使用 GPU。
    std::clog << "[BM25DatasetBuilder] GPU kernel not implemented for build_memory. "
              << "Use CPU kernel for dataset preparation." << std::endl;
    run_cpu_kernel(raw_data, memory);
}

void BM25DatasetBuilder::run_fpga_kernel(
    const data_type::TextDBData& raw_data,
    data_type::BM25IndexMemory& memory
) {
    // build_memory 步骤不直接使用 FPGA 执行计算。
    // CPU 内核在准备数据集时一并完成 FPGA 初始化。
    std::clog << "[BM25DatasetBuilder] FPGA kernel not implemented for build_memory. "
              << "Use CPU kernel which initializes FPGA as part of setup." << std::endl;
    run_cpu_kernel(raw_data, memory);
}

void BM25DatasetBuilder::run_test_kernel(
    const data_type::TextDBData& raw_data,
    data_type::BM25IndexMemory& memory
) {
    // 测试时仅构建 BM25 索引，不进行 FPGA 初始化
    const auto& documents = raw_data.export_data();
    data_type::BM25IndexData index_data;

    for (size_t doc_idx = 0; doc_idx < documents.size(); ++doc_idx) {
        std::unordered_map<int, int> doc_freq;
        for (int token_id : documents[doc_idx]) {
            doc_freq[token_id]++;
        }
        index_data.doc_freqs.push_back(doc_freq);
        for (const auto& [token_id, freq] : doc_freq) {
            index_data.df_map[token_id]++;
        }
    }

    memory = data_type::BM25IndexMemory(index_data);
}

}  // namespace step
}  // namespace heteromm
