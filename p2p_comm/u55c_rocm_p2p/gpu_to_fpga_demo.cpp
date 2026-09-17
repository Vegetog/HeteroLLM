/**
 * GPU → FPGA P2P 传输示例
 *
 * 演示反向 P2P 传输：GPU 将数据写入位于 FPGA 上、
 * 可由两个设备共同访问的缓冲区。
 *
 * 流程：
 *   1. FPGA 在 HBM 中分配 P2P 缓冲区
 *   2. 通过 xrt::bo::map() 将缓冲区映射到主机地址空间
 *   3. 通过 hipHostRegister() 向 GPU 注册映射后的指针
 *   4. GPU 内核通过 P2P 设备指针写入数据
 *   5. 读取 FPGA 缓冲区中的数据（通过主机映射指针或 FPGA 内核）
 *
 * 编译：
 *   hipcc -c gpu_write_kernels.hip -o gpu_write_kernels.o
 *   hipcc gpu_to_fpga_demo.cpp gpu_write_kernels.o \
 *         -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -lxrt_coreutil \
 *         -o gpu_to_fpga_demo
 */

#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cassert>
#include <numeric>
#include <random>
#include <set>

// XRT 头文件
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>

// HIP 头文件
#include <hip/hip_runtime.h>

#define HIP_CHECK(cmd) \
    do { \
        hipError_t error = (cmd); \
        if (error != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ============================================================================
// GPU 内核声明（定义位于 gpu_write_kernels.hip）
// ============================================================================

extern "C" __global__ void write_indices_to_fpga(
    const uint32_t* __restrict__ gpu_indices,
    uint32_t* __restrict__ fpga_buffer,
    uint32_t count);

extern "C" __global__ void write_floats_to_fpga(
    const float* __restrict__ gpu_data,
    float* __restrict__ fpga_buffer,
    uint32_t count);

extern "C" __global__ void generate_pattern_to_fpga(
    uint32_t* __restrict__ fpga_buffer,
    uint32_t count,
    uint32_t start,
    uint32_t stride);

extern "C" __global__ void scatter_write_to_fpga(
    const uint32_t* __restrict__ gpu_data,
    const uint32_t* __restrict__ scatter_indices,
    uint32_t* __restrict__ fpga_buffer,
    uint32_t count,
    uint32_t buffer_size);

// ============================================================================
// 辅助功能：在 FPGA 上分配 P2P 缓冲区
// ============================================================================

struct P2PAllocation {
    xrt::bo    bo;
    void*      host_ptr    = nullptr;
    void*      device_ptr  = nullptr;   // HIP 设备指针
    size_t     size_bytes  = 0;
    bool       is_p2p      = false;     // true 表示 P2P 缓冲区对象（BO），false 表示回退到普通缓冲区
    bool       gpu_registered = false;
};

/**
 * 分配带有 P2P 标志的 xrt::bo，将其映射并注册到 HIP，
 * 使 GPU 内核能够通过 device_ptr 访问该缓冲区。
 */
P2PAllocation alloc_p2p_buffer(xrt::device& fpga_dev, size_t size_bytes) {
    P2PAllocation alloc;
    alloc.size_bytes = size_bytes;

    // --- 1. 分配 FPGA 缓冲区（先尝试 P2P，失败后回退到普通缓冲区） ---
    bool allocated = false;
    constexpr unsigned kMaxGroups = 32;

    for (unsigned g = 0; g < kMaxGroups && !allocated; ++g) {
        try {
            alloc.bo    = xrt::bo(fpga_dev, size_bytes, xrt::bo::flags::p2p, g);
            alloc.is_p2p = true;
            allocated   = true;
        } catch (...) {}
    }
    if (!allocated) {
        for (unsigned g = 0; g < kMaxGroups && !allocated; ++g) {
            try {
                alloc.bo    = xrt::bo(fpga_dev, size_bytes, xrt::bo::flags::normal, g);
                alloc.is_p2p = false;
                allocated   = true;
            } catch (...) {}
        }
    }
    if (!allocated) {
        std::cerr << "Fatal: could not allocate FPGA buffer in any memory group" << std::endl;
        exit(EXIT_FAILURE);
    }

    // --- 2. 映射到主机虚拟地址 ---
    alloc.host_ptr = alloc.bo.map();
    if (!alloc.host_ptr) {
        std::cerr << "Fatal: xrt::bo::map() returned nullptr" << std::endl;
        exit(EXIT_FAILURE);
    }

    // --- 3. 向 HIP 注册 ---
    hipError_t err = hipHostRegister(
        alloc.host_ptr, size_bytes,
        hipHostRegisterMapped | hipHostRegisterIoMemory);

    if (err != hipSuccess) {
        // 如果 IOMMU 阻止访问，IoMemory 注册可能失败，此时采用回退方式。
        // 此时不再是真正的 P2P，需要执行同步操作。
        std::cerr << "  Warning: hipHostRegisterIoMemory failed ("
                  << hipGetErrorString(err) << "), retrying with Mapped only" << std::endl;
        alloc.is_p2p = false;
        err = hipHostRegister(alloc.host_ptr, size_bytes, hipHostRegisterMapped);
        if (err != hipSuccess) {
            std::cerr << "Fatal: hipHostRegister failed: "
                      << hipGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    HIP_CHECK(hipHostGetDevicePointer(&alloc.device_ptr, alloc.host_ptr, 0));
    alloc.gpu_registered = true;

    return alloc;
}

void free_p2p_buffer(P2PAllocation& alloc) {
    if (alloc.gpu_registered) {
        HIP_CHECK(hipHostUnregister(alloc.host_ptr));
        alloc.gpu_registered = false;
    }
}

// ============================================================================
// 测试 1：简单顺序写入
// ============================================================================

bool test_sequential_write(xrt::device& fpga_dev) {
    std::cout << "\n--- Test 1: Sequential GPU→FPGA write ---" << std::endl;

    constexpr uint32_t COUNT = 1024;
    constexpr size_t   BYTES = COUNT * sizeof(uint32_t);

    // 分配 P2P 缓冲区
    auto alloc = alloc_p2p_buffer(fpga_dev, BYTES);
    std::cout << "  P2P buffer: " << BYTES << " B, is_p2p="
              << alloc.is_p2p << std::endl;

    // 从主机端将缓冲区清零，以便随后验证写入结果
    std::memset(alloc.host_ptr, 0, BYTES);
    if (!alloc.is_p2p) {
        alloc.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    // 在 GPU 全局内存中准备源数据
    uint32_t* d_src;
    HIP_CHECK(hipMalloc(&d_src, BYTES));

    std::vector<uint32_t> src_host(COUNT);
    std::iota(src_host.begin(), src_host.end(), 100);   // 100, 101, 102, ...
    HIP_CHECK(hipMemcpy(d_src, src_host.data(), BYTES, hipMemcpyHostToDevice));

    // 启动 GPU 内核：从 GPU 内存写入 FPGA P2P 缓冲区
    dim3 threads(256);
    dim3 blocks((COUNT + 255) / 256);

    auto t0 = std::chrono::high_resolution_clock::now();

    hipLaunchKernelGGL(write_indices_to_fpga, blocks, threads, 0, 0,
                       d_src,
                       static_cast<uint32_t*>(alloc.device_ptr),
                       COUNT);
    HIP_CHECK(hipDeviceSynchronize());

    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    std::cout << "  GPU kernel + sync: " << us << " us" << std::endl;

    // --- 验证前执行同步 ---
    // 非 P2P 模式：先将主机端影子副本同步到 FPGA，再读回以进行验证。
    // P2P 模式：从设备端同步，以确保在 BAR 不与 CPU 保持缓存一致性
    // 的平台上，主机映射也能看到 GPU 写入的数据。
    if (!alloc.is_p2p) {
        std::cout << "  (non-P2P fallback) calling sync_to_device..." << std::endl;
        alloc.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    alloc.bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    auto* result = static_cast<uint32_t*>(alloc.host_ptr);
    int errors = 0;
    for (uint32_t i = 0; i < COUNT; i++) {
        if (result[i] != src_host[i]) {
            if (errors < 5) {
                std::cerr << "    MISMATCH [" << i << "]: got " << result[i]
                          << ", expected " << src_host[i] << std::endl;
            }
            errors++;
        }
    }

    HIP_CHECK(hipFree(d_src));
    free_p2p_buffer(alloc);

    if (errors == 0) {
        std::cout << "  PASSED (" << COUNT << " elements verified)" << std::endl;
    } else {
        std::cerr << "  FAILED: " << errors << " mismatches" << std::endl;
    }
    return errors == 0;
}

// ============================================================================
// 测试 2：GPU 直接在 FPGA 缓冲区中生成规律数据（无需中转缓冲）
// ============================================================================

bool test_pattern_generation(xrt::device& fpga_dev) {
    std::cout << "\n--- Test 2: GPU pattern generation → FPGA ---" << std::endl;

    constexpr uint32_t COUNT  = 2048;
    constexpr size_t   BYTES  = COUNT * sizeof(uint32_t);
    constexpr uint32_t START  = 42;
    constexpr uint32_t STRIDE = 3;

    auto alloc = alloc_p2p_buffer(fpga_dev, BYTES);
    std::memset(alloc.host_ptr, 0, BYTES);
    if (!alloc.is_p2p) alloc.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    dim3 threads(256);
    dim3 blocks((COUNT + 255) / 256);

    hipLaunchKernelGGL(generate_pattern_to_fpga, blocks, threads, 0, 0,
                       static_cast<uint32_t*>(alloc.device_ptr),
                       COUNT, START, STRIDE);
    HIP_CHECK(hipDeviceSynchronize());

    if (!alloc.is_p2p) {
        alloc.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    alloc.bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    auto* result = static_cast<uint32_t*>(alloc.host_ptr);
    int errors = 0;
    for (uint32_t i = 0; i < COUNT; i++) {
        uint32_t expected = START + i * STRIDE;
        if (result[i] != expected) {
            if (errors < 5) {
                std::cerr << "    MISMATCH [" << i << "]: got " << result[i]
                          << ", expected " << expected << std::endl;
            }
            errors++;
        }
    }

    free_p2p_buffer(alloc);

    if (errors == 0) {
        std::cout << "  PASSED (" << COUNT << " elements)" << std::endl;
    } else {
        std::cerr << "  FAILED: " << errors << " mismatches" << std::endl;
    }
    return errors == 0;
}

// ============================================================================
// 测试 3：向 FPGA 散写
// ============================================================================

bool test_scatter_write(xrt::device& fpga_dev) {
    std::cout << "\n--- Test 3: GPU scatter-write → FPGA ---" << std::endl;

    constexpr uint32_t BUFFER_SIZE = 4096;
    constexpr uint32_t WRITE_COUNT = 512;
    constexpr size_t   BYTES = BUFFER_SIZE * sizeof(uint32_t);

    auto alloc = alloc_p2p_buffer(fpga_dev, BYTES);
    // 填入哨兵值，以便验证是否只有写入位置发生了变化
    std::memset(alloc.host_ptr, 0xFF, BYTES);
    if (!alloc.is_p2p) alloc.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // 在主机端生成互不重复的随机散写索引
    std::mt19937 rng(7);
    std::set<uint32_t> idx_set;
    std::uniform_int_distribution<uint32_t> dist(0, BUFFER_SIZE - 1);
    while (idx_set.size() < WRITE_COUNT) idx_set.insert(dist(rng));
    std::vector<uint32_t> indices(idx_set.begin(), idx_set.end());

    // 待写入的数据：值 = 索引 * 10
    std::vector<uint32_t> data(WRITE_COUNT);
    for (uint32_t i = 0; i < WRITE_COUNT; i++) data[i] = indices[i] * 10;

    // 上传到 GPU
    uint32_t *d_data, *d_indices;
    HIP_CHECK(hipMalloc(&d_data,    WRITE_COUNT * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&d_indices, WRITE_COUNT * sizeof(uint32_t)));
    HIP_CHECK(hipMemcpy(d_data,    data.data(),    WRITE_COUNT * sizeof(uint32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_indices, indices.data(), WRITE_COUNT * sizeof(uint32_t), hipMemcpyHostToDevice));

    dim3 threads(256);
    dim3 blocks((WRITE_COUNT + 255) / 256);

    hipLaunchKernelGGL(scatter_write_to_fpga, blocks, threads, 0, 0,
                       d_data, d_indices,
                       static_cast<uint32_t*>(alloc.device_ptr),
                       WRITE_COUNT, BUFFER_SIZE);
    HIP_CHECK(hipDeviceSynchronize());

    if (!alloc.is_p2p) {
        alloc.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    alloc.bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    auto* result = static_cast<uint32_t*>(alloc.host_ptr);
    int errors = 0;
    for (uint32_t i = 0; i < WRITE_COUNT; i++) {
        uint32_t pos = indices[i];
        if (result[pos] != data[i]) {
            if (errors < 5) {
                std::cerr << "    MISMATCH at pos " << pos << ": got " << result[pos]
                          << ", expected " << data[i] << std::endl;
            }
            errors++;
        }
    }

    HIP_CHECK(hipFree(d_data));
    HIP_CHECK(hipFree(d_indices));
    free_p2p_buffer(alloc);

    if (errors == 0) {
        std::cout << "  PASSED (" << WRITE_COUNT << " scattered writes verified)" << std::endl;
    } else {
        std::cerr << "  FAILED: " << errors << " mismatches" << std::endl;
    }
    return errors == 0;
}

// ============================================================================
// 测试 4：带宽基准测试（GPU → FPGA）
// ============================================================================

void test_bandwidth(xrt::device& fpga_dev) {
    std::cout << "\n--- Test 4: GPU→FPGA write bandwidth ---" << std::endl;

    // 测试多种数据大小
    const std::vector<size_t> sizes = {
        4 * 1024,           //   4 KB
        64 * 1024,          //  64 KB
        1024 * 1024,        //   1 MB
        16 * 1024 * 1024,   //  16 MB
    };

    for (size_t bytes : sizes) {
        uint32_t count = bytes / sizeof(uint32_t);
        auto alloc = alloc_p2p_buffer(fpga_dev, bytes);

        // 准备 GPU 端源数据
        uint32_t* d_src;
        HIP_CHECK(hipMalloc(&d_src, bytes));
        HIP_CHECK(hipMemset(d_src, 0xAB, bytes));

        dim3 threads(256);
        dim3 blocks((count + 255) / 256);

        constexpr int WARMUP = 5;
        constexpr int ITERS  = 50;

        // 预热
        for (int i = 0; i < WARMUP; i++) {
            hipLaunchKernelGGL(write_indices_to_fpga, blocks, threads, 0, 0,
                               d_src, static_cast<uint32_t*>(alloc.device_ptr), count);
        }
        HIP_CHECK(hipDeviceSynchronize());

        if (!alloc.is_p2p) {
            std::cout << "  WARNING: non-P2P fallback — bandwidth includes XRT sync cost" << std::endl;
        }

        // 执行计时迭代
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) {
            hipLaunchKernelGGL(write_indices_to_fpga, blocks, threads, 0, 0,
                               d_src, static_cast<uint32_t*>(alloc.device_ptr), count);
            HIP_CHECK(hipDeviceSynchronize());
            if (!alloc.is_p2p) {
                alloc.bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        double sec = std::chrono::duration<double>(t1 - t0).count();
        double gbps = (double(bytes) * ITERS) / sec / 1e9;
        double avg_us = (sec / ITERS) * 1e6;

        char label[32];
        if (bytes >= 1024 * 1024)
            snprintf(label, sizeof(label), "%4zu MB", bytes / (1024 * 1024));
        else
            snprintf(label, sizeof(label), "%4zu KB", bytes / 1024);

        std::cout << "  " << label << ":  " << gbps << " GB/s  (avg "
                  << avg_us << " us/iter)" << std::endl;

        HIP_CHECK(hipFree(d_src));
        free_p2p_buffer(alloc);
    }
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "=== GPU → FPGA P2P Transfer Demo ===" << std::endl;

    // --- 查找 FPGA ---
    std::cout << "\n[Init] Scanning for FPGA devices..." << std::endl;
    xrt::device fpga_dev;
    bool found = false;
    for (unsigned i = 0; i < 16 && !found; i++) {
        try {
            xrt::device dev(i);
            std::string name = dev.get_info<xrt::info::device::name>();
            std::string bdf  = dev.get_info<xrt::info::device::bdf>();
            std::cout << "  Found: index=" << i << "  BDF=" << bdf
                      << "  name=" << name << std::endl;
            fpga_dev = std::move(dev);
            found = true;
        } catch (...) {}
    }
    if (!found) {
        std::cerr << "No FPGA device found." << std::endl;
        return EXIT_FAILURE;
    }

    // --- 查找 GPU ---
    std::cout << "\n[Init] Scanning for GPU devices..." << std::endl;
    int gpu_count = 0;
    HIP_CHECK(hipGetDeviceCount(&gpu_count));
    if (gpu_count == 0) {
        std::cerr << "No GPU found." << std::endl;
        return EXIT_FAILURE;
    }
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    std::cout << "  Using GPU 0: " << props.name << std::endl;

    // --- 运行测试 ---
    int pass = 0, fail = 0;

    test_sequential_write(fpga_dev)  ? pass++ : fail++;
    test_pattern_generation(fpga_dev) ? pass++ : fail++;
    test_scatter_write(fpga_dev)      ? pass++ : fail++;
    test_bandwidth(fpga_dev);

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "  Passed: " << pass << std::endl;
    std::cout << "  Failed: " << fail << std::endl;

    return fail > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}