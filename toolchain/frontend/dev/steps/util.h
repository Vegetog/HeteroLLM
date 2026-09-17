#ifndef HETEROMM_DEV_STEPS_UTIL_H_
#define HETEROMM_DEV_STEPS_UTIL_H_

#include <string>

namespace heteromm {
namespace step {

/**
 * @brief 表示内核执行目标的枚举
 */
enum class KernelType {
    CPU,     // CPU/GPU 内核（常规计算）
    GPU,     // 明确指定的 GPU 内核
    FPGA,    // FPGA 内核
};

/**
 * @brief 将 KernelType 转换为字符串，用于日志输出
 */
inline std::string kernel_type_to_string(KernelType type) {
    switch (type) {
        case KernelType::CPU: return "CPU";
        case KernelType::GPU: return "GPU";
        case KernelType::FPGA: return "FPGA";
        default: return "UNKNOWN";
    }
}

KernelType string_to_kernel_type(const std::string& device) {
    if (device == "cpu") {
        return KernelType::CPU;
    } else if (device == "gpu") {
        return KernelType::GPU;
    } else if (device == "fpga") {
        return KernelType::FPGA;
    }
    return KernelType::CPU;  // 默认使用 CPU
}

//[TODO]：添加通信处理器

}  // namespace step
}  // namespace heteromm

#endif  // HETEROMM_DEV_STEPS_UTIL_H_