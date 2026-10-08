// 读 ONNX 文件里的权重和节点（只读，不执行）：声纹模型 campplus.onnx 直接从 CosyVoice 的模型仓库下，不转格式。
// 自己解 protobuf 的几个字段（ModelProto.graph → GraphProto.node / initializer；NodeProto 的输入、输出、算子名、
// 整数属性；TensorProto 的形状、float 数据、名字；Constant 节点的 value），不依赖 protobuf / onnx 库。
// 只认 float32 的权重（raw_data 或 float_data）；外部数据（external_data）不支持。
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace voice {

struct OnnxTensor {
    std::vector<int64_t> dims;
    std::vector<float> data;
};

struct OnnxNode {
    std::string op;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    /// 整数 / 整数列表属性（kernel_shape、strides、pads、dilations、group…）
    std::map<std::string, std::vector<int64_t>> ints;
    /// 浮点属性（epsilon…）
    std::map<std::string, float> floats;
};

class OnnxWeights {
public:
    bool load(const std::string& path, std::string* error);

    /// 按图里的先后
    const std::vector<OnnxNode>& nodes() const { return m_nodes; }
    /// 初始值（权重）和 Constant 节点的值，按名字
    const OnnxTensor* tensor(const std::string& name) const;
    size_t parameterCount() const;

private:
    std::vector<OnnxNode> m_nodes;
    std::map<std::string, OnnxTensor> m_tensors;
};

} // namespace voice
