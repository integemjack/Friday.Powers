#include "OnnxWeights.h"

#include <cstring>
#include <fstream>
#include <iterator>

namespace voice {

namespace {

// protobuf 线格式：tag = (字段号 << 3) | 类型；类型 0 varint、1 定长 8 字节、2 带长度、5 定长 4 字节
struct Reader {
    const uint8_t* p;
    const uint8_t* end;
    bool ok = true;

    bool done() const { return !ok || p >= end; }

    uint64_t varint()
    {
        uint64_t value = 0;
        for (int shift = 0; shift < 64; shift += 7) {
            if (p >= end) {
                ok = false;
                return 0;
            }
            const uint8_t byte = *p++;
            value |= uint64_t(byte & 0x7f) << shift;
            if (!(byte & 0x80))
                return value;
        }
        ok = false;
        return 0;
    }

    /// 带长度的一段（子消息、字符串、packed 数组）
    Reader sub()
    {
        const uint64_t length = varint();
        if (!ok || length > uint64_t(end - p)) {
            ok = false;
            return { end, end, false };
        }
        Reader r { p, p + length };
        p += length;
        return r;
    }

    std::string string()
    {
        Reader r = sub();
        return std::string(reinterpret_cast<const char*>(r.p), size_t(r.end - r.p));
    }

    uint32_t fixed32()
    {
        if (end - p < 4) {
            ok = false;
            return 0;
        }
        uint32_t v;
        std::memcpy(&v, p, 4);
        p += 4;
        return v;
    }

    void skip(int wire)
    {
        switch (wire) {
        case 0: varint(); break;
        case 1: p = (end - p < 8) ? (ok = false, end) : p + 8; break;
        case 2: sub(); break;
        case 5: fixed32(); break;
        default: ok = false;
        }
    }
};

float asFloat(uint32_t bits)
{
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

/// TensorProto：dims=1、data_type=2、float_data=4、name=8、raw_data=9
bool parseTensor(Reader r, std::string* name, OnnxTensor* out, std::string* error)
{
    int dataType = 0;
    std::string raw;
    bool hasRaw = false;
    while (!r.done()) {
        const uint64_t tag = r.varint();
        const int field = int(tag >> 3), wire = int(tag & 7);
        if (field == 1 && wire == 0) {
            out->dims.push_back(int64_t(r.varint()));
        } else if (field == 1 && wire == 2) {
            Reader packed = r.sub();
            while (!packed.done())
                out->dims.push_back(int64_t(packed.varint()));
        } else if (field == 2 && wire == 0) {
            dataType = int(r.varint());
        } else if (field == 4 && wire == 2) {
            Reader packed = r.sub();
            while (!packed.done())
                out->data.push_back(asFloat(packed.fixed32()));
        } else if (field == 4 && wire == 5) {
            out->data.push_back(asFloat(r.fixed32()));
        } else if (field == 8 && wire == 2) {
            *name = r.string();
        } else if (field == 9 && wire == 2) {
            raw = r.string();
            hasRaw = true;
        } else if (field == 13 && wire == 2) {
            *error = "不支持外部数据（external_data）的张量：" + *name;
            return false;
        } else {
            r.skip(wire);
        }
    }
    if (!r.ok) {
        *error = "张量解析失败：" + *name;
        return false;
    }
    // 只要 float32（1）；别的类型（形状常量用的 int64 等）留空、不报错
    if (dataType != 1) {
        out->data.clear();
        return true;
    }
    if (hasRaw) {
        out->data.resize(raw.size() / 4);
        std::memcpy(out->data.data(), raw.data(), out->data.size() * 4);
    }
    return true;
}

/// AttributeProto：name=1、f=2、i=3、t=5、floats=7、ints=8
bool parseAttribute(Reader r, OnnxNode* node, OnnxTensor* tensor, bool* hasTensor, std::string* error)
{
    std::string name;
    std::vector<int64_t> ints;
    float f = 0;
    bool hasFloat = false;
    while (!r.done()) {
        const uint64_t tag = r.varint();
        const int field = int(tag >> 3), wire = int(tag & 7);
        if (field == 1 && wire == 2) {
            name = r.string();
        } else if (field == 2 && wire == 5) {
            f = asFloat(r.fixed32());
            hasFloat = true;
        } else if (field == 3 && wire == 0) {
            ints.push_back(int64_t(r.varint()));
        } else if (field == 5 && wire == 2) {
            std::string ignored;
            if (!parseTensor(r.sub(), &ignored, tensor, error))
                return false;
            *hasTensor = true;
        } else if (field == 8 && wire == 0) {
            ints.push_back(int64_t(r.varint()));
        } else if (field == 8 && wire == 2) {
            Reader packed = r.sub();
            while (!packed.done())
                ints.push_back(int64_t(packed.varint()));
        } else {
            r.skip(wire);
        }
    }
    if (!r.ok) {
        *error = "属性解析失败";
        return false;
    }
    if (hasFloat)
        node->floats[name] = f;
    if (!ints.empty())
        node->ints[name] = ints;
    return true;
}

} // namespace

bool OnnxWeights::load(const std::string& path, std::string* error)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        *error = "打不开 " + path;
        return false;
    }
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    Reader model { bytes.data(), bytes.data() + bytes.size() };
    bool sawGraph = false;
    while (!model.done()) {
        const uint64_t tag = model.varint();
        const int field = int(tag >> 3), wire = int(tag & 7);
        if (field != 7 || wire != 2) {   // ModelProto.graph
            model.skip(wire);
            continue;
        }
        sawGraph = true;
        Reader graph = model.sub();
        while (!graph.done()) {
            const uint64_t gtag = graph.varint();
            const int gfield = int(gtag >> 3), gwire = int(gtag & 7);
            if (gfield == 1 && gwire == 2) {   // NodeProto：input=1、output=2、op_type=4、attribute=5
                Reader r = graph.sub();
                OnnxNode node;
                OnnxTensor constant;
                bool hasConstant = false;
                while (!r.done()) {
                    const uint64_t ntag = r.varint();
                    const int nfield = int(ntag >> 3), nwire = int(ntag & 7);
                    if (nfield == 1 && nwire == 2)
                        node.inputs.push_back(r.string());
                    else if (nfield == 2 && nwire == 2)
                        node.outputs.push_back(r.string());
                    else if (nfield == 4 && nwire == 2)
                        node.op = r.string();
                    else if (nfield == 5 && nwire == 2) {
                        if (!parseAttribute(r.sub(), &node, &constant, &hasConstant, error))
                            return false;
                    } else
                        r.skip(nwire);
                }
                if (!r.ok) {
                    *error = "节点解析失败";
                    return false;
                }
                if (node.op == "Constant" && hasConstant && !node.outputs.empty())
                    m_tensors[node.outputs.front()] = std::move(constant);
                m_nodes.push_back(std::move(node));
            } else if (gfield == 5 && gwire == 2) {   // initializer
                std::string name;
                OnnxTensor tensor;
                if (!parseTensor(graph.sub(), &name, &tensor, error))
                    return false;
                m_tensors[name] = std::move(tensor);
            } else {
                graph.skip(gwire);
            }
        }
        if (!graph.ok) {
            *error = "计算图解析失败";
            return false;
        }
    }
    if (!model.ok || !sawGraph) {
        *error = path + " 不是 ONNX 模型";
        return false;
    }
    return true;
}

const OnnxTensor* OnnxWeights::tensor(const std::string& name) const
{
    const auto it = m_tensors.find(name);
    return it == m_tensors.end() || it->second.data.empty() ? nullptr : &it->second;
}

size_t OnnxWeights::parameterCount() const
{
    size_t n = 0;
    for (const auto& [name, tensor] : m_tensors)
        n += tensor.data.size();
    return n;
}

} // namespace voice
