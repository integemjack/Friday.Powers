// friday_runtime_tests 的小测试框架（不依赖 Qt：测试程序与 friday-llama 一样 /MD + _ITERATOR_DEBUG_LEVEL=0，
// 才能直接用 common 的 C++ 接口；QtTest 的 Debug 构建是 /MDd，混不到一起）。
//
//   RT_TEST(sequence_cache_trims_part_models) { …; RT_CHECK(x == 1); RT_CHECK_EQ(a, b); if (!model) RT_SKIP("没有模型"); }
//
// 运行：friday_runtime_tests [--list] [名字里的子串 …]（不给就全跑）。每个 WP 把自己的用例写在 tests/test_<领域>.cpp 里。
// 真实模型 / 显卡用例：先拿 GPU 锁（docs/LOCAL_INFERENCE.md §H.6），模型路径从环境变量读（rt_test::modelPath），没设就 RT_SKIP。
#pragma once

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <string>
#include <vector>

namespace rt_test {

struct Case {
    const char* name;
    void (*run)();
};

inline std::vector<Case>& registry()
{
    static std::vector<Case> cases;
    return cases;
}

struct Registrar {
    Registrar(const char* name, void (*run)()) { registry().push_back({ name, run }); }
};

/// 断言失败（RT_CHECK 抛出，main 接住记一条失败，接着跑下一个用例）
struct Failure : std::exception {
    std::string message;
    explicit Failure(std::string text) : message(std::move(text)) {}
    const char* what() const noexcept override { return message.c_str(); }
};

/// 跳过（缺模型、缺显卡……）
struct Skip : std::exception {
    std::string reason;
    explicit Skip(std::string text) : reason(std::move(text)) {}
    const char* what() const noexcept override { return reason.c_str(); }
};

[[noreturn]] inline void fail(const char* file, int line, const std::string& what)
{
    std::ostringstream out;
    out << file << ":" << line << ": " << what;
    throw Failure(out.str());
}

template <class A, class B>
void checkEqual(const A& a, const B& b, const char* expressionA, const char* expressionB, const char* file, int line)
{
    if (a == b)
        return;
    std::ostringstream out;
    out << expressionA << " == " << expressionB << "（实际：" << a << " 与 " << b << "）";
    fail(file, line, out.str());
}

/// 环境变量（没设为空）
inline std::string env(const char* name)
{
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string();
}

/// 真实模型的路径（环境变量，见 §H.6）：FRIDAY_RT_MODEL（Qwen3.5-9B Q4_K_M，混合模型）、FRIDAY_RT_MMPROJ、
/// FRIDAY_RT_SMALL_MODEL（Qwen3-1.7B Q8_0，PART 型）。没设就跳过：RT_REQUIRE_ENV("FRIDAY_RT_MODEL")
inline std::string modelPath(const char* variable)
{
    return env(variable);
}

} // namespace rt_test

#define RT_TEST(name)                                                                \
    static void rt_test_##name();                                                    \
    static const rt_test::Registrar rt_test_registrar_##name(#name, &rt_test_##name); \
    static void rt_test_##name()

#define RT_CHECK(condition)                                     \
    do {                                                        \
        if (!(condition))                                       \
            rt_test::fail(__FILE__, __LINE__, #condition);      \
    } while (0)

#define RT_CHECK_EQ(a, b) rt_test::checkEqual((a), (b), #a, #b, __FILE__, __LINE__)

#define RT_SKIP(reason) throw rt_test::Skip(reason)

/// 环境变量没设就跳过这个用例，设了返回它的值
#define RT_REQUIRE_ENV(variable)                                                          \
    [&]() {                                                                               \
        const std::string value_ = rt_test::env(variable);                                \
        if (value_.empty())                                                               \
            throw rt_test::Skip(std::string("没有设置 ") + (variable));                   \
        return value_;                                                                    \
    }()
