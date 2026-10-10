// friday_runtime_tests 入口：按名字子串挑用例，一个个跑（见 rt_test.h）
#include "rt_test.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

int main(int argc, char** argv)
{
#if defined(_WIN32)
    // 输出里有中文：控制台按 UTF-8 显示
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<std::string> filters;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--list") == 0) {
            for (const rt_test::Case& test : rt_test::registry())
                std::printf("%s\n", test.name);
            return 0;
        }
        filters.emplace_back(argv[i]);
    }

    int passed = 0;
    int failed = 0;
    int skipped = 0;
    for (const rt_test::Case& test : rt_test::registry()) {
        if (!filters.empty()) {
            bool match = false;
            for (const std::string& filter : filters)
                match = match || std::string(test.name).find(filter) != std::string::npos;
            if (!match)
                continue;
        }
        const auto started = std::chrono::steady_clock::now();
        try {
            test.run();
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
            std::printf("PASS   %s（%.0f ms）\n", test.name, ms);
            ++passed;
        } catch (const rt_test::Skip& skip) {
            std::printf("SKIP   %s：%s\n", test.name, skip.reason.c_str());
            ++skipped;
        } catch (const rt_test::Failure& failure) {
            std::printf("FAIL!  %s：%s\n", test.name, failure.message.c_str());
            ++failed;
        } catch (const std::exception& error) {
            std::printf("FAIL!  %s：未捕获的异常：%s\n", test.name, error.what());
            ++failed;
        } catch (...) {
            std::printf("FAIL!  %s：未捕获的未知异常\n", test.name);
            ++failed;
        }
    }
    std::printf("合计：%d 通过，%d 失败，%d 跳过\n", passed, failed, skipped);
    return failed == 0 ? 0 : 1;
}
