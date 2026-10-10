# friday-llm：本地大模型（Friday 的 Power）
#
# 一个可执行文件：llama.cpp（libllama + common + mtmd，ggml 的 CUDA / Vulkan / Metal / CPU 静态链进来）+ 自己的调度器
# （连续批处理、多序列并发、基数树前缀共享）+ OpenAI 兼容的 HTTP 服务。模型不打进文件：Friday 的模型库下载 GGUF，
# 请求里告诉它用哪个文件（docs/llm-protocol.md）。
#
#   cmake -B build -G Ninja -DLLM_CUDA=ON          # 或 -DLLM_VULKAN=ON / -DLLM_METAL=ON
#   cmake --build build                             # 产物 build/bin/friday-llm
cmake_minimum_required(VERSION 3.24)
project(friday-llm VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
if(MSVC)
    # 单文件：静态 C/C++ 运行库，不依赖 VC 运行库 DLL
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>")
    add_compile_definitions(NOMINMAX _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS WIN32_LEAN_AND_MEAN)
endif()

option(LLM_CUDA   "CUDA 后端（NVIDIA）" OFF)
option(LLM_VULKAN "Vulkan 后端（NVIDIA / AMD / Intel / Adreno）" OFF)
option(LLM_METAL  "Metal 后端（Apple）" OFF)
option(LLM_TESTS  "单测（friday-llm-tests）" ON)

# ── llama.cpp：静态、所有后端编进同一个文件；只要 libllama、common、mtmd，不编它的工具、服务器、测试 ──
set(BUILD_SHARED_LIBS      OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL        OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE            OFF CACHE BOOL "")          # 默认可移植；本机调试可 -DGGML_NATIVE=ON
set(GGML_OPENMP            OFF CACHE BOOL "" FORCE)    # 用 ggml 自己的线程池：OpenMP 运行库是动态的
set(GGML_CUDA   ${LLM_CUDA}   CACHE BOOL "" FORCE)
set(GGML_VULKAN ${LLM_VULKAN} CACHE BOOL "" FORCE)
set(GGML_METAL  ${LLM_METAL}  CACHE BOOL "" FORCE)
if(LLM_METAL)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
endif()
if(LLM_CUDA)
    set(GGML_CUDA_NO_VMM ON CACHE BOOL "" FORCE)        # 不直接依赖 libcuda 的 VMM 接口
    if(UNIX AND NOT APPLE)
        # Linux 上 cudart / cuBLAS 都静态链进来，只剩驱动的 libcuda.so
        set(GGML_STATIC ON CACHE BOOL "" FORCE)
    endif()
endif()
set(LLAMA_BUILD_COMMON     ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD       ON  CACHE BOOL "" FORCE)    # tools/mtmd 单独编（看图）
set(LLAMA_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP        OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI         OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL          OFF CACHE BOOL "" FORCE)    # 不从网上下模型（模型由 Friday 下）
set(LLAMA_SUBPROCESS       OFF CACHE BOOL "" FORCE)    # 不要路由模式；mtmd 的视频（要 ffmpeg 子进程）跟着关
set(LLAMA_ALL_WARNINGS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_IS_DEV     OFF CACHE BOOL "" FORCE)

include(FetchContent)
# llama.cpp 的版本（换版本时同时改 src/engine 里照着 server 写的那几处，见 docs/llm-design.md §2）
set(LLM_LLAMA_TAG b11541)
FetchContent_Declare(llama
    URL https://github.com/ggml-org/llama.cpp/archive/refs/tags/${LLM_LLAMA_TAG}.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(civetweb
    URL https://github.com/civetweb/civetweb/archive/refs/tags/v1.16.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(llama)
FetchContent_GetProperties(civetweb)
if(NOT civetweb_POPULATED)
    if(POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif()
    FetchContent_Populate(civetweb)       # 只要 civetweb.c，不走它自己的 CMake（会编测试、SSL）
endif()

# ── civetweb：HTTP 服务器，C 单文件 ──
add_library(civetweb STATIC ${civetweb_SOURCE_DIR}/src/civetweb.c)
target_include_directories(civetweb PUBLIC ${civetweb_SOURCE_DIR}/include PRIVATE ${civetweb_SOURCE_DIR}/src)
target_compile_definitions(civetweb PUBLIC NO_SSL NO_CGI NO_FILES NO_CACHING MG_EXPERIMENTAL_INTERFACES=0)
if(WIN32)
    target_link_libraries(civetweb PUBLIC ws2_32)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(civetweb PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
endif()

# ── 推理内核（不含 HTTP，单测直接链它）──
file(GLOB LLM_ENGINE_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/engine/*.cpp)
add_library(llm_engine STATIC ${LLM_ENGINE_SOURCES})
target_include_directories(llm_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/engine ${llama_SOURCE_DIR}/vendor)
target_link_libraries(llm_engine PUBLIC llama-common mtmd llama)
target_compile_definitions(llm_engine PUBLIC LLM_LLAMA_TAG="${LLM_LLAMA_TAG}")
if(MSVC)
    target_compile_options(llm_engine PRIVATE /bigobj)
endif()

# ── friday-llm ──
add_executable(friday-llm
    src/main.cpp
    src/Server.cpp
    src/ModelHost.cpp)
target_include_directories(friday-llm PRIVATE src)
target_compile_definitions(friday-llm PRIVATE
    LLM_VERSION="${PROJECT_VERSION}"
    $<$<BOOL:${LLM_CUDA}>:LLM_HAS_CUDA>
    $<$<BOOL:${LLM_VULKAN}>:LLM_HAS_VULKAN>
    $<$<BOOL:${LLM_METAL}>:LLM_HAS_METAL>)
target_link_libraries(friday-llm PRIVATE llm_engine civetweb)
if(WIN32)
    target_link_libraries(friday-llm PRIVATE shell32)
else()
    target_link_libraries(friday-llm PRIVATE Threads::Threads)
endif()
if(UNIX AND NOT APPLE)
    # 不依赖目标机器的 libstdc++ 版本
    target_link_options(friday-llm PRIVATE -static-libstdc++ -static-libgcc)
endif()
install(TARGETS friday-llm RUNTIME DESTINATION bin)

if(LLM_TESTS)
    file(GLOB LLM_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
    add_executable(friday-llm-tests ${LLM_TEST_SOURCES})
    target_include_directories(friday-llm-tests PRIVATE tests)
    target_link_libraries(friday-llm-tests PRIVATE llm_engine)
    if(MSVC)
        target_compile_options(friday-llm-tests PRIVATE /bigobj)
    endif()
endif()
