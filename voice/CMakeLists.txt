# friday-voice：本地 GPU 实时语音（Friday 的 Power）
#
# 一个可执行文件：ggml（CUDA / Vulkan / Metal / OpenCL / CPU 静态链进来）+ SenseVoice 识别 + FSMN-VAD + HTTP/WebSocket 服务。
# 模型不打进文件，运行时由 --model / --vad 指定（Friday 负责下载）。
#
#   cmake -B build -G Ninja -DVOICE_CUDA=ON          # 或 -DVOICE_VULKAN=ON，可同时开
#   cmake --build build                               # 产物 build/bin/friday-voice
cmake_minimum_required(VERSION 3.24)
project(friday-voice VERSION 0.1.0 LANGUAGES C CXX)

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

option(VOICE_CUDA   "CUDA 后端（NVIDIA）" OFF)
option(VOICE_VULKAN "Vulkan 后端（NVIDIA / AMD / Intel / Adreno）" OFF)
option(VOICE_METAL  "Metal 后端（Apple）" OFF)
option(VOICE_OPENCL "OpenCL 后端（Adreno 等）" OFF)
option(VOICE_TTS    "流式合成（CosyVoice3，cosyvoice.cpp）" ON)

# ── ggml：静态、所有后端编进同一个文件 ──
set(BUILD_SHARED_LIBS   OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL     OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE         OFF CACHE BOOL "")          # 默认可移植；本机调试可 -DGGML_NATIVE=ON
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP         OFF CACHE BOOL "" FORCE)    # 用 ggml 自己的线程池：OpenMP 运行库（vcomp140.dll / libgomp）是动态的
set(GGML_CUDA   ${VOICE_CUDA}   CACHE BOOL "" FORCE)
set(GGML_VULKAN ${VOICE_VULKAN} CACHE BOOL "" FORCE)
set(GGML_METAL  ${VOICE_METAL}  CACHE BOOL "" FORCE)
set(GGML_OPENCL ${VOICE_OPENCL} CACHE BOOL "" FORCE)
if(VOICE_METAL)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
endif()
if(VOICE_CUDA)
    set(GGML_CUDA_NO_VMM ON CACHE BOOL "" FORCE)     # 不直接依赖 libcuda 的 VMM 接口
    if(UNIX AND NOT APPLE)
        # Linux（Jetson）上 cudart / cuBLAS 都静态链进来，只剩驱动的 libcuda.so（运行时 dlopen）。
        # GGML_STATIC 的 -static 只加在 ggml 自己目录的目标上，不影响我们的可执行文件
        set(GGML_STATIC ON CACHE BOOL "" FORCE)
    endif()
endif()

include(FetchContent)
FetchContent_Declare(ggml
    URL https://github.com/ggml-org/ggml/archive/refs/tags/v0.26.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(civetweb
    URL https://github.com/civetweb/civetweb/archive/refs/tags/v1.16.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(ggml)
FetchContent_GetProperties(civetweb)
if(NOT civetweb_POPULATED)
    if(POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif()
    FetchContent_Populate(civetweb)       # 只要 civetweb.c，不走它自己的 CMake（会编测试、SSL）
endif()

# 单头文件依赖：json.hpp（nlohmann/json 3.12.0）、miniaudio.h（0.11.25，只用解码和重采样）
set(VOICE_HEADER_DIR ${CMAKE_BINARY_DIR}/_deps/headers)
foreach(spec
        "json.hpp|https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp"
        "miniaudio.h|https://raw.githubusercontent.com/mackron/miniaudio/0.11.25/miniaudio.h")
    string(REPLACE "|" ";" parts "${spec}")
    list(GET parts 0 name)
    list(GET parts 1 url)
    if(NOT EXISTS ${VOICE_HEADER_DIR}/${name})
        file(DOWNLOAD ${url} ${VOICE_HEADER_DIR}/${name} STATUS st TLS_VERIFY ON)
        list(GET st 0 code)
        if(NOT code EQUAL 0)
            file(REMOVE ${VOICE_HEADER_DIR}/${name})
            message(FATAL_ERROR "下载 ${name} 失败：${st}")
        endif()
    endif()
endforeach()

# ── civetweb：HTTP + WebSocket 服务器，C 单文件 ──
add_library(civetweb STATIC ${civetweb_SOURCE_DIR}/src/civetweb.c)
target_include_directories(civetweb PUBLIC ${civetweb_SOURCE_DIR}/include PRIVATE ${civetweb_SOURCE_DIR}/src)
target_compile_definitions(civetweb PUBLIC NO_SSL USE_WEBSOCKET NO_CGI NO_FILES NO_CACHING MG_EXPERIMENTAL_INTERFACES=0)
if(WIN32)
    target_link_libraries(civetweb PUBLIC ws2_32)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(civetweb PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
endif()

# ── cosyvoice.cpp：CosyVoice3 合成，只编库本身（不走它的 CMake：它会再带一份 ggml、开 LTO、编工具）──
#   不要 ONNX 前端（音色 prompt_speech.gguf 离线生成好随模型下载）、不要 ICU（数字读法以后自己规整）、不要它的音频读写
if(VOICE_TTS)
    FetchContent_Declare(cosyvoice
        URL https://github.com/Lourdle/cosyvoice.cpp/archive/6743f9299f4a9cecd05b8f7a2ad5a45a65cf98b1.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    # 和 cosyvoice.cpp 的 vendor/pcre2 子模块同一个提交（分词的正则用）
    FetchContent_Declare(pcre2
        URL https://github.com/PCRE2Project/pcre2/archive/7978954dbd2efc6f2196869290553cf1871b4ce6.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_GetProperties(cosyvoice)
    if(NOT cosyvoice_POPULATED)
        FetchContent_Populate(cosyvoice)
    endif()
    set(PCRE2_BUILD_PCRE2_8     ON  CACHE BOOL "" FORCE)
    set(PCRE2_BUILD_PCRE2_16    ON  CACHE BOOL "" FORCE)
    set(PCRE2_BUILD_TESTS       OFF CACHE BOOL "" FORCE)
    set(PCRE2_BUILD_PCRE2GREP   OFF CACHE BOOL "" FORCE)
    set(PCRE2_SUPPORT_JIT       OFF CACHE BOOL "" FORCE)
    set(PCRE2_STATIC_RUNTIME    ON  CACHE BOOL "" FORCE)
    set(BUILD_STATIC_LIBS       ON  CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(pcre2)

    set(CV ${cosyvoice_SOURCE_DIR})
    file(GLOB CV_SOURCES ${CV}/src/*.cpp ${CV}/common/*.cpp)
    list(REMOVE_ITEM CV_SOURCES
        ${CV}/src/cosyvoice-audio.cpp ${CV}/src/cosyvoice-audio-ffmpeg.cpp ${CV}/src/cosyvoice-frontend.cpp)
    include(CheckCXXSourceCompiles)
    check_cxx_source_compiles("#if !defined(__x86_64__) && !defined(_M_X64)\n#error not x86\n#endif\nint main() { return 0; }" VOICE_X86)
    set(CV_DEFS COSYVOICE_NO_ICU)
    if(VOICE_X86)
        # SIMD 各档一个源文件、各用自己的指令集编，运行时按 CPU 挑
        set(CV_GNU_sse42  -msse4.2)
        set(CV_GNU_avx    -msse4.2 -mavx)
        set(CV_GNU_avx2   -msse4.2 -mavx -mavx2 -mfma)
        set(CV_GNU_avx512 -msse4.2 -mavx -mavx2 -mfma -mavx512f -mavx512bw -mavx512dq)
        set(CV_MSVC_sse42  /arch:SSE4.2)
        set(CV_MSVC_avx    /arch:AVX)
        set(CV_MSVC_avx2   /arch:AVX2)
        set(CV_MSVC_avx512 /arch:AVX512)
        list(APPEND CV_DEFS COSYVOICE_HAS_SCALAR COSYVOICE_HAS_SSE42 COSYVOICE_HAS_AVX COSYVOICE_HAS_AVX2 COSYVOICE_HAS_AVX512)
        foreach(name scalar sse42 avx avx2 avx512)
            set(file ${CV}/src/simd/kernels_${name}.cpp)
            list(APPEND CV_SOURCES ${file})
            if(MSVC)
                set(flags ${CV_MSVC_${name}})
            else()
                set(flags ${CV_GNU_${name}})
            endif()
            if(flags)
                set_source_files_properties(${file} PROPERTIES COMPILE_OPTIONS "${flags}")
            endif()
        endforeach()
    else()
        # ARM 等：SSE4.2 那一档用 SIMDe 映射到 NEON（上游的做法），没有运行时分派
        FetchContent_Declare(simde
            URL https://github.com/simd-everywhere/simde/archive/refs/tags/v0.8.2.zip
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
        FetchContent_GetProperties(simde)
        if(NOT simde_POPULATED)
            FetchContent_Populate(simde)
        endif()
        list(REMOVE_ITEM CV_SOURCES ${CV}/src/simd_detect.cpp ${CV}/src/cosyvoice-simd.cpp)
        list(APPEND CV_SOURCES ${CV}/src/simd/kernels_sse42.cpp)
        list(APPEND CV_DEFS COSYVOICE_HAS_SSE42)
        set(CV_SIMDE_DIR ${simde_SOURCE_DIR})
    endif()
    add_library(cosyvoice_core STATIC ${CV_SOURCES})
    target_compile_features(cosyvoice_core PRIVATE cxx_std_20)
    target_include_directories(cosyvoice_core PUBLIC ${CV}/include PRIVATE ${CV}/src ${CV}/common)
    if(CV_SIMDE_DIR)
        target_include_directories(cosyvoice_core SYSTEM PRIVATE ${CV_SIMDE_DIR})
    endif()
    target_compile_definitions(cosyvoice_core PUBLIC COSYVOICE_STATIC PRIVATE ${CV_DEFS})
    target_link_libraries(cosyvoice_core PRIVATE ggml pcre2-8 pcre2-16)
    if(MSVC)
        target_compile_options(cosyvoice_core PRIVATE /Zc:preprocessor /bigobj /W0)
    else()
        target_compile_options(cosyvoice_core PRIVATE -w)
    endif()
endif()

# ── friday-voice ──
add_executable(friday-voice
    src/main.cpp
    src/Backend.cpp
    src/Fbank.cpp
    src/SenseVoice.cpp
    src/FsmnVad.cpp
    src/Streamer.cpp
    src/AudioIO.cpp
    src/Server.cpp)
target_include_directories(friday-voice PRIVATE src ${VOICE_HEADER_DIR})
target_compile_definitions(friday-voice PRIVATE
    VOICE_VERSION="${PROJECT_VERSION}"
    $<$<BOOL:${VOICE_CUDA}>:VOICE_HAS_CUDA>
    $<$<BOOL:${VOICE_VULKAN}>:VOICE_HAS_VULKAN>
    $<$<BOOL:${VOICE_METAL}>:VOICE_HAS_METAL>
    $<$<BOOL:${VOICE_OPENCL}>:VOICE_HAS_OPENCL>)
target_link_libraries(friday-voice PRIVATE ggml civetweb)
if(VOICE_TTS)
    target_sources(friday-voice PRIVATE src/Synthesizer.cpp src/TextSplitter.cpp src/Speaker.cpp)
    target_compile_definitions(friday-voice PRIVATE VOICE_HAS_TTS)
    target_link_libraries(friday-voice PRIVATE cosyvoice_core)
endif()
if(WIN32)
    target_link_libraries(friday-voice PRIVATE shell32)
else()
    target_link_libraries(friday-voice PRIVATE Threads::Threads)
endif()
if(UNIX AND NOT APPLE)
    # 不依赖目标机器的 libstdc++ 版本（Jetson 的 JetPack 4 / 6 都比编译环境旧或不同）
    target_link_options(friday-voice PRIVATE -static-libstdc++ -static-libgcc)
endif()
if(VOICE_TTS)
    # 默认音色随包发，放在可执行文件旁边的 voices/
    add_custom_command(TARGET friday-voice POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/assets/voices $<TARGET_FILE_DIR:friday-voice>/voices)
endif()
install(TARGETS friday-voice RUNTIME DESTINATION bin)
