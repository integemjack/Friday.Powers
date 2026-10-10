# friday-diffusion：本地生图 / 生视频（Friday 的 Power）
#
# 一个可执行文件：stable-diffusion.cpp（带它自己那份 ggml，CUDA / Vulkan / Metal / CPU 静态链进来）+ 任务队列 + HTTP 服务。
# 模型不打进文件：Friday 的模型库下载扩散模型和它要的 VAE、文本编码器，请求里告诉它用哪几个文件（docs/diffusion-protocol.md）。
#
#   cmake -B build -G Ninja -DDIFFUSION_CUDA=ON     # 或 -DDIFFUSION_VULKAN=ON / -DDIFFUSION_METAL=ON
#   cmake --build build                              # 产物 build/bin/friday-diffusion
cmake_minimum_required(VERSION 3.24)
project(friday-diffusion VERSION 0.1.0 LANGUAGES C CXX)

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

option(DIFFUSION_CUDA   "CUDA 后端（NVIDIA）" OFF)
option(DIFFUSION_VULKAN "Vulkan 后端（NVIDIA / AMD / Intel / Adreno）" OFF)
option(DIFFUSION_METAL  "Metal 后端（Apple）" OFF)

set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL      OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE          OFF CACHE BOOL "")
set(GGML_OPENMP          OFF CACHE BOOL "" FORCE)
set(SD_BUILD_EXAMPLES    OFF CACHE BOOL "" FORCE)
set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(SD_WEBP              ON  CACHE BOOL "" FORCE)    # 视频出 WebM（VP8）：libwebp 编码、libwebm 封装
set(SD_WEBM              ON  CACHE BOOL "" FORCE)
set(SD_CUDA   ${DIFFUSION_CUDA}   CACHE BOOL "" FORCE)
set(SD_VULKAN ${DIFFUSION_VULKAN} CACHE BOOL "" FORCE)
set(SD_METAL  ${DIFFUSION_METAL}  CACHE BOOL "" FORCE)
if(DIFFUSION_METAL)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
endif()
if(DIFFUSION_CUDA)
    set(GGML_CUDA_NO_VMM ON CACHE BOOL "" FORCE)
    if(UNIX AND NOT APPLE)
        set(GGML_STATIC ON CACHE BOOL "" FORCE)
    endif()
endif()

include(FetchContent)
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()
# stable-diffusion.cpp 的版本与它的子模块（换版本时一起改：ggml、libwebp、libwebm 的提交见上游的 .gitmodules / thirdparty）
set(DIFFUSION_SD_TAG master-951-f89d9b1)
FetchContent_Declare(sdcpp
    URL https://github.com/leejet/stable-diffusion.cpp/archive/refs/tags/${DIFFUSION_SD_TAG}.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_GetProperties(sdcpp)
if(NOT sdcpp_POPULATED)
    FetchContent_Populate(sdcpp)
endif()
FetchContent_Declare(sdggml
    URL https://github.com/leejet/ggml/archive/d25b121ce1d6ae0c7e62ad080460d9fc1e46dd93.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_DIR ${sdcpp_SOURCE_DIR}/ggml)
FetchContent_Declare(sdwebp
    URL https://github.com/webmproject/libwebp/archive/0c9546f7efc61eac7f79ae115c3f99c91c21c443.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_DIR ${sdcpp_SOURCE_DIR}/thirdparty/libwebp)
FetchContent_Declare(sdwebm
    URL https://github.com/webmproject/libwebm/archive/5bf12267eea773a32fcf4949de52b0add158a8d5.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_DIR ${sdcpp_SOURCE_DIR}/thirdparty/libwebm)
foreach(dep sdggml sdwebp sdwebm)
    FetchContent_GetProperties(${dep})
    if(NOT ${dep}_POPULATED)
        FetchContent_Populate(${dep})
    endif()
endforeach()
# 上游给 libwebm 定死了 MSVC 的动态运行库（/MD），我们是静态的（/MT）：改成 static，免得链接时运行库对不上
file(READ ${sdcpp_SOURCE_DIR}/thirdparty/CMakeLists.txt _sd_thirdparty)
string(REPLACE "set(MSVC_RUNTIME dll)" "set(MSVC_RUNTIME static)" _sd_thirdparty "${_sd_thirdparty}")
file(WRITE ${sdcpp_SOURCE_DIR}/thirdparty/CMakeLists.txt "${_sd_thirdparty}")
add_subdirectory(${sdcpp_SOURCE_DIR} ${CMAKE_BINARY_DIR}/_deps/sdcpp-build)

FetchContent_Declare(civetweb
    URL https://github.com/civetweb/civetweb/archive/refs/tags/v1.16.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_GetProperties(civetweb)
if(NOT civetweb_POPULATED)
    FetchContent_Populate(civetweb)
endif()
add_library(civetweb STATIC ${civetweb_SOURCE_DIR}/src/civetweb.c)
target_include_directories(civetweb PUBLIC ${civetweb_SOURCE_DIR}/include PRIVATE ${civetweb_SOURCE_DIR}/src)
target_compile_definitions(civetweb PUBLIC NO_SSL NO_CGI NO_FILES NO_CACHING MG_EXPERIMENTAL_INTERFACES=0)
if(WIN32)
    target_link_libraries(civetweb PUBLIC ws2_32)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(civetweb PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
endif()

# sd.cpp 示例里读写图片 / 视频的那两个文件（stb、WebP、WebM）
add_library(sd_media STATIC
    ${sdcpp_SOURCE_DIR}/examples/common/media_io.cpp
    ${sdcpp_SOURCE_DIR}/examples/common/log.cpp)
target_include_directories(sd_media PUBLIC ${sdcpp_SOURCE_DIR}/examples/common ${sdcpp_SOURCE_DIR}/examples)
target_compile_definitions(sd_media PUBLIC SD_USE_WEBP SD_USE_WEBM)
target_link_libraries(sd_media PUBLIC stable-diffusion webp libwebpmux webm)
if(MSVC)
    target_compile_options(sd_media PRIVATE /W0)
else()
    target_compile_options(sd_media PRIVATE -w)
endif()

add_executable(friday-diffusion
    src/main.cpp
    src/Generator.cpp
    src/Server.cpp)
target_include_directories(friday-diffusion PRIVATE src ${sdcpp_SOURCE_DIR}/include ${sdcpp_SOURCE_DIR}/thirdparty)
target_compile_definitions(friday-diffusion PRIVATE
    DIFFUSION_VERSION="${PROJECT_VERSION}"
    DIFFUSION_SD_TAG="${DIFFUSION_SD_TAG}")
target_link_libraries(friday-diffusion PRIVATE stable-diffusion sd_media civetweb)
if(WIN32)
    target_link_libraries(friday-diffusion PRIVATE shell32)
else()
    target_link_libraries(friday-diffusion PRIVATE Threads::Threads)
endif()
if(UNIX AND NOT APPLE)
    target_link_options(friday-diffusion PRIVATE -static-libstdc++ -static-libgcc)
endif()
install(TARGETS friday-diffusion RUNTIME DESTINATION bin)
# 开发时直接跑 build/bin 里的：清单放在可执行文件旁边（模型家族的缺省参数从这里读）
add_custom_command(TARGET friday-diffusion POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/power.json $<TARGET_FILE_DIR:friday-diffusion>/power.json)
