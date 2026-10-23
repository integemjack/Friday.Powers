# friday-devices：客户端设备（Friday 的 Power）
#
# 一个小的可执行文件：civetweb（HTTP + WebSocket）+ nlohmann/json，不用显卡、不要模型。
# 远程客户端那台电脑上的串口、摄像头、手机、局域网设备经 Friday 的工作通道开成本机端口、给智能体 MCP 工具（docs/devices-protocol.md）。
#
#   cmake -B build -G Ninja
#   cmake --build build                              # 产物 build/bin/friday-devices
cmake_minimum_required(VERSION 3.24)
project(friday-devices VERSION 0.2.0 LANGUAGES C CXX)

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
    add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS WIN32_LEAN_AND_MEAN _WINSOCK_DEPRECATED_NO_WARNINGS)
endif()

include(FetchContent)
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()
FetchContent_Declare(civetweb
    URL https://github.com/civetweb/civetweb/archive/refs/tags/v1.16.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_GetProperties(civetweb)
if(NOT civetweb_POPULATED)
    FetchContent_Populate(civetweb)       # 只要 civetweb.c，不走它自己的 CMake（会编测试、SSL）
endif()

# 单头文件依赖：json.hpp（nlohmann/json 3.12.0，同 voice）
set(DEVICES_HEADER_DIR ${CMAKE_BINARY_DIR}/_deps/headers)
if(NOT EXISTS ${DEVICES_HEADER_DIR}/json.hpp)
    file(DOWNLOAD https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp ${DEVICES_HEADER_DIR}/json.hpp STATUS st TLS_VERIFY ON)
    list(GET st 0 code)
    if(NOT code EQUAL 0)
        file(REMOVE ${DEVICES_HEADER_DIR}/json.hpp)
        message(FATAL_ERROR "下载 json.hpp 失败：${st}")
    endif()
endif()

add_library(civetweb STATIC ${civetweb_SOURCE_DIR}/src/civetweb.c)
target_include_directories(civetweb PUBLIC ${civetweb_SOURCE_DIR}/include PRIVATE ${civetweb_SOURCE_DIR}/src)
target_compile_definitions(civetweb PUBLIC NO_SSL USE_WEBSOCKET NO_CGI NO_FILES NO_CACHING MG_EXPERIMENTAL_INTERFACES=0)
if(WIN32)
    target_link_libraries(civetweb PUBLIC ws2_32)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(civetweb PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
endif()

add_executable(friday-devices
    src/main.cpp
    src/Bridge.cpp
    src/Monitor.cpp
    src/Ports.cpp
    src/Server.cpp
    src/Socket.cpp
    src/Tools.cpp)
target_include_directories(friday-devices PRIVATE src ${DEVICES_HEADER_DIR})
target_compile_definitions(friday-devices PRIVATE DEVICES_VERSION="${PROJECT_VERSION}")
target_link_libraries(friday-devices PRIVATE civetweb)
if(WIN32)
    target_link_libraries(friday-devices PRIVATE shell32 ws2_32)
else()
    target_link_libraries(friday-devices PRIVATE Threads::Threads)
endif()
if(UNIX AND NOT APPLE)
    target_link_options(friday-devices PRIVATE -static-libstdc++ -static-libgcc)
endif()
install(TARGETS friday-devices RUNTIME DESTINATION bin)
add_custom_command(TARGET friday-devices POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/power.json $<TARGET_FILE_DIR:friday-devices>/power.json)
