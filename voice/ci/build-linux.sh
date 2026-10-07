#!/usr/bin/env bash
# 在 Linux（一般是 CI 的容器）里编 friday-voice：
#   voice/ci/build-linux.sh <平台> <build 目录>
# 平台：
#   linux-arm64-jetson-orin  JetPack 6（在 nvcr.io/nvidia/l4t-cuda:12.2.12-devel 里编，CUDA 12.2 编的在 6.x 的驱动上都能跑），cudart、cuBLAS 静态链接
#   linux-arm64-jetson-nano  JetPack 4（glibc 2.27、CUDA 10.2 编不了 ggml）：在 manylinux2014 里编纯 CPU、只有听写
#   linux-arm64-vulkan       ARM64 Linux + Vulkan（高通 Adreno 等），要 ARMv8.2
set -euo pipefail
target=$1
build=$2
src=$(cd "$(dirname "$0")/.." && pwd)

common=(-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF)
case "$target" in
linux-arm64-jetson-orin)
    export PATH=/usr/local/cuda/bin:$PATH
    args=(-DVOICE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
          -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_CUDA_HOST_COMPILER=g++-11
          -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16)
    ;;
linux-arm64-jetson-nano)
    args=(-DVOICE_TTS=OFF -DGGML_CPU_ARM_ARCH=armv8-a)
    ;;
linux-arm64-vulkan)
    args=(-DVOICE_VULKAN=ON -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16)
    ;;
*)
    echo "不认识的平台：$target" >&2
    exit 2
    ;;
esac

generator=(-G Ninja)
command -v ninja >/dev/null || generator=(-G "Unix Makefiles")
cmake -S "$src" -B "$build" "${generator[@]}" "${common[@]}" "${args[@]}"
cmake --build "$build" -j"$(nproc)"

exe="$build/bin/friday-voice"
"$exe" --version
echo "── 动态依赖（只该有 glibc 一族，libcuda / libvulkan 由驱动提供）"
ldd "$exe" || true
echo "── 要求的最高 glibc：$(objdump -T "$exe" | grep -o 'GLIBC_[0-9.]*' | sort -Vu | tail -1)"
