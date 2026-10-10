#!/usr/bin/env bash
# 在 Linux（一般是 CI 的容器）里编 friday-voice：
#   voice/ci/build-linux.sh <平台> <build 目录>
# 平台：
#   linux-arm64-jetson-orin  JetPack 6（在 nvcr.io/nvidia/l4t-cuda:12.2.12-devel 里编，CUDA 12.2 编的在 6.x 的驱动上都能跑），cudart、cuBLAS 静态链接
#   linux-arm64-vulkan       ARM64 Linux + Vulkan（高通 Adreno 等），要 ARMv8.2
#   linux-x64-vulkan         x64 Linux + Vulkan（在 ubuntu-22.04 运行器上编，glibc 2.35，同 Friday 的 Linux x64 包；要 LunarG 的 Vulkan SDK 和 GCC 13）
#   linux-x64-cuda           x64 Linux + CUDA 13，cudart、cuBLAS 静态链接（先 nvprune 裁到下面这几个架构）
set -euo pipefail
target=$1
build=$2
src=$(cd "$(dirname "$0")/.." && pwd)

common=(-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF)
case "$target" in
linux-arm64-jetson-orin)
    export PATH=/usr/local/cuda/bin:$PATH
    # 静态的 cuBLAS 带着所有架构的内核（包要 400 MB）：容器里先裁到只剩 Orin 的 sm_87
    if command -v nvprune >/dev/null; then
        for lib in libcublas_static.a libcublasLt_static.a; do
            path=/usr/local/cuda/lib64/$lib
            if [ -f "$path" ]; then
                before=$(stat -c %s "$path")
                nvprune --generate-code arch=compute_87,code=sm_87 "$path" -o "/tmp/$lib" && mv "/tmp/$lib" "$path"
                echo "nvprune $lib：$((before / 1048576)) MB → $(($(stat -c %s "$path") / 1048576)) MB"
            fi
        done
    fi
    args=(-DVOICE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
          -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_CUDA_HOST_COMPILER=g++-11
          -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16)
    ;;
linux-x64-vulkan)
    args=(-DVOICE_VULKAN=ON -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13
          -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON)
    ;;
linux-x64-cuda)
    cuda=${CUDA_PATH:-/usr/local/cuda}
    export PATH=$cuda/bin:$PATH
    # 静态 cuBLAS 带着所有架构的内核：只留 Turing 起的这几个（sm_80 的也能在 86 / 89 上跑）
    if command -v nvprune >/dev/null; then
        for lib in libcublas_static.a libcublasLt_static.a; do
            path=$cuda/lib64/$lib
            if [ -f "$path" ]; then
                before=$(stat -c %s "$path")
                nvprune -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86                     -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120 "$path" -o "/tmp/$lib"
                sudo mv "/tmp/$lib" "$path" 2>/dev/null || mv "/tmp/$lib" "$path"
                echo "nvprune $lib：$((before / 1048576)) MB → $(($(stat -c %s "$path") / 1048576)) MB"
            fi
        done
    fi
    args=(-DVOICE_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=75-virtual;86-real;89-real;120a-real" -DCMAKE_CUDA_COMPILER=$cuda/bin/nvcc
          -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_CUDA_HOST_COMPILER=g++-13
          -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON)
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
