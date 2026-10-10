#!/usr/bin/env bash
# 在 Linux 上编一个 Power（llm、diffusion；voice 用 voice/ci/build-linux.sh）：
#   ci/build-linux.sh <power> <平台> <build 目录>
# CMake 选项的前缀是 Power id 的大写（LLM_CUDA、DIFFUSION_VULKAN…）。平台：
#   linux-x64-vulkan         在 ubuntu-22.04 运行器上编（glibc 2.35，同 Friday 的 Linux x64 包），要 LunarG 的 Vulkan SDK 和 GCC 13
#   linux-x64-cuda           x64 + CUDA 13，cudart、cuBLAS 静态链接（先 nvprune 裁到几个架构）
#   linux-arm64-jetson-orin  JetPack 6（在 nvcr.io/nvidia/l4t-cuda:12.2.12-devel 里编），cudart、cuBLAS 静态链接
#   linux-arm64-vulkan       ARM64 Linux + Vulkan（ubuntu:24.04 容器），要 ARMv8.2
set -euo pipefail
power=$1
target=$2
build=$3
root=$(cd "$(dirname "$0")/.." && pwd)
src=$root/$power
prefix=$(echo "$power" | tr '[:lower:]' '[:upper:]')
exe=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['executable'])" "$src/power.json")

prune() {
    # 静态 cuBLAS 带着所有架构的内核：只留要的几个
    local lib_dir=$1
    shift
    command -v nvprune >/dev/null || return 0
    for lib in libcublas_static.a libcublasLt_static.a; do
        path=$lib_dir/$lib
        [ -f "$path" ] || continue
        before=$(stat -c %s "$path")
        nvprune "$@" "$path" -o "/tmp/$lib"
        (sudo mv "/tmp/$lib" "$path" 2>/dev/null || mv "/tmp/$lib" "$path")
        echo "nvprune $lib：$((before / 1048576)) MB → $(($(stat -c %s "$path") / 1048576)) MB"
    done
}

common=(-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -D${prefix}_TESTS=OFF)
case "$target" in
linux-x64-vulkan)
    args=(-D${prefix}_VULKAN=ON -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON)
    ;;
linux-x64-cuda)
    cuda=${CUDA_PATH:-/usr/local/cuda}
    export PATH=$cuda/bin:$PATH
    prune "$cuda/lib64" -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 \
        -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120
    args=(-D${prefix}_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=75-virtual;86-real;89-real;120a-real" -DCMAKE_CUDA_COMPILER=$cuda/bin/nvcc
          -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_CUDA_HOST_COMPILER=g++-13
          -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON)
    ;;
linux-arm64-jetson-orin)
    export PATH=/usr/local/cuda/bin:$PATH
    prune /usr/local/cuda/lib64 --generate-code arch=compute_87,code=sm_87
    args=(-D${prefix}_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
          -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_CUDA_HOST_COMPILER=g++-11
          -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16)
    ;;
linux-arm64-vulkan)
    args=(-D${prefix}_VULKAN=ON -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16)
    ;;
*)
    echo "不认识的平台：$target" >&2
    exit 2
    ;;
esac

generator=(-G Ninja)
command -v ninja >/dev/null || generator=(-G "Unix Makefiles")
cmake -S "$src" -B "$build" "${generator[@]}" "${common[@]}" "${args[@]}"
cmake --build "$build" -j"$(nproc)" --target "$exe"

bin="$build/bin/$exe"
# stable-diffusion.cpp 直接链接驱动的 libcuda（Linux 上没法延迟加载）：CI 机器没装 NVIDIA 驱动时程序起不来，只看依赖。
# 用户机器上 Friday 只在有 libcuda 时才提供 CUDA 包
if ! version=$("$bin" --version 2>&1); then
    if [[ "$version" == *"libcuda.so"* ]]; then
        echo "（这台机器没有 NVIDIA 驱动，缺 libcuda：不跑 --version）"
    else
        echo "$version" >&2
        exit 1
    fi
else
    echo "$version"
fi
echo "── 动态依赖（只该有 glibc 一族，libcuda / libvulkan 由驱动提供）"
ldd "$bin" || true
echo "── 要求的最高 glibc：$(objdump -T "$bin" | grep -o 'GLIBC_[0-9.]*' | sort -Vu | tail -1)"
