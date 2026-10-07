# Windows 本机构建 friday-voice（开发用；CI 见 .github/workflows）。
#   pwsh scripts\build-windows.ps1 -Backend cuda   [-CudaArch 120a-real] [-Clean]
#   pwsh scripts\build-windows.ps1 -Backend vulkan （要装 Vulkan SDK）
#   pwsh scripts\build-windows.ps1 -Backend cpu
param(
    [ValidateSet('cuda', 'vulkan', 'cuda+vulkan', 'cpu')] [string]$Backend = 'cuda',
    [string]$CudaArch = '120a-real',
    [string]$CudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0',
    [string]$VsDevCmd = '',
    [switch]$Native,
    [switch]$Clean,
    [int]$Jobs = 0
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$build = Join-Path $root ("build-" + ($Backend -replace '\+', '-'))
if ($Clean -and (Test-Path $build)) { Remove-Item -Recurse -Force $build }

# MSVC 环境：找 vcvars64.bat（VS / Build Tools）
if (-not $env:VCToolsInstallDir) {
    $vcvars = $VsDevCmd
    if (-not $vcvars) {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
        if (Test-Path $vswhere) {
            $install = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
            if ($install) { $vcvars = Join-Path $install 'VC\Auxiliary\Build\vcvars64.bat' }
        }
        if (-not $vcvars -or -not (Test-Path $vcvars)) { $vcvars = 'G:\work\BuildTools\VC\Auxiliary\Build\vcvars64.bat' }
    }
    Write-Host "▶ 导入 MSVC 环境：$vcvars"
    cmd /c "`"$vcvars`" >nul && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2]) }
    }
}
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
    $ninja = Get-ChildItem -Path (Split-Path $env:VCToolsInstallDir -Parent | Split-Path -Parent | Split-Path -Parent) -Recurse -Filter ninja.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($ninja) { $env:PATH = "$($ninja.DirectoryName);$env:PATH" }
}

$cmakeArgs = @('-S', $root, '-B', $build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release')
if ($Native) { $cmakeArgs += '-DGGML_NATIVE=ON' } else { $cmakeArgs += @('-DGGML_AVX2=ON', '-DGGML_FMA=ON', '-DGGML_F16C=ON') }
if ($Backend -like '*cuda*') {
    $env:CUDA_PATH = $CudaRoot
    $env:PATH = "$CudaRoot\bin;$CudaRoot\bin\x64;$env:PATH"
    $cmakeArgs += @('-DVOICE_CUDA=ON', "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch", "-DCMAKE_CUDA_COMPILER=$CudaRoot\bin\nvcc.exe")
}
if ($Backend -like '*vulkan*') { $cmakeArgs += '-DVOICE_VULKAN=ON' }

& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "cmake 配置失败（$LASTEXITCODE）" }
$buildArgs = @('--build', $build)
if ($Jobs -gt 0) { $buildArgs += @('-j', $Jobs) }
$log = Join-Path $build 'build.log'
& cmake @buildArgs *> $log
$code = $LASTEXITCODE
Get-Content $log | Select-String -Pattern ' error |error C\d|error LNK|: error|FAILED:|fatal error' | Select-Object -First 30 | ForEach-Object { $_.Line }
if ($code -ne 0) { throw "构建失败（$code），完整日志：$log" }
$exe = Join-Path $build 'bin\friday-voice.exe'
Write-Host ("✓ {0}（{1:N1} MB）" -f $exe, ((Get-Item $exe).Length / 1MB))
