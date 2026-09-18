$ErrorActionPreference = "Stop"

# cmake ships inside the VS Build Tools install; find it the way CLAUDE.md says
# to - through vswhere - because Build Tools lives under the x86 Program Files
# and Community lives under the other root.
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$cmake = Join-Path $vs "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
Write-Output "cmake: $cmake"

Set-Location C:\prog\llama.cpp

# CUDA. The first build here was CPU-only because the box had the 3090 driver
# but no toolkit; the toolkit is now installed (winget Nvidia.CUDA 13.3) and the
# 27B at Q4 is 15.7GB, which fits whole into the card's 24GB. On CPU that model
# generates 2.1 tokens a second - one answer every 102 seconds, enough to ask a
# question once per game and nothing more. The point of the card is to make the
# loop affordable, not the single question.
$cudaRoot = Join-Path ${env:ProgramFiles} "NVIDIA GPU Computing Toolkit\CUDA\v13.3"
$env:CUDAToolkit_ROOT = $cudaRoot
$env:PATH = "$cudaRoot\bin;$env:PATH"

& $cmake -S . -B build-cuda -G "Visual Studio 17 2022" -A x64 `
    -DCMAKE_BUILD_TYPE=Release `
    -DGGML_CUDA=ON `
    -DCMAKE_CUDA_ARCHITECTURES=86 `
    -DLLAMA_BUILD_TESTS=OFF `
    -DLLAMA_BUILD_EXAMPLES=OFF `
    -DLLAMA_CURL=OFF
if ($LASTEXITCODE -ne 0) { throw "configure failed" }

& $cmake --build build-cuda --config Release --target llama-cli llama-server -- /m
if ($LASTEXITCODE -ne 0) { throw "build failed" }

Get-ChildItem -Recurse -Path build-cuda -Include llama-cli.exe, llama-server.exe |
    Select-Object FullName, @{n = "MB"; e = { [math]::Round($_.Length / 1MB, 1) } } |
    Format-Table -AutoSize | Out-String -Width 200
Write-Output "BUILD_OK"
