$ErrorActionPreference = "Stop"

# cmake ships inside the VS Build Tools install; find it the way CLAUDE.md says
# to - through vswhere - because Build Tools lives under the x86 Program Files
# and Community lives under the other root.
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$cmake = Join-Path $vs "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
Write-Output "cmake: $cmake"

Set-Location C:\prog\llama.cpp

# CPU only: this box has the RTX 3090 driver but no CUDA toolkit, so there is no
# nvcc to build the GPU backend with. CPU is enough for what this is for - a
# handful of calls per game, not per action.
& $cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
    -DCMAKE_BUILD_TYPE=Release `
    -DLLAMA_BUILD_TESTS=OFF `
    -DLLAMA_BUILD_EXAMPLES=OFF `
    -DLLAMA_CURL=OFF
if ($LASTEXITCODE -ne 0) { throw "configure failed" }

& $cmake --build build --config Release --target llama-cli llama-server -- /m
if ($LASTEXITCODE -ne 0) { throw "build failed" }

Get-ChildItem -Recurse -Path build -Include llama-cli.exe, llama-server.exe |
    Select-Object FullName, @{n = "MB"; e = { [math]::Round($_.Length / 1MB, 1) } } |
    Format-Table -AutoSize | Out-String -Width 200
Write-Output "BUILD_OK"
