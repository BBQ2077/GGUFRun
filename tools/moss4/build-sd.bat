@echo off
rem Rebuild the GGUFRun MOSS runtime: stable-diffusion.cpp master b13a482 + ggufrun-moss-dataspace.patch,
rem gfx1201 (RX 9000), ROCm 10.1.0 pip SDK. Output: RUNTIMES\sd-master-b13a482-moss4-gfx1201-rocm-10.1.0-x64
rem Needs: Visual Studio 2022 (C++ workload, includes CMake + Ninja), git, uv. Nothing is installed system-wide;
rem the SDK venv, source checkout and build tree live in %WORK% (default tools\moss4\.work, about 5 GB).
setlocal
set "HERE=%~dp0"
for %%i in ("%HERE%..\..") do set "ROOT=%%~fi"
if not defined WORK set "WORK=%HERE%.work"
if not defined VCVARS set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set "VENV=%WORK%\venv"
set "SRC=%WORK%\stable-diffusion.cpp"
set "BUILD=%WORK%\build-gfx1201"
set "SD_COMMIT=b13a482"
call "%VCVARS%" >nul || (echo vcvars64.bat not found: set VCVARS & exit /b 1)

if not exist "%VENV%\Scripts\rocm-sdk.exe" (
  uv venv --seed --python 3.12 "%VENV%" || exit /b 1
  "%VENV%\Scripts\python.exe" -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm[libraries,devel]==10.1.0" || exit /b 1
  "%VENV%\Scripts\rocm-sdk.exe" init || exit /b 1
)
if not exist "%SRC%\.git" (
  git clone https://github.com/leejet/stable-diffusion.cpp "%SRC%" || exit /b 2
  git -C "%SRC%" checkout %SD_COMMIT% || exit /b 2
  git -C "%SRC%" submodule update --init --recursive || exit /b 2
  git -C "%SRC%" apply "%HERE%ggufrun-moss-dataspace.patch" || exit /b 2
)

for /f "usebackq delims=" %%i in (`"%VENV%\Scripts\rocm-sdk.exe" path --root`) do set "HIP_PATH=%%i"
for /f "usebackq delims=" %%i in (`"%VENV%\Scripts\rocm-sdk.exe" path --bin`) do set "ROCM_BIN=%%i"
set "HIP_PLATFORM=amd"
set "HIP_DEVICE_LIB_PATH=%HIP_PATH%\lib\llvm\amdgcn\bitcode"
set "LLVM_PATH=%HIP_PATH%\lib\llvm"
set "VSCM=%VSINSTALLDIR%Common7\IDE\CommonExtensions\Microsoft\CMake"
set "PATH=%VSCM%\CMake\bin;%VSCM%\Ninja;%ROCM_BIN%;%VENV%\Scripts;%PATH%"
echo HIP_PATH=%HIP_PATH%
if not exist "%BUILD%\build.ninja" (
  cmake -S "%SRC%" -B "%BUILD%" -G Ninja ^
    -DCMAKE_PREFIX_PATH="%HIP_PATH%" ^
    -DSD_HIPBLAS=ON -DSD_BUILD_SHARED_LIBS=ON -DGGML_NATIVE=OFF ^
    -DCMAKE_C_COMPILER="%HIP_PATH%\lib\llvm\bin\clang.exe" ^
    -DCMAKE_CXX_COMPILER="%HIP_PATH%\lib\llvm\bin\clang++.exe" ^
    -DCMAKE_HIP_COMPILER="%HIP_PATH%\lib\llvm\bin\clang.exe" ^
    -DHIP_PATH="%HIP_PATH%" -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx1201 || exit /b 3
)
cmake --build "%BUILD%" --config Release --parallel %NUMBER_OF_PROCESSORS% || exit /b 3
"%VENV%\Scripts\python.exe" "%HERE%package_runtime.py" "%BUILD%\bin" "%VENV%" || exit /b 4
echo BUILD_OK
