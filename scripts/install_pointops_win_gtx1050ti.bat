@echo off
setlocal

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%

set "ENV_PREFIX=C:\Users\Ivan\Desktop\projects\FUCKYOUKIM\.micromamba\envs\pointcept-gtx1050ti"
set "PATH=%ENV_PREFIX%;%ENV_PREFIX%\bin;%ENV_PREFIX%\Library\bin;%ENV_PREFIX%\Scripts;%PATH%"
set "TORCH_CUDA_ARCH_LIST=6.1"
set "DISTUTILS_USE_SDK=1"

"%ENV_PREFIX%\python.exe" -m pip install --no-build-isolation -e libs\pointops
