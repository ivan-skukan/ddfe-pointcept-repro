@echo off
set "POINTCEPT_ENV=C:\Users\Ivan\Desktop\projects\FUCKYOUKIM\.micromamba\envs\pointcept-gtx1050ti"
set "PATH=%POINTCEPT_ENV%;%POINTCEPT_ENV%\bin;%POINTCEPT_ENV%\Library\bin;%POINTCEPT_ENV%\Scripts;%PATH%"
set "PYTHONPATH=C:\Users\Ivan\Desktop\projects\FUCKYOUKIM\Pointcept"
set "TORCH_CUDA_ARCH_LIST=6.1"
echo Pointcept env ready: %POINTCEPT_ENV%
