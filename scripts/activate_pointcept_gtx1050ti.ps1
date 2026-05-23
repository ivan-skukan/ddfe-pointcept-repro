$env:POINTCEPT_ENV = "C:\Users\Ivan\Desktop\projects\FUCKYOUKIM\.micromamba\envs\pointcept-gtx1050ti"
$env:PATH = "$env:POINTCEPT_ENV;$env:POINTCEPT_ENV\bin;$env:POINTCEPT_ENV\Library\bin;$env:POINTCEPT_ENV\Scripts;$env:PATH"
$env:PYTHONPATH = "C:\Users\Ivan\Desktop\projects\FUCKYOUKIM\Pointcept"
$env:TORCH_CUDA_ARCH_LIST = "6.1"

Write-Host "Pointcept env ready: $env:POINTCEPT_ENV"
