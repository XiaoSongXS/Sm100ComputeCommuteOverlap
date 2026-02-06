
## Build
```shell
mkdir build
cd build
cmake ..
make -j32
```

## Env

nvcr.io/nvidia/nvhpc:26.1-devel-cuda_multi-ubuntu24.04

## Run
```shell
nsys profile --output=green_context_sample_profile --force-overwrite=true ./green_context_sample --delay_high_priority
```