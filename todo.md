# Important:
    (+) git upstream (julia deps push not working)
        -- Use cudacode for cudss
        -- test examples

    - test_interface for compiled plugin
    - cuda in docker?

# Not Important:
    - compile camnas on arm (grace)
    - use release profile of dpsim

# Known Issues:
- CUDSS takes very long on first solve

- Metal package causes shared librari error
    In docker
    ┌ Error: Metal.jl is only supported on Apple Silicon
    └ @ Metal ~/.julia/packages/Metal/TF981/src/initialization.jl:59

- Vscode cmnd+enter adds random execution time overhead
     - use: CUDA_VISIBLE_DEVICES=0 julia -t4,0 test/test_interface.jl



