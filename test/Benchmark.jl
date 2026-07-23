#=
Author: Pascal Bauer <pascal.bauer@rwth-aachen.de>
SPDX-FileCopyrightText: 2025 Pascal Bauer <pascal.bauer@rwth-aachen.de>
=#

module Benchmark

using CAMNAS

using Base
using BenchmarkTools
using CSV, DataFrames


""" Example Use
    # FORCE Accelerator if not using runtime switch
    ENV["JL_MNA_RUNTIME_SWITCH"] = "false"
    ENV["JL_MNA_SPECIFIC_ACCELERATOR_STRATEGY"] = "true"
    ENV["JL_MNA_SPECIFIC_ACCELERATOR"] = "CUDSS Tesla P40(0)"

    include("Benchmark.jl")
    include("Generator.jl")
    include("Utils.jl")
    include("MatrixValidator.jl")

    benchmarkPath = Benchmark.get_next_folder()

    function build_generator_settings()
        # Matrix settings
        generator_settings = []
        dimensions = collect(200:200:2000)
        densities = collect(0.01:0.1:0.2)

        for dimension in dimensions
            for density in densities
                setting = Generator.Settings(
                    dimension=dimension,
                    density=density,
                    seed=1337
                )
                push!(generator_settings, setting)
            end
        end

        return generator_settings
    end

    function prepare_strategies()
        accelerators = ["cpu"] #"Tesla P40(2)", "NVIDIA GH200 144G HBM3e(0)", "cpu"]

        strategies = []
        for accelerator in accelerators
            push!(strategies, Dict("allow_strategies" => true, "specific_accelerator_strategy" => true, "specific_accelerator" => accelerator))
        end

        return strategies
    end

    function save_input(matrix)
        # Save matrix file
        csr_matrix = Utils.to_zerobased_csr(matrix)
        matrix_path = "$benchmarkPath/system_matrix_($(size(matrix, 1)))_($(MatrixValidator.density(matrix))).txt"
        Generator.matrix_to_file(csr_matrix, matrix_path=matrix_path)

        return matrix_path
    end

    function await_config_update(strategy)
        # Change Camnas strategy
        CAMNAS.update_varDict!(strategy)

        # Spinlock
        if ENV["JL_MNA_RUNTIME_SWITCH"] == true
            while CAMNAS.current_accelerator.name != strategy["specific_accelerator"]
                println(CAMNAS.current_accelerator.name)
                println(strategy["specific_accelerator"])
                sleep(2)
            end
        end
    end

    ### Run

    # Generated
    generator_settings_vector = build_generator_settings()
    matrix_iter = Generator.LazyMatrixBuilder(generator_settings_vector)

    # From File
    # matrix_paths = ["test/system_matrix_small.txt", "test/system_matrix_medium.txt", "test/system_matrix_big.txt"]
    # matrix_iter = Generator.LazyMatrixLoader(matrix_paths)
    # rhs_paths = ["test/rhs_small.txt", "test/rhs_medium.txt", "test/rhs_big.txt"]
    # rhs_index = 1

    for matrix in matrix_iter
        matrix_path = save_input(matrix)

        # Calculate decomposition, store state in CAMNAS
        GC.enable(false)
        dpsim_matrix = Utils.julia_to_dpsim(matrix)
        system_matrix_ptr = pointer_from_objref(dpsim_matrix)
        ptr = Base.unsafe_convert(Ptr{dpsim_csr_matrix}, system_matrix_ptr)
        decomp(ptr)

        # RHS Vectors
        RUNS = 10

        # Generated
        rhs_vectors = [Generator.generate_rhs_vector(matrix; prefered_solution=fill(Float64(i), size(matrix, 1))) for i in 1:RUNS] #rand(size(matrix, 1)))

        # # From File
        # rhs_vectors = []
        # for run in 1:RUNS
        #     push!(rhs_vectors, Utils.read_input(Utils.VectorPath(rhs_paths[rhs_index])))
        # end
        # global rhs_index += 1

        strategies = [CAMNAS.varDict] #prepare_strategies()
        for strategy in strategies
            await_config_update(strategy)

            for (i, rhs) in enumerate(rhs_vectors)
                print("Run $i of $(length(rhs_vectors))")
                metrics = Benchmark.benchmark_solve(rhs)
                Benchmark.save_csv("$benchmarkPath/benchmark.csv", metrics, CAMNAS.varDict, matrix_path) # TODO: Add RHS and RESULT
                println(" completed.")
            end
        end

        GC.enable(true)
    end

"""

function get_next_folder(base_dir="benchmark/")
    runs = filter(name -> occursin(r"^run_\d+$", name), readdir(base_dir))

    if isempty(runs)
        return joinpath(base_dir, "run_1")
    end

    nums = parse.(Int, replace.(runs, r"^run_" => ""))
    next_num = maximum(nums) + 1

    return joinpath(base_dir, "run_$(next_num)")
end

function benchmark_solve(rhs::Vector{Float64})
    solve = @elapsed begin
        lhs = zeros(Float64, length(rhs))
        CAMNAS.solve(Base.unsafe_convert(Ptr{Cdouble}, rhs), Base.unsafe_convert(Ptr{Cdouble}, lhs))
    end

    Dict("solve" => solve)
end

"""
    function save_csv(path::AbstractString, benchmark_metrics::Benchmarkmetrics, matrix_path::String)

"""
function save_csv(path::AbstractString, metrics::Dict, strategy::Dict, matrix_path::String)
    data_frame = DataFrame(
        metrics=[metrics],
        strategy=[strategy],
        matrix_path=[matrix_path]
    )

    # Create folder for csv
    mkpath(dirname("$path"))

    append = isfile("$path") # with append no header is written
    CSV.write("$path", data_frame; append=append)
end

end