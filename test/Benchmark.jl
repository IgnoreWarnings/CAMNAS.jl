#=
Author: Pascal Bauer <pascal.bauer@rwth-aachen.de>
SPDX-FileCopyrightText: 2025 Pascal Bauer <pascal.bauer@rwth-aachen.de>
=#

module Benchmark

using CAMNAS

using Base
using BenchmarkTools
using SparseMatricesCSR
using SparseArrays
using CSV, DataFrames
using CUDA
using CUSOLVERRF
using CUDA.CUSPARSE
using LinearAlgebra

include("Utils.jl")
using .Utils

function find_gpu_decomp()
    idx = findfirst(x -> typeof(x) == CAMNAS.Accelerators.CUDAccelerator_LUdecomp, CAMNAS.system_matrix)
    if idx !== nothing
        return CAMNAS.system_matrix[idx].lu_decomp
    end

    throw("No decomp present.")
end

function cuda_benchmark(dpsim_matrix::dpsim_csr_matrix, rhs::Vector{Float64})
    GC.enable(false)

    # run all decomps
    system_matrix_ptr = pointer_from_objref(dpsim_matrix)
    decomp(Base.unsafe_convert(Ptr{dpsim_csr_matrix}, system_matrix_ptr))
    
    host2dev = @elapsed begin
        rhs_d = CuVector(rhs)
    end

    mat = find_gpu_decomp()
    solve = @elapsed begin
        ldiv!(mat, rhs_d)
    end

    dev2host = @elapsed begin
        result = Array(rhs_d)
    end

    GC.enable(true)

    Dict("host2dev" => host2dev, "solve" => solve, "dev2host" => dev2host)
end

function benchmark(rhs::Vector{Float64})
    solve = @elapsed begin
        lhs_vector = zeros(Float64, length(rhs))
        CAMNAS.solve(Base.unsafe_convert(Ptr{Cdouble}, rhs), Base.unsafe_convert(Ptr{Cdouble}, lhs_vector))
    end

    Dict("solve" => solve)
end


"""
    function benchmark(csr::SparseMatrixCSR, rhs::Vector{Float64}; samples::UInt=UInt(3))

Wrapper for `function benchmark(dpsim_matrix::dpsim_csr_matrix, rhs::Vector{Float64}; samples::UInt=UInt(3))`
"""
function cuda_benchmark(csr::SparseMatrixCSC, rhs::Vector{Float64})
    dpsim_matrix = Utils.csc_to_dpsim(csr)
    cuda_benchmark(dpsim_matrix, rhs)
end

"""
    function benchmark(matrix_path::AbstractString, rhs_path::AbstractString; samples::UInt=UInt(3))

Wrapper for `function benchmark(dpsim_matrix::dpsim_csr_matrix, rhs::Vector{Float64}; samples::UInt=UInt(3))`
"""
function cuda_benchmark(matrix_path::AbstractString, rhs_path::AbstractString)
    dpsim_matrix = Utils.read_input(Utils.ArrayPath(matrix_path))
    rhs = Utils.read_input(Utils.VectorPath(rhs_path))
    cuda_benchmark(dpsim_matrix, rhs; samples=samples)
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