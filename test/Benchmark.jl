module Benchmark

using CAMNAS
using Base
using BenchmarkTools
using CSV, DataFrames
using SparseMatricesCSR

include("Utils.jl")
using .Utils

macro no_logging(func)
    quote
        # Save Environment
        saved_debug_env = get(ENV, "JULIA_DEBUG", "") # Use default value if not found
        ENV["JULIA_DEBUG"] = "" # Disable debug information during execution

        # Evaluate function
        result = $(esc(func))

        # Restore environment
        ENV["JULIA_DEBUG"] = saved_debug_env
        return result
    end
end

@kwdef struct Result
    decomp_elapses::Float64
    solve_elapses::Float64
end

@no_logging function benchmark(dpsim_matrix::dpsim_csr_matrix, rhs_vector::Vector{Float64}, accelerator::CAMNAS.AbstractAccelerator; samples::UInt=UInt(3))
    CAMNAS.set_accelerator!(accelerator)

    GC.enable(false)
    system_matrix_ptr = pointer_from_objref(dpsim_matrix)
    decomp_elapses = @belapsed decomp(Base.unsafe_convert(Ptr{dpsim_csr_matrix}, $system_matrix_ptr)) evals = samples
    
    solve_elapses = @belapsed solve(Base.unsafe_convert(Ptr{Cdouble}, $rhs_vector), 
                                Base.unsafe_convert(Ptr{Cdouble}, 
                                zeros(Float64, length($rhs_vector)))) evals = samples
    GC.enable(true)

    Result(
        decomp_elapses,
        solve_elapses
    )
end

function csr_to_dpsim(csr::SparseMatrixCSR)
    matrix = dpsim_csr_matrix(
        Base.unsafe_convert(Ptr{Cdouble}, csr.nzval),
        Base.unsafe_convert(Ptr{Cint}, convert(Array{Int32}, csr.rowptr)), #! Cint expects 32 bit value
        Base.unsafe_convert(Ptr{Cint}, convert(Array{Int32}, csr.colval)),
        Int32(csr.m),
        Int32(length(csr.nzval))
    )
end

function benchmark(csr::SparseMatrixCSR, rhs_vector::Vector{Float64}, accelerator::CAMNAS.AbstractAccelerator; samples::UInt=UInt(3))
    dpsim_matrix = csr_to_dpsim(csr)
    benchmark(dpsim_matrix, rhs_vector,  accelerator; samples = samples)
end

function benchmark(matrix::Matrix, rhs_vector::Vector{Float64}, accelerator::CAMNAS.AbstractAccelerator; samples::UInt=UInt(3))
    csr = Generator.to_zerobased_csr(matrix)
    benchmark(csr, rhs_vector,  accelerator; samples = samples)
end

function benchmark(matrix_path::AbstractString, rhs_path::AbstractString, accelerator::CAMNAS.AbstractAccelerator; samples::UInt=UInt(3))
    dpsim_matrix = Utils.read_input(Utils.ArrayPath(matrix_path))
    rhs_vector = Utils.read_input(Utils.VectorPath(rhs_path))
    benchmark(dpsim_matrix, rhs_vector,  accelerator; samples = samples)
end

function write_csv(path::AbstractString, data_frame::DataFrame)
    # Create folder for benchmark results
    benchmarkPath = "$(@__DIR__)/$path"
    mkpath(benchmarkPath)

    append = isfile("$path/data.csv") # with append no header is written
    CSV.write("$path/data.csv", data_frame; append=append)
end

# begin # Plot
#     using Plotly
#     using CSV, DataFrames

#     dimension = 1500

#     benchmarkPath = "$(@__DIR__)/../benchmark"
#     csv = CSV.read("$benchmarkPath/data.csv", DataFrame)

#     filtered = filter(row -> row.accelerator == "CAMNAS.NoAccelerator()" && row.dimension == dimension, csv)

#     cpu_trace = scatter(x=filtered.density,
#                         y=filtered.decomp_elapses, 
#                         mode="lines", 
#                         name="CAMNAS.NoAccelerator()")

#     filtered = filter(row -> row.accelerator == "CAMNAS.CUDAccelerator()" && row.dimension == dimension, csv)

#     gpu_trace = scatter(x=filtered.density,
#                         y=filtered.decomp_elapses, 
#                         mode="lines", 
#                         name="CAMNAS.CUDAccelerator()")

#     PlotlyJS.display(plot([cpu_trace, gpu_trace], Layout(title="Solvestep of $dimension",
#                                     xaxis=attr(title="Density"),
#                                     yaxis=attr(title="Time"))))
# end

end