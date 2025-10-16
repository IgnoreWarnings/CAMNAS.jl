module Generator

using Random
using SparseMatricesCSR
using LinearAlgebra
using Base

# Note: Density is rounded and not precicse.
function generate_matrix(dimension::UInt; density::Float64 = 0.01, magnitude_off::Float64 = 0.05, delta::Float64 = 0.5, seed=rand(UInt))
    Random.seed!(seed)
    matrix = zeros(Float64, dimension, dimension)

    for i in 1:dimension
        s = max(1, Int(round(density * (dimension - 1))))
        cols = collect(1:dimension)
        deleteat!(cols, i) # remove diagonal index
        cols = randperm(length(cols))[1:s] # pick s random off-diagonal columns
        for j in cols
            matrix[i, j] = rand() * 2magnitude_off - magnitude_off
        end
    
        # diagonal dominance
        row_sum = sum(abs.(matrix[i, :]))
        matrix[i, i] = row_sum + delta
    end

    return matrix
end

@kwdef struct Settings
    dimension::UInt
    density::Float64
    magnitude_off::Float64
    delta::Float64
    seed::UInt
end

function generate_matrix(settings::Settings)
    generate_matrix(
        settings.dimension,
        density = settings.density,
        magnitude_off = settings.magnitude_off,
        delta = settings.delta,
        seed = settings.seed
    )
end

function generate_rhs_vector(matrix::Matrix{Float64}, prefered_solution::Vector{Float64} = ones(Float64, size(matrix, 1))) 
    rhs_vector = matrix*prefered_solution
    return rhs_vector
end

function to_csr(matrix)
    csr = SparseMatrixCSR(matrix)
    csr.colval .-= 1  # Convert column indices to 0-based
    csr.rowptr .-= 1  # Convert row pointers to 0-based
    return csr
end

function to_files(csr::SparseMatrixCSR, rhs_vector::Vector{Float64}; matrix_path="$(@__DIR__)/system_matrix_generated.txt", rhs_path="$(@__DIR__)/rhs_generated.txt")
    # write matrix to file
    io = open(matrix_path, "w");                                                                                                                                                                                                                                                                                                                                               
    write(io, "$(csr.nzval)\n$(csr.rowptr)\n$(csr.colval)\n$(csr.m)\n$(length(csr.nzval))");                                                                                                                                                                                                                                                                                                                                                
    close(io); 

    # write rhs vector to file
    io = open(rhs_path, "w");   
    write(io, "$(rhs_vector)");
    close(io);
end

end