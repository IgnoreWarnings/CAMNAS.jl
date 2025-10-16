using CAMNAS, Test
using Suppressor

@testset "CAMNAS" begin

    # Read and convert test inputs
    sys_mat_str = readlines("system_matrix_small.txt")
    sys_mat_str = replace.(sys_mat_str, r"[\[\],]" => "")

    values = parse.(Float64, split(sys_mat_str[1]))
    rowIndex = parse.(Cint, split(sys_mat_str[2]))
    colIndex = parse.(Cint, split(sys_mat_str[3]))

    sys_mat = dpsim_csr_matrix(
        Base.unsafe_convert(Ptr{Cdouble}, values),
        Base.unsafe_convert(Ptr{Cint}, rowIndex),
        Base.unsafe_convert(Ptr{Cint}, colIndex),
        parse(Int32, sys_mat_str[4]),
        parse(Int32, sys_mat_str[5])
    )
    sys_mat_ptr = pointer_from_objref(sys_mat)

    rhs_vec_strings = readlines("rhs_small.txt")
    rhs_vec_strings = replace.(rhs_vec_strings, r"[\[\],]" => "")
    rhs_vec = parse.(Float64, split(rhs_vec_strings[1]))

    lhs_vec = zeros(Float64, length(rhs_vec))

    @testset "Initialization" begin
        @test init(Base.unsafe_convert(Ptr{dpsim_csr_matrix}, sys_mat_ptr)) == 0
    end 

    @testset "Solving" begin
        @test solve(Base.unsafe_convert(Ptr{Cdouble}, rhs_vec), Base.unsafe_convert(Ptr{Cdouble}, lhs_vec)) == 0
        @test lhs_vec == [1.0, 1/2, 1/3]
    end

    @testset "Decomposition" begin
        # Change system matrix values
        sys_mat_bak = sys_mat.values
        sys_mat.values = Base.unsafe_convert(Ptr{Cdouble}, [4.0, 5.0, 6.0])

        # Decompose new system matrix
        @test decomp(Base.unsafe_convert(Ptr{dpsim_csr_matrix}, sys_mat_ptr)) == 0

        # Solve system with new matrix
        @test solve(Base.unsafe_convert(Ptr{Cdouble}, rhs_vec), Base.unsafe_convert(Ptr{Cdouble}, lhs_vec)) == 0
        @test lhs_vec == [1/4, 1/5, 1/6]

        # Restore system matrix values
        sys_mat.values = sys_mat_bak
    end        

    @testset "Generator" begin
        using LinearAlgebra
        include("Generator.jl")

        settings = Generator.Settings(
            dimension = rand(500:3000),
            density = rand(1:5)/10,
            magnitude_off = 0.05,
            delta = 0.5,
            seed = 1337
        )

        matrix = Generator.generate_matrix(settings)

        # Dimension
        @test (size(matrix, 1) == settings.dimension) && (size(matrix, 2) == settings.dimension)

        # Density
        non_zeroes = count(!iszero, matrix)
        total_elements = size(matrix, 1) * size(matrix, 2)
        density = non_zeroes / total_elements
        @test isapprox(density, settings.density; atol=0.001)

        # Condition
        condition_tresh = 2
        @test cond(matrix) < condition_tresh

        # LU-decomposable
        @test try lu(matrix)
            true
        catch
            false
        end

        # Solving
        rhs = Generator.generate_rhs_vector(matrix)
        x = matrix \ rhs
        @test !any(value-> value < 1-1e-8 || value > 1+1e-8, x) # TODO: replace hardcoded solution by variable solution

        # Random seed
        @test matrix == Generator.generate_matrix(settings) # Reproducability

        new_settings = Generator.Settings(
            dimension = settings.dimension,
            density = settings.density,
            magnitude_off = settings.magnitude_off,
            delta = settings.delta,
            seed = settings.seed + 1
        )

        @test matrix != Generator.generate_matrix(new_settings)

    end
end # testset "CAMNAS"

