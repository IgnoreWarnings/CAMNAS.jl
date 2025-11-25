
begin # Initialization
    ##############################################################
    ## Use this varibale to define the size of the input files ##
    ##############################################################
    const inputType = "generated" # small, medium, big, generated
    ##############################################################

    @assert inputType in ["small", "medium", "big", "generated"]
    ENV["JULIA_DEBUG"] = "CAMNAS" # Enable debug output
    ENV["JL_MNA_RUNTIME_SWITCH"] = "true" # Enable runtime switch
    ENV["JL_MNA_PRINT_ACCELERATOR"] = "true" # Enable printing accelerator in each solve steps
    push!(LOAD_PATH, pwd())
    #push!(LOAD_PATH, "$(pwd())/accelerators")
    @info LOAD_PATH
    using Pkg
    Pkg.activate(LOAD_PATH[4])
    Pkg.status()

    using CAMNAS
    using Profile

    include("Utils.jl")

    if inputType == "generated"
        include("Generator.jl")

        # Generate test matrix
        generator_settings = Generator.Settings(dimension=3, density=0.01)
        matrix = Generator.generate_matrix(generator_settings)

        # matrix to file
        csr_matrix = Utils.to_zerobased_csr(matrix)
        Generator.matrix_to_file(csr_matrix)

        # rhs to file
        rhs_vector = Generator.generate_rhs_vector(matrix) # assign directly
        Generator.rhs_to_file(rhs_vector)
    end

    GC.enable(false) # We cannot be sure that system_matrix is garbage collected before the pointer is passed...
    system_matrix = Utils.read_input(Utils.ArrayPath("$(@__DIR__)/system_matrix_$inputType.txt"))
    system_matrix_ptr = pointer_from_objref(system_matrix)
    rhs_vector = Utils.read_input(Utils.VectorPath("$(@__DIR__)/rhs_$inputType.txt"))
    lhs_vector = zeros(Float64, length(rhs_vector))
    rhs_reset = ones(Float64, length(rhs_vector))

    init(Base.unsafe_convert(Ptr{dpsim_csr_matrix}, system_matrix_ptr))
    GC.enable(true)
end # end Initialization

begin # Decomposition step
    GC.enable(false) # We cannot be sure that system_matrix is garbage collected before the pointer is passed...
    system_matrix = Utils.read_input(Utils.ArrayPath("$(@__DIR__)/system_matrix_$inputType.txt"))
    system_matrix_ptr = pointer_from_objref(system_matrix)
    rhs_vector = read_input(VectorPath("$(@__DIR__)/rhs_$inputSize.txt"))
    lhs_vector = zeros(Float64, length(rhs_vector))
    rhs_reset = ones(Float64, length(rhs_vector))


    @time decomp(Base.unsafe_convert(Ptr{dpsim_csr_matrix}, system_matrix_ptr))
    GC.enable(true)
end # end Decomposition

begin # Solving step 
    @time solve(Base.unsafe_convert(Ptr{Cdouble}, rhs_reset), Base.unsafe_convert(Ptr{Cdouble}, lhs_vector))
end # end Solving

begin # Cleanup step
    cleanup()
end # end Cleanup

begin # Benchmark performance test
    include("Benchmark.jl")
    include("Generator.jl")
    include("Utils.jl")

    # Matrix settings
    generator_settings = []
    dimensions = collect(500:500:500)
    densities = collect(0.01:0.01:0.01)

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

    for generator_setting in generator_settings
        # Generate Test Matrixes and RHS vectors
        matrix = Generator.generate_matrix(generator_setting)
        csr_matrix = Utils.to_zerobased_csr(matrix)
        rhs_vector = Generator.generate_rhs_vector(matrix)

        # Save matrix and rhs to files
        benchmarkPath = "testBenchmark"

        matrix_path = "$benchmarkPath/system_matrix_($(generator_setting.dimension))_($(generator_setting.density)).txt"
        Generator.matrix_to_file(csr_matrix, matrix_path=matrix_path)

        rhs_path = "$benchmarkPath/rhs_($(generator_setting.dimension))_($(generator_setting.density)).txt"
        Generator.rhs_to_file(rhs_vector, rhs_path=rhs_path)

        # Benchmark system performance
        accelerators = ["Tesla P40(1)"]
        for accelerator in accelerators
            @info "Running benchmark ... "

            # Configure accelerator
            # CAMNAS.accelerators_vector
            CAMNAS.update_varDict!(
                Dict(
                    "allow_strategies" => true,
                    "specific_accelerator_strategy" => true,
                    "specific_accelerator" => accelerator))

            #result = Benchmark.benchmark(matrix, rhs_vector)
            #strategy = "CAMNAS.current_accelerator"
            #Benchmark.save_benchmark("$benchmarkPath/test.csv", result, strategy, matrix_path) # TODO: save strat/env
            @info "Done running Benchmark."
        end
    end
end