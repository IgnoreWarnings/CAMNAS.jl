
begin # Initialization
    ##############################################################
    ## Use this varibale to define the size of the input files ##
    ##############################################################
    const inputType = "generated" # small, medium, big, generated
    ##############################################################

    @assert inputType in ["small", "medium", "big", "generated"]
    ENV["JULIA_DEBUG"] = "" # Enable debug output
    ENV["JL_MNA_RUNTIME_SWITCH"] = "true" # Enable runtime switch
    ENV["JL_MNA_PRINT_ACCELERATOR"] = "false" # Enable printing accelerator in each solve steps

    # FORCE Accelerator
    ENV["JL_MNA_RUNTIME_SWITCH"] = "false"
    ENV["JL_MNA_SPECIFIC_ACCELERATOR_STRATEGY"] = "true"
    ENV["JL_MNA_SPECIFIC_ACCELERATOR"] = "NVIDIA GH200 144G HBM3e(0)"

    push!(LOAD_PATH, pwd())
    #push!(LOAD_PATH, "$(pwd())/accelerators")
    @info LOAD_PATH
    using Pkg
    Pkg.activate(LOAD_PATH[4]*"/test")
    Pkg.status()

    using CAMNAS
    using Profile

    include("Utils.jl")

    if inputType == "generated"
        include("Generator.jl")

        # Generate test matrix
        generator_settings = Generator.Settings(dimension=300, density=0.1)
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

# begin # Decomposition step
#     GC.enable(false) # We cannot be sure that system_matrix is garbage collected before the pointer is passed...
#     system_matrix = Utils.read_input(Utils.ArrayPath("$(@__DIR__)/system_matrix_$inputType.txt"))
#     system_matrix_ptr = pointer_from_objref(system_matrix)
#     rhs_vector = Utils.read_input(Utils.VectorPath("$(@__DIR__)/rhs_$inputType.txt"))
#     lhs_vector = zeros(Float64, length(rhs_vector))
#     rhs_reset = ones(Float64, length(rhs_vector))

    
#     @time decomp(Base.unsafe_convert(Ptr{dpsim_csr_matrix}, system_matrix_ptr))
#     GC.enable(true)
# end # end Decomposition

# begin # Solving step 
#     @time solve(Base.unsafe_convert(Ptr{Cdouble}, rhs_reset), Base.unsafe_convert(Ptr{Cdouble}, lhs_vector))
# end # end Solving

# begin # Cleanup step
#     cleanup()
# end # end Cleanup

begin # Benchmark performance test
    include("Benchmark.jl")
    include("Generator.jl")
    include("Utils.jl")
    include("MatrixValidator.jl")

    function next_run_folder(base_dir="benchmark/")
        runs = filter(name -> occursin(r"^run_\d+$", name), readdir(base_dir))

        if isempty(runs)
            return joinpath(base_dir, "run_1")
        end

        nums = parse.(Int, replace.(runs, r"^run_" => ""))
        next_num = maximum(nums) + 1

        return joinpath(base_dir, "run_$(next_num)")
    end

    benchmarkPath = next_run_folder()

    function build_generator_settings()
        # Matrix settings
        generator_settings = []
        dimensions = collect(200:200:20000)
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

    function save_input(matrix)
        # Save matrix file
        csr_matrix = Utils.to_zerobased_csr(matrix)
        matrix_path = "$benchmarkPath/system_matrix_($(size(matrix, 1)))_($(MatrixValidator.density(matrix))).txt"
        Generator.matrix_to_file(csr_matrix, matrix_path=matrix_path)

        return matrix_path
    end

    function prepare_strategies()
        accelerators = ["NVIDIA GH200 144G HBM3e(0)"] #"Tesla P40(2)", "NVIDIA GH200 144G HBM3e(0)", "cpu"]

        strategies = []
        for accelerator in accelerators
            push!(strategies, Dict("allow_strategies" => true, "specific_accelerator_strategy" => true,"specific_accelerator" => accelerator))
        end

        return strategies
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
    generator_settings_vector = build_generator_settings()
    for generator_settings in generator_settings_vector
        matrix = Generator.generate_matrix(generator_settings)
        matrix_path = save_input(matrix)

        # Calculate decomposition, store state in CAMNAS
        GC.enable(false)
        dpsim_matrix = Utils.julia_to_dpsim(matrix)
        system_matrix_ptr = pointer_from_objref(dpsim_matrix)
        ptr = Base.unsafe_convert(Ptr{dpsim_csr_matrix}, system_matrix_ptr)
        decomp(ptr)
        
        strategies = prepare_strategies()
        for strategy in strategies
            await_config_update(strategy)

            RUNS = 10
            rhs_vectors = [ Generator.generate_rhs_vector(matrix; prefered_solution=fill(Float64(i), size(matrix, 1))) for i in  1:RUNS] #rand(size(matrix, 1)))
            for (i, rhs) in enumerate(rhs_vectors)
                print("Run $i of $(length(rhs_vectors))")
                metrics = Benchmark.benchmark(rhs)
                Benchmark.save_csv("$benchmarkPath/benchmark.csv", metrics, CAMNAS.varDict, matrix_path) # TODO: Add RHS and RESULT
                println(" completed.")
            end
        end

        GC.enable(true)
    end

end
