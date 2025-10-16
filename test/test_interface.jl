
begin # Initialization
    ##############################################################
    ## Use this varibale to define the size of the input files ##
    ##############################################################
    const inputValues = "generated" # small, medium, big, generated
    ##############################################################

    @assert inputValues in ["small", "medium", "big", "generated"]
    ENV["JULIA_DEBUG"] = "CAMNAS"#"" # Enable debug output
    # ENV["JL_MNA_RUNTIME_SWITCH"] = "true" # Enable runtime switch
    # ENV["JL_MNA_PRINT_ACCELERATOR"] = "true" # Enable printing accelerator in each solve steps
    push!(LOAD_PATH, pwd())
    #push!(LOAD_PATH, "$(pwd())/accelerators")
    @info LOAD_PATH
    using Pkg
    Pkg.activate(LOAD_PATH[4])
    Pkg.status()

    using CAMNAS
    using Profile

    struct ArrayPath path::String end
    struct VectorPath path::String end

    function read_input(path::ArrayPath)
        # Read system matrix from file
        system_matrix_strings = readlines(path.path)

        # Sanize strings
        system_matrix_strings = replace.(system_matrix_strings, r"[\[\],]" => "")

        # Convert system to dpsim_csr_matrix
        values = parse.(Float64, split(system_matrix_strings[1]))
        rowIndex = parse.(Cint, split(system_matrix_strings[2]))
        colIndex = parse.(Cint, split(system_matrix_strings[3]))

        system_matrix = CAMNAS.dpsim_csr_matrix(
            Base.unsafe_convert(Ptr{Cdouble}, values),
            Base.unsafe_convert(Ptr{Cint}, rowIndex),
            Base.unsafe_convert(Ptr{Cint}, colIndex),
            parse(Int32, system_matrix_strings[4]),
            parse(Int32, system_matrix_strings[5])
        )

        return system_matrix
    end

    function read_input(path::VectorPath)
        # Reard right hand side vector from file
        rhs_vector_strings = readlines(path.path)

        # Sanitize rhs strings and parse into Float64 vector
        rhs_vector_strings = replace.(rhs_vector_strings, r"[\[\],]" => "")
        rhs_vector = parse.(Float64, split(rhs_vector_strings[1]))
    end

    if inputValues == "generated"
        include("Generator.jl")
        dimension::UInt = 3
        matrix = Generator.generate_matrix(dimension)
        csr_matrix = Generator.to_csr(matrix)
        rhs_vector = Generator.generate_rhs_vector(matrix) # assign directly

        Generator.to_files(csr_matrix, rhs_vector)
    end

    system_matrix = read_input(ArrayPath("$(@__DIR__)/system_matrix_$inputValues.txt"))
    rhs_vector = read_input(VectorPath("$(@__DIR__)/rhs_$inputValues.txt"))

    GC.enable(false) # We cannot be sure that system_matrix is garbage collected before the pointer is passed...

    system_matrix_ptr = pointer_from_objref(system_matrix)
    @debug "deref system_matrix_ptr: $(Base.dereference(system_matrix_ptr))"

    lhs_vector = zeros(Float64, length(rhs_vector))
    rhs_reset = ones(Float64, length(rhs_vector))

    init(Base.unsafe_convert(Ptr{dpsim_csr_matrix}, system_matrix_ptr))
    GC.enable(true)
end # end Initialization

begin # Decomposition step
    GC.enable(false)
    system_matrix = read_input(ArrayPath("$(@__DIR__)/system_matrix_$inputValues.txt"))
    system_matrix_ptr = pointer_from_objref(system_matrix)
    @time decomp(Base.unsafe_convert(Ptr{dpsim_csr_matrix}, system_matrix_ptr))
    GC.enable(true)
end # end Decomposition

begin # Solving step 
    @time solve(Base.unsafe_convert(Ptr{Cdouble}, rhs_vector), Base.unsafe_convert(Ptr{Cdouble}, lhs_vector))
end # end Solving

begin # Cleanup step
    cleanup()
end # end Cleanup

begin # Benchmark
    include("Generator.jl")

    using BenchmarkTools
    using CSV, DataFrames

    # Benchmark parameters
    BenchmarkTools.DEFAULT_PARAMETERS.samples = 3
    accelerators = [CAMNAS.NoAccelerator(), CAMNAS.CUDAccelerator()]
    dimensions = UInt[1000:500:2000...]
    densities = Float64[0.01:0.01:0.3...]

    # Disable debug information during benchmark
    saved_debug_env = ENV["JULIA_DEBUG"]
    ENV["JULIA_DEBUG"] = "" # Disable debug output

    # Create folder for benchmark results
    benchmarkPath = "$(@__DIR__)/../benchmark"
    mkpath(benchmarkPath)
    
    for dimension in dimensions
        for density in densities
            # Generate Test Matrixes and RHS vectors
            print("Generating...")
            matrix = Generator.generate_matrix(dimension; density=density)
            csr_matrix = Generator.to_csr(matrix)
            rhs_vector = Generator.generate_rhs_vector(matrix)
            matrix_path = "$benchmarkPath/system_matrix_($dimension)_($density).txt"
            rhs_path = "$benchmarkPath/rhs_($dimension)_($density).txt"
            Generator.to_files(csr_matrix, rhs_vector; 
                                matrix_path=matrix_path,
                                rhs_path=rhs_path)
            println("Done.")

            for (i,accelerator) in enumerate(accelerators)
                # Set accelerator
                CAMNAS.set_accelerator!(accelerator)
                println("Benchmarking: $(CAMNAS.accelerator) on $dimension x $dimension with density=$density")

                system_matrix = read_input(ArrayPath(matrix_path))
                rhs_vector = read_input(VectorPath(rhs_path))

                GC.enable(false)
                
                # Decomposition
                system_matrix_ptr = pointer_from_objref(system_matrix)
                decomp_elapses = @belapsed decomp(Base.unsafe_convert(Ptr{dpsim_csr_matrix}, system_matrix_ptr))

                # Solving
                lhs_vector = zeros(Float64, length(rhs_vector))
                solve_elapses = @elapsed solve(Base.unsafe_convert(Ptr{Cdouble}, rhs_vector), Base.unsafe_convert(Ptr{Cdouble}, lhs_vector))
                
                GC.enable(true)

                data_frame = DataFrame(accelerator = [accelerator], dimension = [dimension], density = [density], decomp_elapses = [decomp_elapses], solve_elapses = [solve_elapses] )
                append = isfile("$benchmarkPath/data.csv") # with append no header is written
                CSV.write("$benchmarkPath/data.csv", data_frame; append=append)

                println(" Done.")
            end
        end

    end

    ENV["JULIA_DEBUG"] = saved_debug_env # Restore env setting
end

begin # Plot
    using Plotly
    using CSV, DataFrames

    dimension = 1500

    benchmarkPath = "$(@__DIR__)/../benchmark"
    csv = CSV.read("$benchmarkPath/data.csv", DataFrame)

    filtered = filter(row -> row.accelerator == "CAMNAS.NoAccelerator()" && row.dimension == dimension, csv)

    cpu_trace = scatter(x=filtered.density,
                        y=filtered.decomp_elapses, 
                        mode="lines", 
                        name="CAMNAS.NoAccelerator()")

    filtered = filter(row -> row.accelerator == "CAMNAS.CUDAccelerator()" && row.dimension == dimension, csv)

    gpu_trace = scatter(x=filtered.density,
                        y=filtered.decomp_elapses, 
                        mode="lines", 
                        name="CAMNAS.CUDAccelerator()")

    PlotlyJS.display(plot([cpu_trace, gpu_trace], Layout(title="Solvestep of $dimension",
                                    xaxis=attr(title="Density"),
                                    yaxis=attr(title="Time"))))
end
