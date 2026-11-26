# build.jl


import PackageCompiler

using Pkg
using TOML
using Logging

const build_dir = @__DIR__
const target_dir = ARGS[1]

const project = Base.active_project()
const project_dir = dirname(project)

function parse_backend_extensions(project)
    toml = TOML.parsefile(project)
    extensions = get(toml, "extensions", Dict())

    backends = Dict{String, Vector{String}}()

    for (extension, dependencies) in extensions
        packages = dependencies isa String ?
            [dependencies] : String.(dependencies)

        backends[extension] = packages
    end

    return backends
end

function extension_test_code(extensions)
    code = String[]

    push!(code, "using CAMNAS")

    for packages in values(extensions)
        for package in packages
            push!(code, "using $package")
        end
    end

    for extension in keys(extensions)
        push!(
            code,
            """
            @assert Base.get_extension(CAMNAS, :$extension) !== nothing
            """
        )
    end

    return join(code, "\n")
end


function probe_backend(extension, packages)
    mktempdir() do probe_dir
        try
            @info "Testing extension $extension" packages

            # Create isolated environment.
            Pkg.activate(probe_dir)

            # Add local CAMNAS package.
            Pkg.develop(path=project_dir)

            # Install all extension dependencies together.
            Pkg.add(packages)

            # Test in a fresh Julia process.
            test_code = """
                using CAMNAS
                $(join(["using $p" for p in packages], "\n"))

                @assert Base.get_extension(CAMNAS, :$extension) !== nothing
            """

            cmd = `$(Base.julia_cmd()) --startup-file=no --project=$probe_dir -e $test_code`

            run(cmd)

            @info "Extension $extension is available"
            return true

        catch err
            @warn "Extension $extension unavailable" packages exception=(
                err, catch_backtrace()
            )

            return false

        finally
            Pkg.activate(project_dir)
        end
    end
end

function probe_backends()
    extensions = parse_backend_extensions(project)

    available = String[]
    successful_extensions = Dict{String, Vector{String}}()

    for (extension, packages) in extensions
        if probe_backend(extension, packages)
            append!(available, packages)
            successful_extensions[extension] = packages
        end
    end

    return unique(available), successful_extensions
end

function test_combined_environment(extensions)
    isempty(extensions) && return true

    test_code = extension_test_code(extensions)

    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$project_dir -e $test_code`

    try
        run(cmd)
        @info "All detected extensions activated successfully"
        return true

    catch err
        @warn "Combined extension test failed" exception=(
            err, catch_backtrace()
        )
        return false
    end
end

function prepare_backends()
    available, successful_extensions = probe_backends()

    @info "Detected backend packages" available
    @info "Detected extensions" (collect(keys(successful_extensions)))

    # Install all compatible packages together.
    if !isempty(available)
        Pkg.activate(project_dir)
        Pkg.add(available)

        # Test the actual combined environment.
        if !test_combined_environment(successful_extensions)
            error("Combined backend environment failed validation")
        end
    end

    return successful_extensions
end


Pkg.activate(project_dir)

successful_extensions = prepare_backends()

@info "Backend preparation completed"
@info "Enabled extensions" (collect(keys(successful_extensions)))

@info "Creating CAMNAS solver library in $target_dir"
PackageCompiler.create_library("$(build_dir)/..", target_dir;
                                lib_name="camnasjl",
                                precompile_execution_file="$(@__DIR__)/precompile_statements.jl",
                                incremental=false,
                                filter_stdlibs=false,
                                include_lazy_artifacts=true,
                                header_files = ["$(@__DIR__)/camnasjl.h"],
                                force=true,
                                cpu_target="native"
                            )