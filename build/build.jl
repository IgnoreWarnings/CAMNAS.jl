# build.jl


import PackageCompiler

using Pkg
using TOML
using Logging
using Test

const build_dir = @__DIR__
const target_dir = ARGS[1]

project = Base.active_project()
project_dir = dirname(project)

function require_without_errors(package_name)
    logger = TestLogger(min_level=Logging.Error)

    with_logger(logger) do
        Base.require(Main, Symbol(package_name))
    end

    if !isempty(logger.logs)
        error("Loading $package_name produced errors")
    end
end

function add_weakdeps_for_compilation()
    toml = TOML.parsefile(project)
    weakdeps = get(toml, "weakdeps", Dict())

    # Create a temporary environment to install the weak dependencies if possible
    Pkg.activate(; temp = true)

    addable = String[]

    for (package_name, _) in weakdeps
        @info("Testing host for weak dependency $package_name support...")

        try
            Pkg.add(package_name)

            # Check for error messages
            require_without_errors(package_name)

            # TODO: More checks like Cuda functional

            push!(addable, package_name)
        catch err
            @warn "Could not add weak dependency $package_name; reason: $err" exception=(err, catch_backtrace())
        finally
            # Remove the package from the temporary environment to avoid conflicts
            Pkg.rm(package_name)
        end
    end

    return addable
end

addable = add_weakdeps_for_compilation()

Pkg.add(addable)

#Pkg.precompile()

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