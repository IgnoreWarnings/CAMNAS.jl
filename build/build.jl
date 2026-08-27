# build.jl


import PackageCompiler

using Pkg
using TOML
using Logging
using Test

const build_dir = @__DIR__
const target_dir = ARGS[1]

# FIXME: Remove once the issue with PackageCompiler and Julia v1.11. is resolved.
# See: https://github.com/JuliaLang/PackageCompiler.jl/issues/990
delete!(ENV, "JULIA_NUM_THREADS")

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

    added = String[]

    for (package_name, _) in weakdeps
        println("Trying weak dependency: $package_name")

        try
            Pkg.add(package_name)

            # Check that the package can actually be loaded.
            require_without_errors(package_name)

            # TODO: More checks like Cuda functional

            push!(added, package_name)
        catch err
            @warn "Could not add weak dependency $package_name; removing it" exception=(err, catch_backtrace())

            # Make sure the failed package is not left in the environment.
            try
                @info "Removing $package_name"
                Pkg.rm(package_name)
            catch rmerr
                @warn "Could not remove failed weak dependency $package_name" exception=(rmerr, catch_backtrace())
            end
        end
    end

    return added
end

added = add_weakdeps_for_compilation()
println("The following packages where added sucessfully: $added")

println("Creating CAMNAS solver library in $target_dir")
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