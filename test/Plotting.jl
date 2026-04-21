module Plotting

using CSV
using DataFrames
using PlotlyJS
using Statistics

include("Utils.jl")

function plot_gpu(csv_path::String)
    df = CSV.read(csv_path, DataFrame)

    # Allow Dicts
    df.metrics = convert(Vector{Any}, df.metrics)
    df.strategy = convert(Vector{Any}, df.strategy)

    # Parse Dict strings
    df.metrics .= eval.(Meta.parse.(df.metrics))
    df.strategy .= eval.(Meta.parse.(df.strategy))

    traces = Vector{PlotlyBase.AbstractTrace}()
    strategy_groups = groupby(df, :strategy)

    for (i, strategy_group) in enumerate(strategy_groups)
        strat = strategy_group[1, :strategy]
        accelerator_name = string(strat["specific_accelerator"])

        matrix_labels = String[]
        host2dev_times = Float64[]
        solve_times = Float64[]
        dev2host_times = Float64[]

        for benchmark in eachrow(strategy_group)
            # Fast dimension read row number 
            open(benchmark.matrix_path, "r") do io
                for (i, line) in enumerate(eachline(io))
                    if i == 4 #Row number 
                        push!(matrix_labels, string(line))
                        break
                    end
                end
            end

            push!(host2dev_times, Float64(benchmark.metrics["host2dev"]))
            push!(solve_times,   Float64(benchmark.metrics["solve"]))
            push!(dev2host_times,Float64(benchmark.metrics["dev2host"]))
        end

        push!(traces,
            bar(
                x=matrix_labels,
                y=host2dev_times,
                name="$accelerator_name host2dev",
                legendgroup=accelerator_name,
                offsetgroup=string(i),
                alignmentgroup = "all"
            )
        )

        push!(traces,
            bar(
                x=matrix_labels,
                y=solve_times,
                name="$accelerator_name solve",
                legendgroup=accelerator_name,
                offsetgroup=string(i),
                alignmentgroup = "all"
            )
        )

        push!(traces,
            bar(
                x=matrix_labels,
                y=dev2host_times,
                name="$accelerator_name dev2host",
                legendgroup=accelerator_name,
                offsetgroup=string(i),
                alignmentgroup = "all"
            )
        )
    end

    layout = Layout(
        title="GPU Benchmark Metrics",
        barmode="stack",
        xaxis_title="Matrix Dimension",
        yaxis_title="Time (s)",
        #yaxis_type="log2"
    )

    plt = plot(traces, layout)
    display(plt)

end

function read_dimension(matrix_path)
    # Fast dimension read row number
    open(matrix_path, "r") do io
        for (i, line) in enumerate(eachline(io))
            if i == 4 #Row number 
                return parse(Int, string(line))
            end
        end
    end
end

"""
    function plot_metric(csv_paths, metric::String)

# Arguments
- `csv_paths`: The CSVs to be plotted.
- `metric::String`: The metric to plot.

# Example
```julia
plot_metric(["testBenchmark/ghost_cpu/benchmark.csv", 
                "testBenchmark/ghost_gpu/benchmark.csv",
                "testBenchmark/grace/benchmark.csv",
            ], "solve")
```
"""
function plot_metric(csv_paths, metric::String)
    # Join CSVs
    dfs = [CSV.read(csv_path, DataFrame) for csv_path in csv_paths]
    df = vcat(dfs...)

    # Parse Dict strings
    df.metrics = convert(Vector{Any}, df.metrics)
    df.metrics .= eval.(Meta.parse.(df.metrics))
    df.strategy = convert(Vector{Any}, df.strategy)
    df.strategy .= eval.(Meta.parse.(df.strategy))

    plot_metric(df, metric)
end

function plot_metric(df::DataFrame, metric::String)
    # Store points per accelerator
    data = Dict{String, Tuple{Vector{Int}, Vector{Float64}}}()

    matrix_groups = groupby(df, :matrix_path)
    for matrix_group in matrix_groups
        # Dimension
        matrix_path = matrix_group[1, :matrix_path]
        dimension = read_dimension(matrix_path)

        strategy_groups = groupby(matrix_group, :strategy)
        for strategy_group in strategy_groups
            # Strategy
            strat = strategy_group[1, :strategy]
            accelerator_name = string(strat["specific_accelerator"])

            # Average
            solve_average = mean(Float64(run.metrics[metric]) for run in eachrow(strategy_group))

            # Initialize if needed
            if !haskey(data, accelerator_name)
                data[accelerator_name] = (Int[], Float64[])
            end

            # Add to points
            push!(data[accelerator_name][1], dimension)
            push!(data[accelerator_name][2], solve_average)
        end
    end

    # Create bar traces
    traces = PlotlyJS.GenericTrace[]

    for (acc, (xs, ys)) in data
        push!(traces, bar(
            name = acc,
            x = xs,
            y = ys
        ))
    end

    layout = Layout(
        title="$metric, avg. of N runs per dimension",
        # barmode="stack",
        xaxis_title="Matrix Dimension",
        yaxis_title="Time (s)",
    )

    plt = plot(traces, layout)
    display(plt)

    savefig(plt, "benchmark_plot.html")
end

begin
    plot_metric(["testBenchmark/ghost_cpu/benchmark.csv", 
                "testBenchmark/ghost_gpu/benchmark.csv",
                "testBenchmark/grace/benchmark.csv",
            ], "solve")
end

# function plot_metric_vs_dimension(csv_path::String)
#     # Read CSV
#     df = CSV.read(csv_path, DataFrame)

#     # CSV structure
#     # decomp_elapses,solve_elapses,strategy(Dict),matrix_path

#     # Convert strategy column to Any so it can hold Dicts
#     df.strategy = convert(Vector{Any}, df.strategy)

#     # Convert all stringified Dicts into real Dicts
#     df.strategy .= eval.(Meta.parse.(df.strategy))

#     traces = Vector{PlotlyBase.AbstractTrace}()
#     strategy_groups = groupby(df, :strategy)
#     for strategy_group in strategy_groups
#         solve_times = []
#         dimensions = []
#         densities = []
#         for benchmark in eachrow(strategy_group)
#             # Load Matrix
#             matrix = Utils.read_input(Utils.ArrayPath(benchmark.matrix_path))
#             push!(dimensions, matrix.row_number)

#             density = matrix.nnz / (matrix.row_number^2)
#             #print(density)
#             push!(densities, density)

#             push!(solve_times, benchmark.solve_elapses)

#         end

#         trace = scatter3d(
#             x=densities,
#             y=dimensions,
#             z=solve_times,
#             mode="markers",
#             marker=attr(size=8),
#             name=strategy_group[1, :].strategy["specific_accelerator"],
#         )

#         push!(traces, trace)

#     end

#     layout = Layout(
#         title = "Solve Time vs Matrix Dimension",
#         scene=attr(
#             xaxis=attr(title="Density"),
#             yaxis=attr(title="Matrix Dimension"),
#             zaxis=attr(title="Solve Time")
#         ),
#         hovermode="closest",
#         type="scatter3d",
#         #scattermode="group",
#         scattergap=0.5
#     )

#     plt = plot(traces, layout)

#     display(plt)

#     #return plt
# end

# begin
#     plot_metric_vs_dimension("testBenchmark/run_1/benchmark.csv")
# end

end
