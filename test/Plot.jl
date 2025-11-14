module Plot

using Plotly
using CSV, DataFrames

function plot()
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

end