module Plot

using Plotly
using CSV, DataFrames


function trace()
    scatter(
        x=filtered.density,
        y=filtered.decomp_elapses, 
        mode="lines", 
        name="CAMNAS.NoAccelerator()"
    )
end

function plot(csv_path)
    csv_path = "$(@__DIR__)/../benchmark"
    csv = CSV.read("$csv_path", DataFrame)

    for row in csv
    filtered = filter(row -> row.accelerator == "CAMNAS.NoAccelerator()" && row.dimension == dimension, csv)

    PlotlyJS.display(plot([cpu_trace, gpu_trace], Layout(title="Solvestep of $dimension",
                                    xaxis=attr(title="Density"),
                                    yaxis=attr(title="Time"))))
end

end