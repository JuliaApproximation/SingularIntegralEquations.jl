include("plot.jl")
include("show.jl")
if isdir(Pkg.dir("DualNumbers"))
    include("normalderivative.jl")
end
