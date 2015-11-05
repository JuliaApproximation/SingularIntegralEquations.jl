include("Hilbert.jl")
include("OffHilbert.jl")
include("LowRankIntegralOperator.jl")
include("HierarchicalDSF.jl")
include("HierarchicalOperator.jl")
include("hierarchicalsolve.jl")


for OP in (:domainspace,:rangespace)
    @eval $OP{S<:Operator,V<:AbstractLowRankOperator}(H::HierarchicalOperator{S,V})=PiecewiseSpace(map($OP,diagonaldata(H)))
end
