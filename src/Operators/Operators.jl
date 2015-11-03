include("Hilbert.jl")
include("OffHilbert.jl")
include("LowRankIntegralOperator.jl")
include("hierarchicalsolve.jl")


for OP in (:domainspace,:rangespace)
    @eval $OP{S<:Union{Operator,HierarchicalMatrix},V<:AbstractLowRankOperator}(H::HierarchicalMatrix{S,V})=PiecewiseSpace(map($OP,diagonaldata(H)))
end
