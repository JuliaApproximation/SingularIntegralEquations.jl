include("Hilbert.jl")
include("OffHilbert.jl")
include("hierarchicalsolve.jl")


for OP in (:domainspace,:rangespace)
    @eval $OP(H::HierarchicalMatrix)=PiecewiseSpace(map($OP,H.diagonaldata))
end
