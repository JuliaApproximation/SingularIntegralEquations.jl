module SIE
    using Base, ApproxFun

export cauchy, cauchyintegral
import ApproxFun
import ApproxFun: PeriodicDomain, bandinds, dirichlettransform, idirichlettransform!, Curve,CurveSpace, OpenCurveSpace, ClosedCurveSpace,transform,SpaceOperator, rangespace, domainspace, addentries!, BandedOperator, PeriodicSpace, AnySpace, canonicalspace, domain, promotedomainspace, AnyDomain, CalculusOperator,SumSpace,PiecewiseSpace, interlace,Multiplication,ArraySpace,DiagonalArrayOperator


function cauchy(s::Integer,f,z)
    @assert abs(s) == 1

    cauchy(s==1,f,z)
end


include("Hilbert.jl")
include("Cauchy.jl")

include("HilbertFunctions.jl")

include("circlecauchy.jl")
include("periodiclinecauchy.jl")
include("intervalcauchy.jl")
include("singfuncauchy.jl")

include("vectorcauchy.jl")

end #module


