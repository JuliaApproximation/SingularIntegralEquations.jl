module SIE
    using Base, ApproxFun

export Cauchy, cauchy, hilbert, hilbertinverse, cauchyintegral
import ApproxFun
import ApproxFun: PeriodicDomain, BandedShiftOperator, bandinds, dirichlettransform, idirichlettransform!, Curve,CurveSpace, OpenCurveSpace, ClosedCurveSpace,transform,SpaceOperator, rangespace, domainspace, addentries!, BandedOperator, PeriodicDomainSpace, AnySpace, canonicalspace, domain, promotedomainspace, AnyDomain, CalculusOperator,SumSpace,PiecewiseSpace


function cauchy(s::Integer,f,z)
    @assert abs(s) == 1
    
    cauchy(s==1,f,z)
end



include("circlecauchy.jl")
include("periodiclinecauchy.jl")
include("intervalcauchy.jl")
include("singfuncauchy.jl")

include("Hilbert.jl")

include("vectorcauchy.jl")

end #module


