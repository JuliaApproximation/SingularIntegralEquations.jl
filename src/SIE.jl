module SIE
    using Base, ApproxFun

export cauchy, cauchyintegral, stieltjes, logkernel


import ApproxFun
import ApproxFun: PeriodicDomain, bandinds, 
                  Curve,CurveSpace, OpenCurveSpace, ClosedCurveSpace,transform,SpaceOperator,
                  rangespace, domainspace, addentries!, BandedOperator, PeriodicSpace, AnySpace,
                  canonicalspace, domain, promotedomainspace, AnyDomain, CalculusOperator,
                  SumSpace,PiecewiseSpace, interlace,Multiplication,ArraySpace,DiagonalArrayOperator,
                  BandedMatrix,bazeros,ChebyshevDirichlet,
                  RealBasis,ComplexBasis,AnyBasis,UnsetSpace,ReImSpace,ReImOperator,
                  ProductFun, mappoint, PeriodicLineSpace, PeriodicLineDirichlet,JacobiRecurrence

function cauchy(s,f,z)
    if isa(s,Bool)
        error("Override cauchy for "*string(typeof(f)))
    end

    @assert abs(s) == 1

    cauchy(s==1,f,z)
end

hilbert(f)=Hilbert()*f
hilbert(f,z)=hilbert(f)[z]

stieltjes(s,f,z)=-2π*im*cauchy(s,f,z)
stieltjes(f,z)=-2π*im*cauchy(f,z)


include("Hilbert.jl")
include("Stieltjes.jl")

include("HilbertFunctions.jl")

include("circlecauchy.jl")
include("intervalcauchy.jl")
include("singfuncauchy.jl")

include("vectorcauchy.jl")

include("KernelFun.jl")

include("periodicline.jl")
include("arc.jl")

if isdir(Pkg.dir("TikzGraphs"))
    include("introspect.jl")
end


end #module


