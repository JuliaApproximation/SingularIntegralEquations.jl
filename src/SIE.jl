module SIE
    using Base, ApproxFun

export cauchy, cauchyintegral
import ApproxFun
import ApproxFun: PeriodicDomain, bandinds, dirichlettransform, idirichlettransform!,
                  Curve,CurveSpace, OpenCurveSpace, ClosedCurveSpace,transform,SpaceOperator,
                  rangespace, domainspace, addentries!, BandedOperator, PeriodicSpace, AnySpace,
                  canonicalspace, domain, promotedomainspace, AnyDomain, CalculusOperator,
                  SumSpace,PiecewiseSpace, interlace,Multiplication,ArraySpace,DiagonalArrayOperator,
                  BandedMatrix,bazeros,
                  RealBasis,ComplexBasis,AnyBasis,UnsetSpace,ReImSpace,ReImOperator,
                  ProductFun, mappoint, PeriodicLineSpace, PeriodicLineDirichlet

function cauchy(s,f,z)
    if isa(s,Bool)
        error("Override cauchy for "*string(typeof(f)))
    end

    @assert abs(s) == 1

    cauchy(s==1,f,z)
end

hilbert(f)=Hilbert()*f
hilbert(f,z)=hilbert(f)[z]

stieljes(s,f,z)=-2π*im*cauchy(s,f,z)
stieljes(f,z)=-2π*im*cauchy(f,z)


include("Hilbert.jl")
include("Stieltjes.jl")

include("HilbertFunctions.jl")

include("circlecauchy.jl")
include("intervalcauchy.jl")
include("singfuncauchy.jl")

include("vectorcauchy.jl")

include("KernelFun.jl")

include("periodicline.jl")

if isdir(Pkg.dir("TikzGraphs"))
    include("introspect.jl")
end


end #module


