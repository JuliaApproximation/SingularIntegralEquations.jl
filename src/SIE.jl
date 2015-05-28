module SIE
    using Base, ApproxFun

export cauchy, cauchyintegral, stieltjes, logkernel,stieltjesintegral


import ApproxFun
import ApproxFun: bandinds,CurveSpace,transform,itransform,transform!,itransform!,SpaceOperator,
                  rangespace, domainspace, addentries!, BandedOperator, AnySpace,
                  canonicalspace, domain, promotedomainspace, AnyDomain, CalculusOperator,
                  SumSpace,PiecewiseSpace, interlace,Multiplication,ArraySpace,DiagonalArrayOperator,
                  BandedMatrix,bazeros,ChebyshevDirichlet,PolynomialSpace,AbstractProductSpace,evaluate,order,
                  RealBasis,ComplexBasis,AnyBasis,UnsetSpace,ReImSpace,ReImOperator,BivariateFun,linesum,complexlength,
                  ProductFun, LowRankFun, mappoint, PeriodicLineSpace, PeriodicLineDirichlet,Recurrence, CompactFunctional,
                  real,UnivariateSpace, setdomain

function cauchy(s,f,z)
    if isa(s,Bool)
        error("Override cauchy for "*string(typeof(f)))
    end

    @assert abs(s) == 1

    cauchy(s==1,f,z)
end

hilbert(f)=Hilbert()*f
hilbert(f,z)=hilbert(f)[z]

#TODO: cauchy ->stieljtjes
#TODO: stieltjes -> offhilbert
stieltjes(s,f,z)=-2π*im*cauchy(s,f,z)
stieltjes(f,z)=-2π*im*cauchy(f,z)
cauchyintegral(u,z)=im/(2π)*stieltjesintegral(u,z)

# Fractal set of Intervals. α is width, n is number of levels

export cantor

function cantor{T}(d::Interval{T},n::Int,α::Number)
    a,b = d.a,d.b
    if n == 0
        return d
    else
        d = Interval{T}(zero(T),one(T))
        C = d/α ∪ (α-1+d)/α
        for k=2:n
            C = C/α ∪ (α-1+C)/α
        end
        return a+(b-a)*C
    end
end

include("Hilbert.jl")
include("OffHilbert.jl")

include("HilbertFunctions.jl")

include("circlecauchy.jl")
include("intervalcauchy.jl")
include("singfuncauchy.jl")

include("vectorcauchy.jl")

include("./GreensFun/GreensFun.jl")

include("periodicline.jl")
include("arc.jl")

include("asymptotics.jl")

if isdir(Pkg.dir("TikzGraphs"))
    include("introspect.jl")
end


end #module


