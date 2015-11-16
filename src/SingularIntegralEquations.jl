
__precompile__()
module SingularIntegralEquations
    using Base, ApproxFun

export cauchy, cauchyintegral, stieltjes, logkernel,
       stieltjesintegral, hilbert, pseudohilbert, pseudocauchy


import Base: values,getindex,setindex!,*,.*,+,.+,-,.-,==,<,<=,>,
                >=,./,/,.^,^,\,∪,transpose


import ApproxFun
import ApproxFun: bandinds,SpaceOperator,dotu,linedotu,eps2,
                  plan_transform,plan_itransform,transform,itransform,transform!,itransform!,
                  rangespace, domainspace, addentries!, BandedOperator, AnySpace,
                  canonicalspace, domain, space, Space, promotedomainspace, promoterangespace, AnyDomain, CalculusOperator,
                  SumSpace,PiecewiseSpace, interlace,Multiplication,ArraySpace,DiagonalArrayOperator,
                  BandedMatrix,bazeros,ChebyshevDirichlet,PolynomialSpace,AbstractProductSpace,evaluate,order,
                  RealBasis,ComplexBasis,AnyBasis,UnsetSpace,ReImSpace,ReImOperator,BivariateFun,linesum,complexlength,
                  Fun, ProductFun, LowRankFun, mappoint, PeriodicLineSpace, PeriodicLineDirichlet,Recurrence, FiniteFunctional,
                  real, UnivariateSpace, setdomain, eps, choosedomainspace, isapproxinteger, BlockOperator,
                  ConstantSpace,ReOperator,DirectSumSpace,TupleSpace, AlmostBandedOperator, ZeroSpace,
                  DiagonalInterlaceOperator, LowRankPertOperator



# we don't override for Bool and Function to make overriding below easier
# TODO: change when cauchy(f,z,s) calls cauchy(f.coefficients,space(f),z,s)
function stieltjes(f,z,s)
    if isa(s,Bool)
        error("Override cauchy for "*string(typeof(f)))
    end

    @assert isa(s,Function)

    stieltjes(f,z,s==+)
end

hilbert(f)=Hilbert()*f
hilbert(f,z)=hilbert(f)(z)

#TODO: stieltjes -> offhilbert

cauchy(f,z,s...)=stieltjes(f,z,s...)*(im/(2π))
cauchyintegral(u,z,s...)=stieltjesintegral(u,z,s...)*(im/(2π))


include("LinearAlgebra/LinearAlgebra.jl")
include("Operators/Operators.jl")
include("FundamentalSolutions/FundamentalSolutions.jl")

include("stieltjesmoment.jl")

include("circlecauchy.jl")
include("intervalcauchy.jl")
include("singfuncauchy.jl")

include("vectorcauchy.jl")

include("GreensFun/GreensFun.jl")

include("periodicline.jl")
include("arc.jl")
include("curve.jl")
include("asymptotics.jl")

include("fractals.jl")
include("clustertree.jl")

if isdir(Pkg.dir("TikzGraphs"))
    include("introspect.jl")
end

include("Extras/Extras.jl")

end #module
