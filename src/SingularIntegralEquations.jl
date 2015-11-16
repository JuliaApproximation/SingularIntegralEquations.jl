
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

for OP in (:stieltjes,:stieltjesintegral)
    @eval begin
        $OP(f::Fun,z,s...)=$OP(space(f),coefficients(f),z,s...)
        $OP(f::Fun,z,s::Function)=$OP(f,z,s==+)
    end
end

hilbert(f)=Hilbert()*f
hilbert(S,f,z)=hilbert(Fun(f,S))(z)
hilbert(f::Fun,z)=hilbert(space(f),coefficients(f),z)

logkernel(f::Fun,z)=logkernel(space(f),coefficients(f),z)





cauchy(f...)=stieltjes(f...)*(im/(2π))
cauchyintegral(u...)=stieltjesintegral(u...)*(im/(2π))


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
