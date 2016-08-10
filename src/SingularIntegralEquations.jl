
__precompile__()
module SingularIntegralEquations
    using Base, Compat, BandedMatrices, ApproxFun, DualNumbers

export cauchy, cauchyintegral, stieltjes, logkernel,
       stieltjesintegral, hilbert, pseudohilbert, pseudocauchy,
       stieltjesjacobimoment, logjacobimoment


import Base: values,getindex,setindex!,*,.*,+,.+,-,.-,==,<,<=,>,
                >=,./,/,.^,^,\,∪,transpose


import BandedMatrices: bzeros

import Compat: view

import ApproxFun
import ApproxFun: bandinds,SpaceOperator,bilinearform,linebilinearform,dotu,
                  plan_transform,plan_itransform,transform,itransform,transform!,itransform!,
                  rangespace, domainspace, addentries!, promotespaces, InterlaceOperator,
                  canonicalspace, domain, space, Space, promotedomainspace, promoterangespace, AnyDomain, CalculusOperator,
                  SumSpace,PiecewiseSpace, interlace,Multiplication,ArraySpace,DiagonalArrayOperator,
                  BandedMatrix,ChebyshevDirichlet,PolynomialSpace,AbstractProductSpace,evaluate,order,
                  RealBasis,ComplexBasis,AnyBasis,UnsetSpace,BivariateFun,linesum,complexlength,
                  Fun, ProductFun, LowRankFun, mappoint, JacobiZ,
                  real, UnivariateSpace, RealUnivariateSpace, setdomain, eps, choosedomainspace, isapproxinteger, BlockOperator,
                  ConstantSpace,ReOperator,DirectSumSpace,TupleSpace, ZeroSpace,
                  DiagonalInterlaceOperator, LowRankPertOperator, LaurentDirichlet, setcanonicaldomain,
                  IntervalCurve,PeriodicCurve, reverseorientation, op_eltype, @wrapper, mobius,
                  defaultgetindex, WeightSpace, pochhammer, spacescompatible, ∞, LowRankMatrix


# we don't override for Bool and Function to make overriding below easier
# TODO: change when cauchy(f,z,s) calls cauchy(f.coefficients,space(f),z,s)

for OP in (:stieltjes,:stieltjesintegral,:pseudostieltjes)
    @eval begin
        $OP(f::Fun)=$OP(space(f),coefficients(f))
        $OP(f::Fun,z,s...)=$OP(space(f),coefficients(f),z,s...)
        $OP(f::Fun,z,s::Function)=$OP(f,z,s==+)
    end
end

hilbert(f)=Hilbert()*f
hilbert(S,f,z)=hilbert(Fun(f,S))(z)
hilbert(f::Fun,z)=hilbert(space(f),coefficients(f),z)

logkernel(f::Fun,z)=logkernel(space(f),coefficients(f),z)





cauchy(f...)=stieltjes(f...)*(im/(2π))
pseudocauchy(f...)=pseudostieltjes(f...)*(im/(2π))
cauchyintegral(u...)=stieltjesintegral(u...)*(im/(2π))


include("LinearAlgebra/LinearAlgebra.jl")
include("Operators/Operators.jl")
include("FundamentalSolutions/FundamentalSolutions.jl")
include("HypergeometricFunctions/HypergeometricFunctions.jl")

include("JacobiQWeight.jl")
include("JacobiQ.jl")
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
