
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
import ApproxFun: bandinds, blockbandinds, SpaceOperator, bilinearform, linebilinearform,dotu,
                  plan_transform,plan_itransform,transform,itransform,transform!,itransform!,
                  rangespace, domainspace, promotespaces, InterlaceOperator, coefficientmatrix,
                  canonicalspace, domain, space, Space, promotedomainspace, promoterangespace, AnyDomain, CalculusOperator,
                  ConcreteDefiniteIntegral, ConcreteDefiniteLineIntegral,
                  SumSpace,PiecewiseSpace, interlace,Multiplication,ArraySpace, diagonalarrayoperator,
                  BandedMatrix,ChebyshevDirichlet,PolynomialSpace,AbstractProductSpace,evaluate,order,
                  RealBasis,ComplexBasis,AnyBasis,UnsetSpace, MultivariateFun, BivariateFun,linesum,complexlength,
                  Fun, ProductFun, LowRankFun, mappoint, JacobiZ,
                  real, UnivariateSpace, RealUnivariateSpace, setdomain, eps, choosedomainspace, isapproxinteger, BlockOperator,
                  ConstantSpace,ReOperator,DirectSumSpace,TupleSpace, ZeroSpace,
                  LowRankPertOperator, LaurentDirichlet, setcanonicaldomain,
                  IntervalCurve,PeriodicCurve, reverseorientation, op_eltype, @wrapper, mobius,
                  defaultgetindex, WeightSpace, pochhammer, spacescompatible, ∞, LowRankMatrix, refactorsvd!

import ApproxFun: testbandedoperator

# we don't override for Bool and Function to make overriding below easier
# TODO: change when cauchy(f,z,s) calls cauchy(f.coefficients,space(f),z,s)

for OP in (:stieltjes,:stieltjesintegral,:pseudostieltjes)
    @eval begin
        $OP(f::Fun)=$OP(space(f),coefficients(f))
        $OP(f::Fun,z,s...)=$OP(space(f),coefficients(f),z,s...)
        $OP(f::Fun,z,s::Function)=$OP(f,z,s==+)
    end
end

hilbert(f) = Hilbert()*f
hilbert(S,f,z) = hilbert(Fun(f,S))(z)
hilbert(f::Fun,z) = hilbert(space(f),coefficients(f),z)

logkernel(f::Fun,z) = logkernel(space(f),coefficients(f),z)





cauchy(f...) = stieltjes(f...)*(im/(2π))
pseudocauchy(f...) = pseudostieltjes(f...)*(im/(2π))
cauchyintegral(u...) = stieltjesintegral(u...)*(im/(2π))


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


using Base.Test

function testsies(S::Space)
    testbandedoperator(SingularIntegral(S,0))
    testbandedoperator(SingularIntegral(S,1))
    testbandedoperator(Hilbert(S))
    p=ApproxFun.checkpoints(S)[1] # random point on contour
    x=Fun(domain(S))
    z=2.12312231+1.433453443534im # random point not on contour

    for k=1:5
        f=Fun([zeros(k-1);1],S)
        @test_approx_eq (SingularIntegral(S,0)*f)(p) logkernel(f,p)
        @test_approx_eq (Hilbert(S,1)*f)(p) hilbert(f,p)
        @test abs(linesum(f*log(abs(x-z)))/π-logkernel(f,z)) ≤ 100eps()
        @test abs(sum(f/(z-x))-stieltjes(f,z)) ≤ 100eps()
        @test_approx_eq cauchy(f,p,+)-cauchy(f,p,-) f(p)
        @test_approx_eq im*(cauchy(f,p,+)+cauchy(f,p,-)) hilbert(f,p)
    end
end

end #module
