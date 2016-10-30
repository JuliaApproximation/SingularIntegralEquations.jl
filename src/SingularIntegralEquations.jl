
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
import ApproxFun: bandinds, blockbandinds, SpaceOperator, bilinearform, linebilinearform,dotu, blocklengths,
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

import DualNumbers: value

"""
`Directed` represents a number that is a limit from either left (s=true) or right (s=false)
For functions with branch cuts, it is assumed that the value is on the branch cut,
Therefore not requiring tolerances.  This will naturally give the analytic continuation.
"""
immutable Directed{s,T} <: Number
    x::T
    Directed(x::T) = new(x)
    Directed(x::Number) = new(T(x))
end


(::Type{Directed{s}}){s}(x) = Directed{s,eltype(x)}(x)

Base.convert{s,T}(::Type{Directed{s,T}},x::Directed{s}) = Directed{s,T}(T(x.x))
Base.convert{s,T}(::Type{Directed{s,T}},x::T) = Directed{s,T}(x)
Base.convert{s,T}(::Type{Directed{s,T}},x::Real) = Directed{s,T}(T(x))
Base.convert{s,T}(::Type{Directed{s,T}},x::Complex) = Directed{s,T}(T(x))

const ⁺ = Directed{true}(true)
const ⁻ = Directed{false}(true)

orientationsign(::Type{Directed{true}}) = 1
orientationsign(::Type{Directed{false}}) = -1
orientation{s}(::Type{Directed{s}}) = s
orientation{s}(::Directed{s}) = s
value(x::Directed) = x.x
value(x::Number) = x
value(x::Fun) = x
reverseorientation{s}(x::Directed{s}) = Directed{!s}(x.x)
reverseorientation(x::Number) = x


for OP in (:*,:+,:-,:/)
    @eval begin
        $OP{s}(a::Directed{s}) = Directed{s}($OP(a.x))
        $OP{s}(a::Directed{s},b::Directed{s}) = Directed{s}($OP(a.x,b.x))
        $OP{s}(a::Directed{s},b::Number) = Directed{s}($OP(a.x,b))
        $OP{s}(a::Number,b::Directed{s}) = Directed{s}($OP(a,b.x))
    end
end

real{s,T}(::Type{Directed{s,T}}) = real(T)

# abs, real and imag delete orientation.
for OP in (:(Base.isfinite),:(Base.isinf),:(Base.abs),:(Base.real),:(Base.imag),:(Base.angle))
    @eval $OP(a::Directed) = $OP(a.x)
end


# branchcuts of log, sqrt, etc. are oriented from (0,-∞)
Base.log(x::Directed{true}) = log(-x.x) - π*im
Base.log(x::Directed{false}) = log(-x.x) + π*im
Base.log1p(x::Directed) = log(1+x)
Base.sqrt(x::Directed{true}) = -im*sqrt(-x.x)
Base.sqrt(x::Directed{false}) = im*sqrt(-x.x)
^(x::Directed{true},a::Integer) = x.x^a
^(x::Directed{false},a::Integer) = x.x^a
^(x::Directed{true},a::Number) = exp(-a*π*im)*(-x.x)^a
^(x::Directed{false},a::Number) = exp(a*π*im)*(-x.x)^a




for OP in (:stieltjes,:stieltjesintegral,:pseudostieltjes)
    @eval $OP(f::Fun) = $OP(space(f),coefficients(f))
    @eval $OP(f::Fun,z) = $OP(space(f),coefficients(f),z)
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

function testsieoperators(S::Space)
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
    end
end


function testsieeval(S::Space;posdirection=im)
    p=ApproxFun.checkpoints(S)[1] # random point on contour
    x=Fun(domain(S))
    z=2.12312231+1.433453443534im # random point not on contour

    for k=1:5
        f=Fun([zeros(k-1);1],S)
        @test abs(sum(f/(z-x))-stieltjes(f,z)) ≤ 100eps()
        @test_approx_eq stieltjes(f,p*⁺) stieltjes(f,p+eps()*posdirection)
        @test_approx_eq stieltjes(f,p*⁻) stieltjes(f,p-eps()*posdirection)
        @test_approx_eq cauchy(f,p*⁺)-cauchy(f,p*⁻) f(p)
        @test_approx_eq im*(cauchy(f,p*⁺)+cauchy(f,p*⁻)) hilbert(f,p)

        @test abs(linesum(f*log(abs(x-z)))/π-logkernel(f,z)) ≤ 100eps()
        @test_approx_eq logkernel(f,p) logkernel(f,p+eps()*posdirection)
    end
end

function testsies(S::Space;posdirection=im)
    testsieoperators(S)
    testsieeval(S;posdirection=posdirection)
end

end #module
