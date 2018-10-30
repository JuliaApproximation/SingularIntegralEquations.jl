
__precompile__()
module SingularIntegralEquations
    using Base, BandedMatrices, ApproxFun, DualNumbers, RecipesBase, DomainSets,
            LinearAlgebra, Random, SpecialFunctions, LowRankApprox, InteractiveUtils

export cauchy, cauchyintegral, stieltjes, logkernel,
       stieltjesintegral, hilbert, pseudohilbert, pseudocauchy,
       stieltjesjacobimoment, logjacobimoment, singularintegral


import Base: values, getindex, setindex!, *, +, -, ==, <, <=, >,
                >=, /, ^, \, ∪, transpose, convert, Array, Vector, Matrix,
                AbstractVector, AbstractMatrix, AbstractArray

import Base.Broadcast: broadcasted, DefaultArrayStyle

import LinearAlgebra: ldiv!, mul!, rank, cond, qr

import ApproxFun
import DomainSets: UnionDomain

import ApproxFun: bandwidths, blockbandwidths, SpaceOperator, bilinearform, linebilinearform,dotu, blocklengths,
                  plan_transform,plan_itransform,transform,itransform,transform!,itransform!,
                  rangespace, domainspace, promotespaces, InterlaceOperator, coefficientmatrix,
                  canonicalspace, domain, space, Space, promotedomainspace, promoterangespace, AnyDomain, CalculusOperator,
                  ConcreteDefiniteIntegral, ConcreteDefiniteLineIntegral,
                  SumSpace,PiecewiseSpace, interlace,Multiplication, VectorSpace, ArraySpace,
                  BandedMatrix,ChebyshevDirichlet,PolynomialSpace,AbstractProductSpace,evaluate,order,
                  UnsetSpace, MultivariateFun, BivariateFun,linesum,complexlength,
                  Fun, ProductFun, LowRankFun, mappoint, JacobiZ, prectype,
                  real, UnivariateSpace, RealUnivariateSpace, setdomain, eps, choosedomainspace, isapproxinteger,
                  ConstantSpace,ReOperator,DirectSumSpace, ArraySpace, ZeroSpace,
                  LowRankPertOperator, LaurentDirichlet, setcanonicaldomain, SubSpace,
                  IntervalCurve, PeriodicCurve, reverseorientation, @wrapper, mobius,
                  defaultgetindex, WeightSpace, pochhammer, spacescompatible, ∞, LowRankMatrix, SubOperator,
                  Block, BlockBandedMatrix, BandedBlockBandedMatrix, DFunction, Infinity,
                  component, ncomponents, factor, nfactors, components, factors, rangetype,
                  VFun, Point, dynamic, pieces, npieces, piece, cfstype, isreal, IntervalOrSegmentDomain,
                  IntervalOrSegment

import ApproxFun: testbandedoperator

import DualNumbers: dual

import LowRankApprox: refactorsvd!

export ⁺, ⁻

"""
`Directed` represents a number that is a limit from either left (s=true) or right (s=false)
For functions with branch cuts, it is assumed that the value is on the branch cut,
Therefore not requiring tolerances.  This will naturally give the analytic continuation.
"""
struct Directed{s,T} <: Number
    x::T
    Directed{s,T}(x::T) where {s,T} = new{s,T}(x)
    Directed{s,T}(x::Number) where {s,T} = new{s,T}(T(x))
end


Directed{s}(x) where {s} = Directed{s,eltype(x)}(x)

convert(::Type{Directed{s,T}},x::Directed{s}) where {s,T} = Directed{s,T}(T(x.x))
convert(::Type{Directed{s,T}},x::T) where {s,T} = Directed{s,T}(x)
convert(::Type{Directed{s,T}},x::Real) where {s,T} = Directed{s,T}(T(x))
convert(::Type{Directed{s,T}},x::Complex) where {s,T} = Directed{s,T}(T(x))

const ⁺ = Directed{true}(true)
const ⁻ = Directed{false}(true)

orientationsign(::Type{Directed{true}}) = 1
orientationsign(::Type{Directed{false}}) = -1
orientation(::Type{Directed{s}}) where {s} = s
orientation(::Directed{s}) where {s} = s

# removes direction from a number
undirected(x::Number) = x
undirected(x::Fun) = x
undirected(x::Directed) = undirected(x.x)  # x might also have directeion
reverseorientation(x::Number) = x
reverseorientation(x::Directed{s}) where {s} = Directed{!s}(reverseorientation(x.x))


for OP in (:*,:+,:-,:/)
    @eval begin
        $OP(a::Directed{s}) where {s} = Directed{s}($OP(a.x))
        $OP(a::Directed{s},b::Directed{s}) where {s} = Directed{s}($OP(a.x,b.x))
        $OP(a::Directed{s},b::Number) where {s} = Directed{s}($OP(a.x,b))
        $OP(a::Number,b::Directed{s}) where {s} = Directed{s}($OP(a,b.x))
    end
end

real(::Type{Directed{s,T}}) where {s,T} = real(T)

# abs, real and imag delete orientation.
for OP in (:(Base.isfinite),:(Base.isinf),:(Base.abs),:(Base.real),:(Base.imag),:(Base.angle))
    @eval $OP(a::Directed) = $OP(a.x)
end


# branchcuts of log, sqrt, etc. are oriented from (0,-∞)
Base.log(x::Directed{true}) = log(-x.x) - π*im
Base.log(x::Directed{false}) = log(-x.x) + π*im
Base.log1p(x::Directed) = log(1+x)
Base.sqrt(x::Directed{true}) = real(x.x) ≥ 0 ? sqrt(complex(x.x)) : -im*sqrt(-x.x)
Base.sqrt(x::Directed{false}) = real(x.x) ≥ 0 ? sqrt(complex(x.x)) : im*sqrt(-x.x)
^(x::Directed{true},a::Integer) = x.x^a
^(x::Directed{false},a::Integer) = x.x^a
^(x::Directed{true},a::Number) = exp(-a*π*im)*(-x.x)^a
^(x::Directed{false},a::Number) = exp(a*π*im)*(-x.x)^a


dual(a::Directed{s},b) where {s} = Directed{s}(dual(undirected(a),b))



for OP in (:stieltjes,:stieltjesintegral,:pseudostieltjes)
    @eval $OP(f::Fun) = $OP(space(f),coefficients(f))
    @eval $OP(f::Fun,z) = $OP(space(f),coefficients(f),z)
end

hilbert(f) = Hilbert()*f
hilbert(S,f,z) = hilbert(Fun(S,f))(z)
hilbert(f::Fun,z) = hilbert(space(f),coefficients(f),z)

logkernel(f::Fun,z) = logkernel(space(f),coefficients(f),z)





cauchy(f...) = stieltjes(f...)*(im/(2π))
pseudocauchy(f...) = pseudostieltjes(f...)*(im/(2π))
cauchyintegral(u...) = stieltjesintegral(u...)*(im/(2π))

function singularintegral(k::Integer,s::Space,f,z)
    k == 0 && return logkernel(s,f,z)
    k == 1 && isreal(domain(s)) && return stieltjes(s,f,z)/π
    k == 1 && domain(s) isa Segment && return stieltjes(s,f/sign(domain(s)),z)/π
    error("Not implemented")
end

singularintegral(k::Integer,s::PiecewiseSpace,f,z) =
    mapreduce(g -> singularintegral(k, g, z), +, pieces(Fun(s,f)))


function csingularintegral(k::Integer,s::Space,f,z)
    k == 0 && error("Not defined")
    k == 1 && return stieltjes(s,f,z)/π
    return epsilon(csingularintegral(k-1,s,f,dual(z,1)))
end


singularintegral(k::Integer,f::Fun,z) = singularintegral(k,space(f),coefficients(f),z)
csingularintegral(k::Integer,f::Fun,z) = csingularintegral(k,space(f),coefficients(f),z)

# Modifier spaces

stieltjes(sp::SubSpace,v,z) = stieltjes(sp.space,coefficients(v,sp,sp.space),z)


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


# if isdir(Pkg.dir("TikzGraphs"))
#     include("introspect.jl")
# end

include("Extras/Extras.jl")


using Test

function testsieoperators(S::Space)
    testbandedoperator(SingularIntegral(S,0))
    testbandedoperator(SingularIntegral(S,1))
    testbandedoperator(Hilbert(S))
    p=ApproxFun.checkpoints(S)[1] # random point on contour
    x=Fun(domain(S))
    z=2.12312231+1.433453443534im # random point not on contour

    for k=1:5
        f=Fun(S,[zeros(k-1);1])
        @test (SingularIntegral(S,0)*f)(p) ≈ logkernel(f,p)
        @test (Hilbert(S,1)*f)(p) ≈ hilbert(f,p)
    end
end


function testsieeval(S::Space;posdirection=im)
    p=ApproxFun.checkpoints(S)[1] # random point on contour
    x=Fun(domain(S))
    z=2.12312231+1.433453443534im # random point not on contour

    for k=1:5
        f=Fun(S,[zeros(k-1);1])
        @test abs(sum(f/(z-x))-stieltjes(f,z)) ≤ 100eps()
        @test stieltjes(f,(p)⁺) ≈ stieltjes(f,p+eps()*posdirection)
        @test stieltjes(f,(p)⁻) ≈ stieltjes(f,p-eps()*posdirection)
        @test cauchy(f,(p)⁺)-cauchy(f,(p)⁻) ≈ f(p)
        @test im*(cauchy(f,(p)⁺)+cauchy(f,(p)⁻)) ≈ hilbert(f,p)

        @test abs(linesum(f*logabs(x-z))/π-logkernel(f,z)) ≤ 100eps()
        @test logkernel(f,p) ≈ logkernel(f,p+eps()*posdirection)
    end
end

function testsies(S::Space;posdirection=im)
    testsieoperators(S)
    testsieeval(S;posdirection=posdirection)
end

function testsies(d::IntervalOrSegmentDomain; posdirection=im)
    testsies(JacobiWeight(-0.5,-0.5,Chebyshev(d)))
    testsies(JacobiWeight(0.5,0.5,Ultraspherical(1,d)))
    testsies(JacobiWeight(-0.5,-0.5,ChebyshevDirichlet{1,1}(d)))
    testsieeval(Legendre(d);posdirection=posdirection)
end

end #module
