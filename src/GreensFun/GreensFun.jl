include("CauchyWeight.jl")
include("Geometry.jl")
include("evaluation.jl")
include("skewProductFun.jl")
include("convolutionProductFun.jl")

# GreensFun

export GreensFun

# In GreensFun, K must be <: BivariateFun, since a kernel
# could consist of a Vector of ProductFun's and LowRankFun's.
# If there are GreensFun's in the Vector, then we recurse.

struct GreensFun{K<:BivariateFun,T} <: BivariateFun{T}
    kernels::Vector{K}
    function GreensFun{K,T}(Kernels::Vector{K}) where {K,T}
        if greensfun_checkdomains(Kernels)
            if greensfun_checkgreensfun(Kernels)
                return GreensFun(vcat(map(kernels,Kernels)...))
            end
            new{K,T}(Kernels)
        else
            error("Cannot create GreensFun: all kernel domains must equal $(domain(Kernels[1]))")
        end
    end
end

function greensfun_checkdomains(Kernels::Vector{K}) where K
    d = domain(Kernels[1])
    ret = true
    for i in 2:length(Kernels)
        ret *= (domain(Kernels[i]) == d)
    end
    ret
end

function greensfun_checkgreensfun(Kernels::Vector{K}) where K
    ret = true
    for i in 1:length(Kernels)
        ret *= !(typeof(Kernels[i]) <: GreensFun)
    end
    !ret
end

GreensFun(kernels::Vector{K}) where {K<:MultivariateFun} =
    GreensFun{eltype(kernels),mapreduce(cfstype,promote_type,kernels)}(kernels)

GreensFun(F::K) where {K<:MultivariateFun} = GreensFun(K[F])

convert(::Type{GreensFun{K,T}},A::GreensFun{K,T}) where {K<:BivariateFun,T} = A
convert(::Type{GreensFun{K,T}},A::GreensFun) where {K<:BivariateFun,T} = error("Cannot convert GreensFun")

Base.length(G::GreensFun) = length(G.kernels)
Base.transpose(G::GreensFun) = GreensFun(mapreduce(transpose,+,G.kernels))
convert(::Type{GreensFun},F::Union{ProductFun,LowRankFun}) = GreensFun(F)
rank(G::GreensFun) = error("Not all kernels are low rank approximations.")

domain(G::GreensFun) = domain(first(G.kernels))
(G::GreensFun)(x,y)=evaluate(G,x,y)
evaluate(G::GreensFun,x,y) = mapreduce(f->evaluate(f,x,y),+,G.kernels)
kernels(B::BivariateFun) = B
kernels(G::GreensFun) = G.kernels

rank(G::GreensFun{L}) where {L<:LowRankFun} = mapreduce(rank,+,G.kernels)
slices(G::GreensFun{L}) where {L<:LowRankFun} = mapreduce(x->x.A,vcat,G.kernels),mapreduce(x->x.B,vcat,G.kernels)
slices(G::GreensFun{L},k::Int) where {L<:LowRankFun} = slices(G)[k]
LowRankIntegralOperator(G::GreensFun{L}) where {L<:LowRankFun} = LowRankIntegralOperator(slices(G)...)

Base.promote_rule(::Type{GreensFun{K,T}},::Type{GreensFun{K1,T1}}) where {K,T,K1,T1} = GreensFun{promote_type(K,K1),promote_type(T,T1)}

getindex(⨍::Operator,G::GreensFun) = mapreduce(f->⨍[f],+,G.kernels)

# avoid ambiguity
for TYP in (:(ApproxFun.DefiniteLineIntegralWrapper),:DefiniteLineIntegral)
    @eval function getindex(⨍::$TYP,B::Matrix{F}) where F<:BivariateFun
        m,n = size(B)
        wsp = domainspace(⨍)
        @assert m == length(wsp.spaces)
        ⨍j = DefiniteLineIntegral(component(wsp,1))
        ret = Array{Operator{promote_type(eltype(⨍j),map(cfstype,B)...)}}(undef,m,n)
        for j=1:n
            ⨍j = DefiniteLineIntegral(component(wsp,j))
            for i=1:m
                ret[i,j] = ⨍j[B[i,j]]
            end
        end
        ops=promotespaces(ret)
        InterlaceOperator(ops,
                          PiecewiseSpace(map(domainspace,ops[1,:])),
                          PiecewiseSpace(map(rangespace,ops[:,1])))
    end
end

# Algebra with BivariateFun's

+(F::GreensFun,G::GreensFun) = GreensFun([F.kernels;G.kernels])
-(F::GreensFun,G::GreensFun) = GreensFun([F.kernels;-G.kernels])
+(G::GreensFun,B::BivariateFun) = GreensFun([G.kernels;kernels(B)])
-(G::GreensFun,B::BivariateFun) = GreensFun([G.kernels;-kernels(B)])
+(B::BivariateFun,G::GreensFun) = GreensFun([kernels(B);G.kernels])
-(B::BivariateFun,G::GreensFun) = GreensFun([kernels(B);-G.kernels])

# work around 0.4 bug

# Custom operations on Arrays required to infer type of resulting Array{GreensFun}

for op in (:+,:-)
    @eval begin
        function $op(A::Array{GreensFun{F,T1}},B::Array{GreensFun{G,T2}}) where {F<:BivariateFun,T1,G<:BivariateFun,T2}
            C = similar(A, GreensFun{promote_type(F,G),promote_type(T1,T2)}, promote_shape(size(A),size(B)))
            for i=1:length(A)
                @inbounds C[i] = $op(A[i],B[i])
            end
            return C
        end
    end
end

## TODO: Get ProductFun & LowRankFun in different CauchyWeight spaces to promote to GreensFun.
#=
+{S<:UnivariateSpace,V<:UnivariateSpace,O1,O2}(F::ProductFun{S,V,CauchyWeight{O1}},G::ProductFun{S,V,CauchyWeight{O2}}) = GreensFun([F,G])
+{S<:UnivariateSpace,V<:UnivariateSpace,O1,SS}(F::ProductFun{S,V,CauchyWeight{O1}},G::ProductFun{S,V,SS}) = GreensFun([F,G])
+{S<:UnivariateSpace,V<:UnivariateSpace,SS<:AbstractProductSpace,O1}(F::ProductFun{S,V,SS},G::ProductFun{S,V,CauchyWeight{O1}}) = GreensFun([F,G])

-{S<:UnivariateSpace,V<:UnivariateSpace,O1,O2}(F::ProductFun{S,V,CauchyWeight{O1}},G::ProductFun{S,V,CauchyWeight{O2}}) = GreensFun([F,-G])
-{S<:UnivariateSpace,V<:UnivariateSpace,O1,SS}(F::ProductFun{S,V,CauchyWeight{O1}},G::ProductFun{S,V,SS}) = GreensFun([F,-G])
-{S<:UnivariateSpace,V<:UnivariateSpace,SS<:AbstractProductSpace,O1}(F::ProductFun{S,V,SS},G::ProductFun{S,V,CauchyWeight{O1}}) = GreensFun([F,-G])
=#

## Constructors

GreensFun(f::Function,args...;kwds...) = GreensFun(dynamic(f),args...;kwds...)
GreensFun(f::Function,g::Function,args...;kwds...) = GreensFun(dynamic(f),dynamic(g),args...;kwds...)

function GreensFun(f::DFunction,ss::SS;method::Symbol=:lowrank,kwds...) where SS<:AbstractProductSpace
    if method == :standard
        F = ProductFun(f,ss,kwds...)
    elseif method == :convolution
        F = convolutionProductFun(f,ss,kwds...)
    elseif method == :unsplit
        # Approximate imaginary & smooth part.
        F1 = skewProductFun((x,y)->imag(f(x,y)),ss.space;kwds...)
        # Extract diagonal value.
        d = domain(ss)
        xm,ym = mean([first(d[1]),last(d[1])]),mean([first(d[2]),last(d[2])])
        F1m = F1(xm,ym)
        # Set this normalized part to be the singular part.
        F1 = ProductFun(-coefficients(F1)/F1m/2,ss)
        # Approximate real & smooth part after singular extraction.
        m,n = size(F1)
        if typeof(ss.space) <: TensorSpace && all(k->typeof(ss.space.spaces[k]) <: Chebyshev,1:2)
            F2 = skewProductFun((x,y)->f(x,y) - F1(x,y),ss.space,nextpow2(m),nextpow2(n)+1)
        elseif typeof(ss.space) <: TensorSpace && all(k->typeof(ss.space.spaces[k]) <: Laurent,1:2)
            F2 = skewProductFun((x,y)->f(x,y) - F1(x,y),ss.space,nextpow2(m),nextpow2(n))
        end
        F = [F1;F2]
    elseif method == :lowrank
        F = LowRankFun(f,ss;method=:standard,kwds...)
    elseif method == :Cholesky
        F = LowRankFun(f,ss;method=method,kwds...)
    end
    GreensFun(F)
end

function GreensFun(f::DFunction,g::DFunction,ss::SS;method::Symbol=:unsplit,kwds...) where SS<:AbstractProductSpace
    if method == :unsplit
        # Approximate Riemann function of operator.
        G = skewProductFun(g,ss.space;kwds...)
        # Normalize and set to the singular part.
        F1 = ProductFun(-coefficients(G)/2,ss)
        # Approximate real & smooth part after singular extraction.
        m,n = size(F1)
        if typeof(ss.space) <: TensorSpace && all(k->typeof(ss.space.spaces[k]) <: Chebyshev,1:2)
            F2 = skewProductFun((x,y)->f(x,y) - F1(x,y),ss.space,nextpow2(m),nextpow2(n)+1)
        elseif typeof(ss.space) <: TensorSpace && all(k->typeof(ss.space.spaces[k]) <: Laurent,1:2)
            F2 = skewProductFun((x,y)->f(x,y) - F1(x,y),ss.space,nextpow2(m),nextpow2(n))
        end
        F = [F1;F2]
    end
    GreensFun(F)
end

# Array of GreensFun on TensorSpace of PiecewiseSpaces

function GreensFun(f::DFunction,ss::AbstractProductSpace{Tuple{PWS1,PWS2}};method::Symbol=:lowrank,tolerance::Symbol=:absolute,kwds...) where {PWS1<:PiecewiseSpace,PWS2<:PiecewiseSpace}
    M,N = ncomponents(factor(ss,1)),ncomponents(factor(ss,2))
    @assert M == N
    G = Array{GreensFun}(undef,N,N)
    if method == :standard
        for i=1:N,j=1:N
            G[i,j] = GreensFun(f,component(ss,i,j);method=method,kwds...)
        end
    elseif method == :convolution
        for i=1:N,j=i:N
            G[i,j] = GreensFun(f,component(ss,i,j);method=method,kwds...)
        end
        for i=2:N,j=1:i-1
            G[i,j] = transpose(G[j,i])
        end
    elseif method == :unsplit
        maxF = Array{Number}(undef,N)
        for i=1:N
          G[i,i] = GreensFun(f,component(ss,i,i);method=method,kwds...)
          maxF[i] = one(real(mapreduce(cfstype,promote_type,G[i,i].kernels)))/2π
        end
        for i=1:N,j=i+1:N
            G[i,j] = GreensFun(f,component(ss,i,j).space;method=:lowrank,tolerance=(tolerance,max(maxF[i],maxF[j])),kwds...)
        end
        for i=2:N,j=1:i-1
            G[i,j] = transpose(G[j,i])
        end
    elseif method == :lowrank
        for i=1:N,j=1:N
            G[i,j] = GreensFun(f,component(ss,i,j);method=method,kwds...)
        end
    elseif method == :Cholesky
        if tolerance == :relative
            for i=1:N
                G[i,i] = GreensFun(f,component(ss,i,i);method=method,kwds...)
                for j=i+1:N
                    G[i,j] = GreensFun(f,component(ss,i,j);method=:lowrank,kwds...)
                end
            end
        elseif tolerance == :absolute
            maxF = Array{Number}(undef,N)
            for i=1:N
                F,maxF[i] = LowRankFun(f,component(ss,i,i);method=method,retmax=true,kwds...)
                G[i,i] = GreensFun(F)
            end
            for i=1:N,j=i+1:N
                G[i,j] = GreensFun(f,component(ss,i,j);method=:lowrank,tolerance=(tolerance,max(maxF[i],maxF[j])),kwds...)
            end
        end
        for i=2:N,j=1:i-1
            G[i,j] = transpose(G[j,i])
        end
    end
    G
end

function GreensFun(f::DFunction,g::DFunction,ss::AbstractProductSpace{Tuple{PWS1,PWS2}};method::Symbol=:unsplit,tolerance::Symbol=:absolute,kwds...) where {PWS1<:PiecewiseSpace,PWS2<:PiecewiseSpace}
    M,N = ncomponents(factor(ss.space,1)),ncomponents(factor(ss.space,2))
    @assert M == N
    G = Array{GreensFun}(undef,N,N)
    if method == :unsplit
        maxF = Array{Number}(undef,N)
        for i=1:N
            G[i,i] = GreensFun(f,g,ss[i,i];method=method,kwds...)
            maxF[i] = one(real(mapreduce(cfstype,promote_type,G[i,i].kernels)))/2π
        end
        for i=1:N
            for j=i+1:N
                G[i,j] = GreensFun(f,ss[i,j].space;method=:lowrank,tolerance=(tolerance,max(maxF[i],maxF[j])),kwds...)
            end
        end
        for i=1:N,j=1:i-1
            G[i,j] = transpose(G[j,i])
        end
    end
    G
end
