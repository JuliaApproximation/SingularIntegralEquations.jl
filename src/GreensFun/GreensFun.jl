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

immutable GreensFun{K<:BivariateFun,T} <: BivariateFun{T}
    kernels::Vector{K}
    function GreensFun(Kernels::Vector{K})
        @assert all(map(domain,Kernels).==domain(Kernels[1]))
        if any(K->K<:GreensFun,map(typeof,Kernels))
            return GreensFun(vcat(map(kernels,Kernels)...))
        end
        new(Kernels)
    end
end
GreensFun{K<:BivariateFun}(kernels::Vector{K}) = GreensFun{eltype(kernels),mapreduce(eltype,promote_type,kernels)}(kernels)

GreensFun{K<:BivariateFun}(F::K) = GreensFun(K[F])

Base.length(G::GreensFun) = length(G.kernels)
Base.transpose(G::GreensFun) = GreensFun(mapreduce(transpose,+,G.kernels))
Base.convert(::Type{GreensFun},F::Union{ProductFun,LowRankFun}) = GreensFun(F)
Base.rank(G::GreensFun) = error("Not all kernels are low rank approximations.")

domain(G::GreensFun) = domain(first(G.kernels))
evaluate(G::GreensFun,x,y) = mapreduce(f->evaluate(f,x,y),+,G.kernels)
kernels(B::BivariateFun) = B
kernels(G::GreensFun) = G.kernels

Base.rank{L<:LowRankFun}(G::GreensFun{L}) = mapreduce(rank,+,G.kernels)
slices{L<:LowRankFun}(G::GreensFun{L}) = mapreduce(x->x.A,vcat,G.kernels),mapreduce(x->x.B,vcat,G.kernels)
slices{L<:LowRankFun}(G::GreensFun{L},k::Int) = slices(G)[k]
LowRankIntegralOperator{L<:LowRankFun}(G::GreensFun{L}) = LowRankIntegralOperator(slices(G)...)

Base.getindex(⨍::Operator,G::GreensFun) = mapreduce(f->getindex(⨍,f),+,G.kernels)

function Base.getindex{F<:BivariateFun}(⨍::DefiniteLineIntegral,B::Matrix{F})
    m,n = size(B)
    wsp = domainspace(⨍)
    @assert m == length(wsp.spaces)
    ⨍j = DefiniteLineIntegral(wsp[1])
    ret = Array(Any,m,n)
    for j=1:n
        ⨍j = DefiniteLineIntegral(wsp[j])
        for i=1:m
            ret[i,j] = ⨍j[B[i,j]]
        end
    end
    interlace(mapreduce(typeof,promote_type,ret)[ret[j,i] for j=1:n,i=1:m])
end

# Algebra with BivariateFun's

+(F::GreensFun,G::GreensFun) = GreensFun([F.kernels;G.kernels])
-(F::GreensFun,G::GreensFun) = GreensFun([F.kernels;-G.kernels])
+(G::GreensFun,B::BivariateFun) = GreensFun([G.kernels;kernels(B)])
-(G::GreensFun,B::BivariateFun) = GreensFun([G.kernels;-kernels(B)])
+(B::BivariateFun,G::GreensFun) = GreensFun([kernels(B);G.kernels])
-(B::BivariateFun,G::GreensFun) = GreensFun([kernels(B);-G.kernels])

# Custom operations on Arrays required to infer type of resulting Array{GreensFun}

for op in (:+,:-)
    @eval begin
        function $op{F<:BivariateFun,T1,G<:BivariateFun,T2}(A::Array{GreensFun{F,T1}},B::Array{GreensFun{G,T2}})
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

function GreensFun{SS<:AbstractProductSpace}(f::Function,ss::SS;method::Symbol=:lowrank,kwds...)
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
        F1m = F1[xm,ym]
        # Set this normalized part to be the singular part.
        F1 = ProductFun(-coefficients(F1)/F1m/2,ss)
        # Approximate real & smooth part after singular extraction.
        m,n = size(F1)
        if typeof(ss.space) <: TensorSpace{(Chebyshev,Chebyshev)}
            F2 = skewProductFun((x,y)->f(x,y) - F1[x,y],ss.space,nextpow2(m),nextpow2(n)+1)
        elseif typeof(ss.space) <: TensorSpace{(Laurent,Laurent)}
            F2 = skewProductFun((x,y)->f(x,y) - F1[x,y],ss.space,nextpow2(m),nextpow2(n))
        end
        F = [F1,F2]
    elseif method == :lowrank
        F = LowRankFun(f,ss;method=:standard,kwds...)
    elseif method == :Cholesky
        F = LowRankFun(f,ss;method=method,kwds...)
    end
    GreensFun(F)
end

function GreensFun{SS<:AbstractProductSpace}(f::Function,g::Function,ss::SS;method::Symbol=:unsplit,kwds...)
    if method == :unsplit
        # Approximate Riemann function of operator.
        G = skewProductFun(g,ss.space;kwds...)
        # Normalize and set to the singular part.
        F1 = ProductFun(-coefficients(G)/2,ss)
        # Approximate real & smooth part after singular extraction.
        m,n = size(F1)
        if typeof(ss.space) <: TensorSpace{(Chebyshev,Chebyshev)}
            F2 = skewProductFun((x,y)->f(x,y) - F1[x,y],ss.space,nextpow2(m),nextpow2(n)+1)
        elseif typeof(ss.space) <: TensorSpace{(Laurent,Laurent)}
            F2 = skewProductFun((x,y)->f(x,y) - F1[x,y],ss.space,nextpow2(m),nextpow2(n))
        end
        F = [F1,F2]
    end
    GreensFun(F)
end

# Array of GreensFun on TensorSpace of PiecewiseSpaces

function GreensFun{PWS1<:PiecewiseSpace,PWS2<:PiecewiseSpace}(f::Function,ss::AbstractProductSpace{Tuple{PWS1,PWS2}};method::Symbol=:lowrank,hierarchical::Bool=false,tolerance::Symbol=:absolute,kwds...)
    M,N = length(ss[1]),length(ss[2])
    if hierarchical
        return hierarchicalGreensFun(f,ss;method=method,tolerance=tolerance,kwds...)
    end
    @assert M == N
    G = Array(GreensFun,N,N)
    if method == :standard
        for i=1:N,j=1:N
            G[i,j] = GreensFun(f,ss[i,j];method=method,kwds...)
        end
    elseif method == :convolution
        for i=1:N,j=i:N
            G[i,j] = GreensFun(f,ss[i,j];method=method,kwds...)
        end
        for i=2:N,j=1:i-1
            G[i,j] = transpose(G[j,i])
        end
    elseif method == :unsplit
        maxF = Array(Number,N)
        for i=1:N
          G[i,i] = GreensFun(f,ss[i,i];method=method,kwds...)
          maxF[i] = one(real(mapreduce(eltype,promote_type,G[i,i].kernels)))/2π
        end
        for i=1:N,j=i+1:N
            G[i,j] = GreensFun(f,ss[i,j].space;method=:lowrank,tolerance=(tolerance,max(maxF[i],maxF[j])),kwds...)
        end
        for i=2:N,j=1:i-1
            G[i,j] = transpose(G[j,i])
        end
    elseif method == :lowrank
        for i=1:N,j=1:N
            G[i,j] = GreensFun(f,ss[i,j];method=method,kwds...)
        end
    elseif method == :Cholesky
        if tolerance == :relative
            for i=1:N
                G[i,i] = GreensFun(f,ss[i,i];method=method,kwds...)
                for j=i+1:N
                    G[i,j] = GreensFun(f,ss[i,j];method=:lowrank,kwds...)
                end
            end
        elseif tolerance == :absolute
            maxF = Array(Number,N)
            for i=1:N
                F,maxF[i] = LowRankFun(f,ss[i,i];method=method,retmax=true,kwds...)
                G[i,i] = GreensFun(F)
            end
            for i=1:N,j=i+1:N
                G[i,j] = GreensFun(f,ss[i,j];method=:lowrank,tolerance=(tolerance,max(maxF[i],maxF[j])),kwds...)
            end
        end
        for i=2:N,j=1:i-1
            G[i,j] = transpose(G[j,i])
        end
    end
    mapreduce(typeof,promote_type,G)[G[i,j] for i=1:N,j=1:N]
end

function GreensFun{PWS1<:PiecewiseSpace,PWS2<:PiecewiseSpace}(f::Function,g::Function,ss::AbstractProductSpace{Tuple{PWS1,PWS2}};method::Symbol=:unsplit,tolerance::Symbol=:absolute,kwds...)
    M,N = length(ss[1]),length(ss[2])
    @assert M == N
    G = Array(GreensFun,N,N)
    if method == :unsplit
        maxF = Array(Number,N)
        for i=1:N
            G[i,i] = GreensFun(f,g,ss[i,i];method=method,kwds...)
            maxF[i] = one(real(mapreduce(eltype,promote_type,G[i,i].kernels)))/2π
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
    mapreduce(typeof,promote_type,G)[G[i,j] for i=1:N,j=1:N]
end

function hierarchicalGreensFun{PWS1<:PiecewiseSpace,PWS2<:PiecewiseSpace}(f::Function,ss::AbstractProductSpace{Tuple{PWS1,PWS2}};method::Symbol=:lowrank,tolerance::Symbol=:absolute,kwds...)
    N = length(ss[2])
    @assert length(ss[1]) == N && ispow2(N)
    N2 = div(N,2)
    meth1 = method == :lowrank ? :standard : method
    meth2 = method == :Cholesky || method == :lowrank ? :standard : method
    if N2 == 1
        G11 = GreensFun(LowRankFun(f,ss[1,1];method=meth1,kwds...))
        G22 = GreensFun(LowRankFun(f,ss[2,2];method=meth1,kwds...))
        G21 = GreensFun(LowRankFun(f,ss[2,1];method=meth2,kwds...))
        G12 = method == :Cholesky ? transpose(G21) : GreensFun(LowRankFun(f,ss[1,2];method=meth2,kwds...))
        return HierarchicalMatrix((G11,G22),(G21,G12))
    elseif N2 ≥ 2
        G21 = GreensFun(LowRankFun(f,ss[1+N2:N,1:N2];method=meth2,kwds...))
        G12 = method == :Cholesky ? transpose(G21) : GreensFun(LowRankFun(f,ss[1:N2,1+N2:N];method=meth2,kwds...))
        return HierarchicalMatrix((hierarchicalGreensFun(f,ss[1:N2,1:N2];method=method,tolerance=tolerance,kwds...),hierarchicalGreensFun(f,ss[1+N2:N,1+N2:N];method=method,tolerance=tolerance,kwds...)),(G21,G12))
    end
end

Base.size{F<:GreensFun,G<:GreensFun}(H::HierarchicalMatrix{F,G}) = 2^nlevels(H),2^nlevels(H)

function domain{F<:GreensFun,G<:GreensFun}(H::HierarchicalMatrix{F,G})
    H11,H22 = diagonaldata(H)
    H21,H12 = offdiagonaldata(H)
    m1,n2 = domain(H12)[1],domain(H12)[2]
    m2,n1 = domain(H21)[1],domain(H21)[2]
    @assert (m1,n1) == (domain(H11)[1],domain(H11)[2])
    @assert (m2,n2) == (domain(H22)[1],domain(H22)[2])
    (m1∪m2)*(n1∪n2)
end

function Base.getindex{G<:GreensFun,L<:LowRankFun,T}(⨍::DefiniteLineIntegral,H::HierarchicalMatrix{G,GreensFun{L,T}})
    H11,H22 = diagonaldata(H)
    HierarchicalOperator((DefiniteLineIntegral(domain(H11)[2])[H11],DefiniteLineIntegral(domain(H22)[2])[H22]),map(LowRankIntegralOperator,offdiagonaldata(H)))
end
