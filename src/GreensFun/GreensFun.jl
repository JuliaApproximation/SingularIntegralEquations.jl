include("CauchyWeight.jl")
include("Geometry.jl")
include("evaluation.jl")

# GreensFun

export GreensFun

immutable GreensFun <: BivariateFun
    kernels::Vector{Union(ProductFun,LowRankFun)}
    function GreensFun(kernels)
        d = domain(kernels[1])
        [@assert domain(kernels[i]) == d for i=2:length(kernels)]
        new(kernels)
    end
end

GreensFun(F::Union(ProductFun,LowRankFun)) = GreensFun([F])

Base.length(G::GreensFun) = length(G.kernels)
Base.transpose(G::GreensFun) = mapreduce(transpose,+,G.kernels)
Base.convert(::Type{GreensFun},F::Union(ProductFun,LowRankFun)) = GreensFun(F)
function Base.rank(G::GreensFun)
    if all([typeof(G.kernels[i]) <: LowRankFun for i=1:length(G)])
        return tuple(map(rank,G.kernels)...)
    else
        error("Not all kernels are low rank approximations.")
    end
end
domain(G::GreensFun) = domain(first(G.kernels))
evaluate(G::GreensFun,x,y) = mapreduce(f->evaluate(f,x,y),+,G.kernels)


function GreensFun{SS<:AbstractProductSpace}(f::Function,ss::SS;method::Symbol=:lowrank,kwds...)
    if method == :standard
        F = ProductFun(f,ss,kwds...)
    elseif method == :convolution
        F = convolutionProductFun(f,ss,kwds...)
    elseif method == :lowrank
        F = LowRankFun(f,ss;method=:standard,kwds...)
    elseif method == :Cholesky
        F = LowRankFun(f,ss;method=method,kwds...)
    end
    GreensFun(F)
end

# Array of GreensFun on TensorSpace of PiecewiseSpaces

function GreensFun{PWS1<:PiecewiseSpace,PWS2<:PiecewiseSpace}(f::Function,ss::AbstractProductSpace{(PWS1,PWS2)};method::Symbol=:lowrank,tolerance::Symbol=:absolute,kwds...)
    pws1,pws2 = ss[1],ss[2]
    M,N = length(pws1),length(pws2)
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
        for i=1:N,j=1:i-1
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
            for i=1:N
                for j=i+1:N
                    G[i,j] = GreensFun(f,ss[i,j];method=:lowrank,tolerance=(tolerance,max(maxF[i],maxF[j])),kwds...)
                end
            end
        end
        for i=1:N,j=1:i-1
            G[i,j] = transpose(G[j,i])
        end
    end
    G
end



# TODO: We are missing unary operation + for a ProductFun
#=
for op = (:+,:-)
    @eval begin
        $op{S,V,O,T}(F::ProductFun{S,V,CauchyWeight{0},T},G::ProductFun{S,V,CauchyWeight{O},T}) = GreensFun([F,$op(G)])
    end
end
=#


+{S<:FunctionSpace,V<:FunctionSpace,O1,O2,T1,T2}(F::ProductFun{S,V,CauchyWeight{O1},T1},G::ProductFun{S,V,CauchyWeight{O2},T2}) = GreensFun([F,G])
+{S<:FunctionSpace,V<:FunctionSpace,O1,SS,T1,T2}(F::ProductFun{S,V,CauchyWeight{O1},T1},G::ProductFun{S,V,SS,T2}) = GreensFun([F,G])
+{S<:FunctionSpace,V<:FunctionSpace,SS,O1,T1,T2}(F::ProductFun{S,V,SS,T1},G::ProductFun{S,V,CauchyWeight{O1},T2}) = GreensFun([F,G])

+(F::GreensFun,G::ProductFun) = GreensFun([F.kernels,G])
+(F::ProductFun,G::GreensFun) = GreensFun([F,G.kernels])

+(F::GreensFun,G::GreensFun) = GreensFun([F.kernels,G.kernels])

-{S<:FunctionSpace,V<:FunctionSpace,O1,O2,T1,T2}(F::ProductFun{S,V,CauchyWeight{O1},T1},G::ProductFun{S,V,CauchyWeight{O2},T2}) = GreensFun([F,-G])
-{S<:FunctionSpace,V<:FunctionSpace,O1,SS,T1,T2}(F::ProductFun{S,V,CauchyWeight{O1},T1},G::ProductFun{S,V,SS,T2}) = GreensFun([F,-G])
-{S<:FunctionSpace,V<:FunctionSpace,SS<:AbstractProductSpace,O1,T1,T2}(F::ProductFun{S,V,SS,T1},G::ProductFun{S,V,CauchyWeight{O1},T2}) = GreensFun([F,-G])

-(F::GreensFun,G::ProductFun) = GreensFun([F.kernels,-G])
-(F::ProductFun,G::GreensFun) = GreensFun([F,-G.kernels])

-(F::GreensFun,G::GreensFun) = GreensFun([F.kernels,-G.kernels])


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
    ret = mapreduce(typeof,promote_type,ret)[ret[j,i] for j=1:n,i=1:m]

    ret
end


export convolutionProductFun

#
# A new ProductFun constructor for bivariate functions on Intervals
# defined as the distance of their arguments.
#
function convolutionProductFun{U<:FunctionSpace,V<:FunctionSpace}(f::Function,u::U,v::V;tol=eps())
    du,dv = domain(u),domain(v)
    ext = extrema(du,dv)
    if ext[1] == 0
        ff = Fun(z->f(0,z),Chebyshev(Interval(-ext[2]/2,ext[2]/2)))
        fd,T = ff[0],eltype(ff)
        c = chop(coefficients(ff),norm(coefficients(ff),Inf)*100eps(T))
        N = length(c)
        N1 = isa(du,PeriodicDomain) ? 2N : N
        N2 = isa(dv,PeriodicDomain) ? 2N : N
        return ProductFun((x,y)->x==y?fd:f(x,y),u⊗v,N1,N2;tol=tol)
    else
        ff = Fun(z->f(0,z),Chebyshev(Interval(ext...)))
        c = chop(coefficients(ff),norm(coefficients(ff),Inf)*100eps(eltype(ff)))
        N = length(c)
        N1 = isa(du,PeriodicDomain) ? 2N : N
        N2 = isa(dv,PeriodicDomain) ? 2N : N
        return ProductFun(f,u⊗v,N1,N2;tol=tol)
    end
end

convolutionProductFun{U<:FunctionSpace,V<:FunctionSpace,T}(f::Function,ss::TensorSpace{(U,V),T,2};kwds...) = convolutionProductFun(f,ss[1],ss[2];kwds...)



#
# ProductFun constructors for functions on periodic intervals.
#

#
# Suppose we are interested in K(ϕ-θ). Then, K(⋅) is periodic
# whether it's viewed as bivariate or univariate.
#
function convolutionProductFun{S<:Fourier,T,U<:Fourier,V<:Fourier}(f::Fun{S,T},u::U,v::V;tol=eps())
    df,du,dv = domain(f),domain(u),domain(v)
    @assert df == du == dv && isa(df,PeriodicInterval)
    c = coefficients(f)
    N = length(c)
    X = zeros(T,N,N)
    X[1,1] += c[1]
    @inbounds for i=2:2:N-1
        X[i,i] += c[i+1]
        X[i+1,i] += c[i]
        X[i,i+1] -= c[i]
        X[i+1,i+1] += c[i+1]
    end
    if mod(N,2)==0 X[N,N-1],X[N-1,N] = c[N],-c[N] end
    ProductFun(X,u⊗v)
end

function convolutionProductFun{S<:CosSpace,T,U<:Fourier,V<:Fourier}(f::Fun{S,T},u::U,v::V;tol=eps())
    df,du,dv = domain(f),domain(u),domain(v)
    @assert df == du == dv && isa(df,PeriodicInterval)
    c = coefficients(f)
    N = 2length(c)-1
    X = zeros(T,N,N)
    X[1,1] += c[1]
    @inbounds for i=2:2:N
        X[i,i] += c[i/2+1]
        X[i+1,i+1] += c[i/2+1]
    end
    ProductFun(X,u⊗v)
end

function convolutionProductFun{S<:SinSpace,T,U<:Fourier,V<:Fourier}(f::Fun{S,T},u::U,v::V;tol=eps())
    df,du,dv = domain(f),domain(u),domain(v)
    @assert df == du == dv && isa(df,PeriodicInterval)
    c = coefficients(f)
    N = 2length(c)+1
    X = zeros(T,N,N)
    @inbounds for i=2:2:N
        X[i+1,i] += c[i/2]
        X[i,i+1] -= c[i/2]
    end
    ProductFun(X,u⊗v)
end

function convolutionProductFun{S<:Laurent,T,U<:Laurent,V<:Laurent}(f::Fun{S,T},u::U,v::V;tol=eps())
    df,du,dv = domain(f),domain(u),domain(v)
    @assert df == du == dv && isa(df,PeriodicInterval)
    c = coefficients(f)
    N = length(c)
    X = mod(N,2) == 0 ? zeros(T,N+1,N) : zeros(T,N,N)
    X[1,1] += c[1]
    @inbounds for i=2:2:N-1
        X[i+1,i] += c[i]
        X[i,i+1] += c[i+1]
    end
    if mod(N,2)==0 X[N+1,N] = c[N] end
    ProductFun(X,u⊗v)
end

function convolutionProductFun{S<:Taylor,T,U<:Laurent,V<:Laurent}(f::Fun{S,T},u::U,v::V;tol=eps())
    df,du,dv = domain(f),domain(u),domain(v)
    @assert df == du == dv && isa(df,PeriodicInterval)
    c = coefficients(f)
    N = 2length(c)-1
    X = zeros(T,N-1,N)
    X[1,1] += c[1]
    @inbounds for i=2:2:N-1
        X[i,i+1] += c[i/2+1]
    end
    ProductFun(X,u⊗v)
end

function convolutionProductFun{S<:Hardy{false},T,U<:Laurent,V<:Laurent}(f::Fun{S,T},u::U,v::V;tol=eps())
    df,du,dv = domain(f),domain(u),domain(v)
    @assert df == du == dv && isa(df,PeriodicInterval)
    c = coefficients(f)
    N = 2length(c)
    X = zeros(T,N+1,N)
    @inbounds for i=2:2:N
        X[i+1,i] += c[i/2]
    end
    ProductFun(X,u⊗v)
end
