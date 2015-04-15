include("CauchyWeight.jl")
include("Geometry.jl")
include("evaluation.jl")

# GreensFun

export GreensFun

immutable GreensFun <: BivariateFun
    kernels::Vector{ProductFun}
    function GreensFun(kernels)
        #[@assert eltype(kernels[i]) == eltype(kernels[1]) for i=1:n]
        # TODO: should probably be a space assertion but complicated by enrichment.
        [@assert domain(kernels[i]) == domain(kernels[1]) for i=2:length(kernels)]
        new(kernels)
    end
end

GreensFun(F::ProductFun) = GreensFun([F])

Base.length(G::GreensFun) = length(G.kernels)
Base.convert{B<:ProductFun}(::Type{GreensFun},F::B) = GreensFun(F)
evaluate(G::GreensFun,x,y) = mapreduce(f->evaluate(f,x,y),+,G.kernels)


GreensFun{SS<:AbstractProductSpace}(f::Function,ss::SS;method::Symbol=:convolution,tol=100eps()) = GreensFun(ProductFun(f,ss.spaces...;method=method,tol=tol))

# Array of GreensFun on TensorSpace of PiecewiseSpaces

function GreensFun{PWS1<:PiecewiseSpace,PWS2<:PiecewiseSpace}(f::Function,ss::AbstractProductSpace{PWS1,PWS2};method::Symbol=:convolution,tol=100eps())
    pws1,pws2 = ss.spaces
    M,N = length(pws1),length(pws2)
    G = Array(GreensFun,M,N)
    for i=1:M,j=1:N
        G[i,j] = ProductFun(f,pws1[i],pws2[j];method=method,tol=tol)
    end
    G
end

function GreensFun{O}(f::Function,ss::CauchyWeight{O};method::Symbol=:convolution,tol=100eps())
    pws1,pws2 = ss.space.spaces
    if !isa(pws1,PiecewiseSpace) && !isa(pws2,PiecewiseSpace)
        G = GreensFun(ProductFun(f,ss;method=method,tol=tol))
    elseif isa(pws1,PiecewiseSpace) && isa(pws2,PiecewiseSpace)
        M,N = length(pws1),length(pws2)
        G = Array(GreensFun,M,N)
        for i=1:M,j=1:N
            G[i,j] = i == j ?  ProductFun(f,CauchyWeight{O}(pws1[i]⊗pws2[j]);method=method,tol=tol) : ProductFun((x,y)->f(x,y).*cauchyweight(O,x,y),pws1[i],pws2[j];method=method,tol=tol)
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

export LowRankCholeskyFun

function findcholeskyapproxmax!(f::Function,X::Vector,pts::Vector,grid)
    @inbounds for k=1:grid
        X[k]+=f(pts[k],pts[k])
    end
    maxabsf,impt = findmax(abs(X))
    maxabsf,pts[impt]
end

function findcholeskyapproxmax!(A::Fun,B::Fun,X::Vector,pts::Vector,grid)
    dX = A[pts].*B[pts]
    X[:] -= dX[:]
    maxabsf,impt = findmax(abs(X))
    maxabsf,pts[impt]
end

function LowRankCholeskyFun(f::Function,dx::FunctionSpace;grid::Integer=64,maxrank::Integer=100)

    Td = eltype(domain(dx))

    # We start by sampling on the given grid, find the approximate maximum and create the first rank-one approximation.
    pts=points(dx,grid)
    Tf = typeof(f(pts[1],pts[1]))
    T = promote_type(Tf,Td)
    X = zeros(T,grid)
    maxabsf,r=findcholeskyapproxmax!(f,X,pts,grid)
    a=Fun(x->f(x,r),dx)

    # If necessary, we resize the grid to be at least as large as the
    # length of the first row/column Fun and we recompute the values of X.
    if grid < length(a)
        grid = max(grid,length(a))
        pts=points(dx,grid)
        X = zeros(T,grid)
        maxabsf,r=findcholeskyapproxmax!(f,X,pts,grid)
        a=Fun(x->f(x,r),dx)
    end

    A,B,tol=typeof(a)[],typeof(a)[],100maxabsf*eps(T)
    tol10 = tol/10

    # Eat, drink, subtract rank-one, repeat.
    for k=1:maxrank

        if norm(a.coefficients,Inf) < tol return LowRankFun(A,B) end

        A,B=[A;a/sqrt(abs(a[r]))],[B;a/(sqrt(abs(a[r]))*sign(a[r]))]

        maxabsf,r=findcholeskyapproxmax!(A[k],B[k],X,pts,grid)

        Br=map(q->q[r],B)

        a=Fun(x->f(x,r),dx,grid) - dot(conj(Br),A)

        chop!(a,tol10)

    end
    warn("Maximum rank of " * string(maxrank) * " reached")
    return LowRankFun(A,B)
end





function ProductFun(f,u,v;method::Symbol=:standard,tol=eps())
    if method == :standard
        ProductFun(f,u,v;tol=tol)
    elseif method == :convolution
        ConvolutionProductFun(f,u,v;tol=tol)
    end
end

#
# A new ProductFun constructor for bivariate functions on Intervals
# defined as the distance of their arguments.
#
function ConvolutionProductFun{U<:FunctionSpace,V<:FunctionSpace}(f::Function,u::U,v::V;tol=eps())
    du,dv = domain(u),domain(v)
    ext2 = extrema2(du,dv)
    if ext2[1] == 0
        ff = Fun(z->f(0,z),Chebyshev(Interval(-ext2[2]/2,ext2[2]/2)))
        fd,T = ff[0],eltype(ff)
        c = chop(coefficients(ff),norm(coefficients(ff),Inf)*100eps(T))
        N = length(c)
        N1 = isa(u,PeriodicSpace) ? 2N : N
        N2 = isa(v,PeriodicSpace) ? 2N : N
        return ProductFun((x,y)->x==y?fd:f(x,y),u⊗v,N1,N2;tol=tol)
    else
        ff = Fun(z->f(0,z),Chebyshev(Interval(ext2...)))
        c = chop(coefficients(ff),norm(coefficients(ff),Inf)*100eps(eltype(ff)))
        N = length(c)
        N1 = isa(u,PeriodicSpace) ? 2N : N
        N2 = isa(v,PeriodicSpace) ? 2N : N
        return ProductFun(f,u⊗v,N1,N2;tol=tol)
    end
end

#
# ProductFun constructors for functions on periodic intervals.
#

#
# Suppose we are interested in K(ϕ-θ). Then, K(⋅) is periodic
# whether it's viewed as bivariate or univariate.
#
function ConvolutionProductFun{S<:Fourier,T,U<:Fourier,V<:Fourier}(f::Fun{S,T},u::U,v::V)
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

function ConvolutionProductFun{S<:CosSpace,T,U<:Fourier,V<:Fourier}(f::Fun{S,T},u::U,v::V)
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

function ConvolutionProductFun{S<:SinSpace,T,U<:Fourier,V<:Fourier}(f::Fun{S,T},u::U,v::V)
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

function ConvolutionProductFun{S<:Laurent,T,U<:Laurent,V<:Laurent}(f::Fun{S,T},u::U,v::V)
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

function ConvolutionProductFun{S<:Taylor,T,U<:Laurent,V<:Laurent}(f::Fun{S,T},u::U,v::V)
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

function ConvolutionProductFun{S<:Hardy{false},T,U<:Laurent,V<:Laurent}(f::Fun{S,T},u::U,v::V)
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
