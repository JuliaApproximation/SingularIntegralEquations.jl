include("CauchyWeight.jl")
include("Geometry.jl")

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


GreensFun{SS<:AbstractProductSpace}(f::Function,ss::SS;method::Symbol=:convolution) = GreensFun(ProductFun(f,ss;method=method))

# Array of GreensFun on TensorSpace of PiecewiseSpaces

function GreensFun{PWS1<:PiecewiseSpace,PWS2<:PiecewiseSpace}(f::Function,ss::AbstractProductSpace{PWS1,PWS2};method::Symbol=:convolution)
    pws1,pws2 = ss.spaces
    M,N = length(pws1),length(pws2)
    G = Array(GreensFun,M,N)
    for i=1:M,j=1:N
        G[i,j] = ProductFun(f,pws1[i],pws2[j];method=method)
    end
    G
end

function GreensFun{O}(f::Function,ss::CauchyWeight{O};method::Symbol=:convolution)
    pws1,pws2 = ss.space.spaces
    M,N = length(pws1),length(pws2)
    if M == N == 1
        G = GreensFun(f,ss;method=method)
    else
        G = Array(GreensFun,M,N)
        for i=1:M,j=1:N
            G[i,j] = i == j ?  ProductFun(f,CauchyWeight{O}(pws1[i]⊗pws2[j]);method=method) : ProductFun((x,y)->f(x,y).*cauchyweight(O,x,y),pws1[i],pws2[j];method=method)
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
    ret = Array(typeof(⨍j[B[1,1]]),m,n)
    for j=1:n
        ⨍j = DefiniteLineIntegral(wsp[j])
        for i=1:m
            ret[i,j] = ⨍j[B[i,j]]
        end
    end

    ret
end

export LowRankPositiveDefiniteFun

function LowRankPositiveDefiniteFun(f::Function,spx::FunctionSpace,spy::FunctionSpace)
    dx,dy = domain(spx),domain(spy)
    dz = Interval([dx.a+dy.a,dx.b+dy.b])
    ff = Fun(x->f(-x/2,x/2),Chebyshev(dz))
    T,fd = eltype(ff),ff[(dx.a+dx.b)/2]
    fnew(x,y) = x == y ? fd : f(x,y)
    tol = maxabs(coefficients(ff))*100eps(T)
    c = chop(coefficients(ff),tol)
    N = length(c)
    pts=points(Chebyshev(dz),N)
    r=((dx.a+dx.b)/2,(dy.a+dy.b)/2)
    rold=(r[1]+1,r[2]+1)
    a=Fun(x->fnew(x,r[2]),dx)
    A=typeof(a)[]
    while norm(a.coefficients) > tol && r != rold
        A=[A;a/sqrt(abs(a[r[1]]))]
        r,rold=findposdefapproxmax((x,y)->fnew(x,y)-evaluate(A,A,x,y),pts),r
        Br=map(q->q[r[2]],A)
        a=Fun(x->fnew(x,r[2]),dx; method="abszerocoefficients") - dotu(Br,A)
        a=chop!(a,tol)
    end
    LowRankFun(A,A)
end

function findposdefapproxmax(f::Function,pts::Vector)
    fv = eltype(f(pts[1]/2,pts[1]/2))[abs(f(ptsk/2,ptsk/2)) for ptsk in pts]
    mpt = pts[indmax(fv)]/2
    mpt,mpt
end








function ProductFun(f,u,v;method::Symbol=:standard)
    if method == :standard
        ProductFun(f,u,v)
    elseif method == :convolution
        ConvolutionProductFun(f,u,v)
    end
end

#
# A new ProductFun constructor for bivariate functions on Intervals
# defined as the distance of their arguments.
#
function ConvolutionProductFun{U<:FunctionSpace,V<:FunctionSpace}(f::Function,u::U,v::V)
    du,dv = domain(u),domain(v)
    ext2 = extrema2(du,dv)
    if ext2[1] == 0
        ff = Fun(z->f(0,z),Chebyshev(Interval(-ext2[2]/2,ext2[2]/2)))
        fd,T = ff[0],eltype(ff)
        c = chop(coefficients(ff),norm(coefficients(ff),Inf)*100eps(T))
        N = length(c)
        X = chop!(coefficients(ProductFun((x,y)->x==y?fd:f(x,y),u⊗v,N,N)),norm(coefficients(ff),Inf)*100eps(T))
    else
        ff = Fun(z->f(0,z),Chebyshev(Interval(ext2...)))
        c = chop(coefficients(ff),norm(coefficients(ff),Inf)*100eps(eltype(ff)))
        N = length(c)
        X = chop!(coefficients(ProductFun(f,u⊗v,N,N)),norm(coefficients(ff),Inf)*100eps(eltype(ff)))
    end
    ProductFun(X,u⊗v)
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
