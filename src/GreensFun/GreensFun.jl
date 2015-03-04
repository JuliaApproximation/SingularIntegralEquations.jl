include("CauchyWeight.jl")
include("PrincipalValue.jl")

# GreensFun

export GreensFun

immutable GreensFun <: BivariateFun
    kernels::Vector{ProductFun}
    function GreensFun(kernels)
        n = length(kernels)
        #[@assert eltype(kernels[i]) == eltype(kernels[1]) for i=1:n]
        # TODO: should probably be a space assertion but complicated by enrichment.
        # TODO: ProductDomain needs equality.
        [@assert domain(kernels[i]).domains[j] == domain(kernels[1]).domains[j] for i=1:n,j=1:length(domain(kernels[1]))]
        new(kernels)
    end
end

Base.length(G::GreensFun) = length(G.kernels)

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

+{S<:FunctionSpace,V<:FunctionSpace,O,T}(F::GreensFun,G::ProductFun{S,V,CauchyWeight{O},T}) = GreensFun([F.kernels,G])
+{S<:FunctionSpace,V<:FunctionSpace,O,T}(F::ProductFun{S,V,CauchyWeight{O},T},G::GreensFun) = GreensFun([F,G.kernels])

+(F::GreensFun,G::ProductFun) = GreensFun([F.kernels,G])
+(F::ProductFun,G::GreensFun) = GreensFun([F,G.kernels])

+(F::GreensFun,G::GreensFun) = GreensFun([F.kernels,G.kernels])

-{S<:FunctionSpace,V<:FunctionSpace,O1,O2,T1,T2}(F::ProductFun{S,V,CauchyWeight{O1},T1},G::ProductFun{S,V,CauchyWeight{O2},T2}) = GreensFun([F,-G])
-{S<:FunctionSpace,V<:FunctionSpace,O1,SS,T1,T2}(F::ProductFun{S,V,CauchyWeight{O1},T1},G::ProductFun{S,V,SS,T2}) = GreensFun([F,-G])
-{S<:FunctionSpace,V<:FunctionSpace,SS<:AbstractProductSpace,O1,T1,T2}(F::ProductFun{S,V,SS,T1},G::ProductFun{S,V,CauchyWeight{O1},T2}) = GreensFun([F,-G])

-{S<:FunctionSpace,V<:FunctionSpace,O,T}(F::GreensFun,G::ProductFun{S,V,CauchyWeight{O},T}) = GreensFun([F.kernels,-G])
-{S<:FunctionSpace,V<:FunctionSpace,O,T}(F::ProductFun{S,V,CauchyWeight{O},T},G::GreensFun) = GreensFun([F,-G.kernels])

-(F::GreensFun,G::ProductFun) = GreensFun([F.kernels,-G])
-(F::ProductFun,G::GreensFun) = GreensFun([F,-G.kernels])

-(F::GreensFun,G::GreensFun) = GreensFun([F.kernels,-G.kernels])

function evaluate(G::GreensFun,x,y)
    ret = evaluate(G.kernels[1],x,y)
    for i = 2:length(G)
        ret += evaluate(G.kernels[i],x,y)
    end

    ret
end

function Base.getindex(⨍::PrincipalValue,G::GreensFun)
    ret = ⨍[G.kernels[1]]
    for i = 2:length(G)
        ret += ⨍[G.kernels[i]]
    end

    ret
end

export LowRankPositiveDefiniteFun

function LowRankPositiveDefiniteFun(f::Function,spx::FunctionSpace,spy::FunctionSpace)
    dx,dy = domain(spx),domain(spy)
    T,dz = eltype(dx),Chebyshev([dx.a+dy.a,dx.b+dy.b])
    ff = Fun(x->f(-x/2,x/2),dz)
    fd = ff[(dx.a+dx.b)/2]
    fnew(x,y) = x == y ? fd : f(x,y)
    tol = maxabs(coefficients(ff))*100eps(T)
    c = chop(coefficients(ff),tol)
    N = length(c)
    pts=points(dz,N)
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
# defined as the difference of their arguments.
#
function ConvolutionProductFun{U<:PolynomialSpace,V<:PolynomialSpace}(f::Function,u::Union(U,JacobiWeight{U}),v::Union(V,JacobiWeight{V}))
    du,dv = domain(u),domain(v)
    @assert length(du) == length(dv)
    ff = Fun(x->f(-x/2,x/2),Chebyshev([du.a+dv.a,du.b+dv.b]))
    T,fd = eltype(ff),ff[(du.a+du.b)/2]
    c = chop(coefficients(ff),maxabs(coefficients(ff))*100eps(T))
    N = length(c)

    if N ≤ 3000
        if N ≤ 3 N=3;pad!(c,3) end
        X = zeros(T,N,N)
        chebyshevaddition!(c,X)
        cspu,cspv = canonicalspace(u),canonicalspace(v)
        [X[1:N+1-k,k] = coefficients(vec(X[1:N+1-k,k]),cspu,u) for k=1:N]
        [X[k,1:N+1-k] = coefficients(vec(X[k,1:N+1-k]),cspv,v) for k=1:N]
        return ProductFun(X,u⊗v)
    else
        return ProductFun((x,y)->x==y?fd:f(x,y),u⊗v,N,N)
    end
end

function chebyshevaddition!{T<:Number}(c::Vector{T},X::Matrix{T})
    N = length(c)
    un = one(T)
    C1,C2 = zeros(T,N,N),zeros(T,N,N)

    C1[1,1] = un
    cn = c[1]

    X[1,1] += cn*C1[1,1]

    C2[2,1] = -un/2
    C2[1,2] = un/2
    cn = c[2]

    X[2,1] += cn*C2[2,1]
    X[1,2] += cn*C2[1,2]

    C1[1,1] = -un/2
    C1[3,1] = un/4
    C1[2,2] = -un
    C1[1,3] = un/4
    cn = c[3]

    X[1,1] += cn*C1[1,1]
    X[3,1] += cn*C1[3,1]
    X[2,2] += cn*C1[2,2]
    X[1,3] += cn*C1[1,3]

    @inbounds for n=4:N
        #
        # There are 11 unique recurrence relationships for the coefficients. The main recurrence is:
        #
        # C[i,j,n] = (C[i,j+1,n-1]+C[i,j-1,n-1]-C[i+1,j,n-1]-C[i-1,j,n-1])/2 - C[i,j,n-2],
        #
        # and the other 10 come from shutting some terms off if they are out of bounds,
        # or for the row C[2,1:n,n] or column C[1:n,2,n] terms are turned on. This follows from
        # the reflection of Chebyshev polynomials: 2T_m(x)T_n(x) = T_{m+n}(x) + T_|m-n|(x).
        # For testing of stability, they should always be equal to:
        # C[1:n,1:n,n] = coefficients(ProductFun((x,y)->cos((n-1)*acos((y-x)/2)))).
        #
        C2[1,1] = (C1[1,2]-C1[2,1])/2 - C2[1,1]
        C2[2,1] = (C1[2,2]-C1[3,1])/2 - C1[1,1] - C2[2,1]
        C2[n,1] = C1[n-1,1]/(-2)
        C2[1,2] = (C1[1,3]-C1[2,2])/2 + C1[1,1] - C2[1,2]
        C2[2,2] = (C1[2,3]-C1[3,2])/2 + C1[2,1] - C1[1,2] - C2[2,2]
        C2[1,n] = C1[1,n-1]/2
        for k=n-2:-2:3
            C2[k,1] = (C1[k,2]-C1[k-1,1]-C1[k+1,1])/2 - C2[k,1]
            C2[1,k] = (C1[1,k+1]+C1[1,k-1]-C1[2,k])/2 - C2[1,k]
        end
        for k=n-1:-2:3
            C2[k,2] = (C1[k,3]-C1[k-1,2]-C1[k+1,2])/2 + C1[k,1] - C2[k,2]
            C2[2,k] = (C1[2,k+1]+C1[2,k-1]-C1[3,k])/2 - C1[1,k] - C2[2,k]
        end
        for j=n:-1:3,i=n-j+1:-2:3
            C2[i,j] = (C1[i,j+1]+C1[i,j-1]-C1[i+1,j]-C1[i-1,j])/2 - C2[i,j]
        end

        cn = c[n]
        for j=n:-1:1,i=n-j+1:-2:1
            X[i,j] += cn*C2[i,j]
        end

        for j=1:n,i=1:n-j+1
            C1[i,j],C2[i,j] = C2[i,j],C1[i,j]
        end
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
