include("CauchyWeight.jl")

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
# defined as the difference of their arguments.
#
function ConvolutionProductFun{U<:PolynomialSpace,V<:PolynomialSpace}(f::Function,u::Union(U,JacobiWeight{U}),v::Union(V,JacobiWeight{V}))
    du,dv = domain(u),domain(v)
    pts,ext = points(Chebyshev(du)⊗Chebyshev(dv),3,3),extrema(du,dv)
    A = abs2(pts[2]-pts[1])
    ApproxFun.transform!(Chebyshev(du)⊗Chebyshev(dv),A)
    println("These are the Chebyshev coefficients of the absolute value squared on the two intervals: ",A)
    println("These are the extrema of the intervals in question: ",ext)

    #ab2 = ProductFun((x,y)->abs2(y-x),Chebyshev(du)⊗Chebyshev(dv),3,3)
    #println("This is the absolute value squared on the two intervals: ",ab2)
    #ff = Fun(z->f((du.a+du.b-z)/2,(dv.b+dv.a-im*z)/2),Chebyshev([-1,1]))
    #fd,B,C = ff[0],(dv.b-dv.a)/2D,(du.b-du.a)/2D
    #fd,B,C = ff[zero(T)],real(length(dv)/2D),real(length(du)/2D)

    ff = Fun(z->f(0,sqrt(z)),Chebyshev(Interval(ext...)))
    fd,T = f[0],eltype(ff)
    c = chop(coefficients(ff),maxabs(coefficients(ff))*100eps(T))
    N = length(c)

    if N ≤ 3000
        if N ≤ 3 N=3;pad!(c,3) end
        X = zeros(promote_type(T,typeof(A)),N,N)
        println("This is the typeof(c): ",typeof(c)," this is A: ",A," and this is the typeof(X): ",typeof(X))
        chebyshevaddition!(c,A,X)



        cspu,cspv = canonicalspace(u),canonicalspace(v)
        [X[1:N+1-k,k] = coefficients(vec(X[1:N+1-k,k]),cspu,u) for k=1:N]
        [X[k,1:N+1-k] = coefficients(vec(X[k,1:N+1-k]),cspv,v) for k=1:N]
        return ProductFun(X,u⊗v)
    else
        return ProductFun((x,y)->x==y?fd:f(x,y),u⊗v,N,N)
    end
end

function chebyshevaddition!{T<:Number}(c::Vector{T},A::Matrix{T},X::Matrix{T})
    N = length(c)
    un = one(T)
    C1,C2 = zeros(T,N,N),zeros(T,N,N)

    C1[1,1] = un
    cn = c[1]

    X[1,1] += cn*C1[1,1]

    C2[2,1] = -C
    C2[1,2] = B

    cn = c[2]

    X[2,1] += cn*C2[2,1]
    X[1,2] += cn*C2[1,2]

    C1[1,1] = B^2+C^2-un
    C1[3,1] = C^2
    C1[2,2] = -4B*C
    C1[1,3] = B^2
    cn = c[3]

    X[1,1] += cn*C1[1,1]
    X[3,1] += cn*C1[3,1]
    X[2,2] += cn*C1[2,2]
    X[1,3] += cn*C1[1,3]

    @inbounds for n=4:N
        C2[1,1] = B*C1[1,2] - C*C1[2,1] - C2[1,1]
        C2[2,1] = B*C1[2,2] - C*C1[3,1] - 2C*C1[1,1] - C2[2,1]
        C2[n,1] = -C*C1[n-1,1]
        C2[1,2] = B*C1[1,3]-C*C1[2,2] + 2B*C1[1,1] - C2[1,2]
        C2[2,2] = B*C1[2,3]-C*C1[3,2] + 2B*C1[2,1] - 2C*C1[1,2] - C2[2,2]
        C2[1,n] = B*C1[1,n-1]
        for k=n-2:-2:3
            C2[k,1] = B*C1[k,2]-C*C1[k-1,1]-C*C1[k+1,1] - C2[k,1]
            C2[1,k] = B*C1[1,k+1]+B*C1[1,k-1]-C*C1[2,k] - C2[1,k]
        end
        for k=n-1:-2:3
            C2[k,2] = B*C1[k,3]-C*C1[k-1,2]-C*C1[k+1,2] + 2B*C1[k,1] - C2[k,2]
            C2[2,k] = B*C1[2,k+1]+B*C1[2,k-1]-C*C1[3,k] - 2C*C1[1,k] - C2[2,k]
        end
        for j=n:-1:3,i=n-j+1:-2:3
            C2[i,j] = B*C1[i,j+1]+B*C1[i,j-1]-C*C1[i+1,j]-C*C1[i-1,j] - C2[i,j]
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
# Geometry for Intervals
#
function dist2(c,d::Interval)
    if in(c,d)
        ret = zero(c)
    else
        a,b = d.a,d.b
        x1,y1 = real(a),imag(a)
        x2,y2 = real(b),imag(b)
        x3,y3 = real(c),imag(c)
        px,py = x2-x1,y2-y1
        u = ((x3-x1)px+(y3-y1)py)/(px^2+py^2)
        u = u > 1 ? 1 : u ≥ 0 ? u : 0
        dx,dy = x1+u*px-x3,y1+u*py-y3
        dx^2+dy^2
    end
end
dist(c,d::Interval) = sqrt(dist2(c,d))

function extrema2(d1::Interval,d2::Interval)
    a,b = d1.a,d1.b
    c,d = d2.a,d2.b
    extrema((dist2(a,d2),dist2(b,d2),dist2(c,d1),dist2(d,d1),abs2(a-c),abs2(a-d),abs2(b-c),abs2(b-d)))
end
Base.extrema(d1::Interval,d2::Interval) = sqrtuple(extrema2(d1,d2))
sqrtuple(x) = sqrt(x[1]),sqrt(x[2])

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
