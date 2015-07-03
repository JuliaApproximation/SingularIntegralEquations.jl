export convolutionProductFun

#
# A ProductFun constructor for bivariate functions on Intervals
# defined as the distance of their arguments.
#
function convolutionProductFun{U<:UnivariateSpace,V<:UnivariateSpace}(f::Function,u::U,v::V;tol=eps())
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

convolutionProductFun{U<:UnivariateSpace,V<:UnivariateSpace,T}(f::Function,ss::TensorSpace{(U,V),T,2};kwds...) = convolutionProductFun(f,ss[1],ss[2];kwds...)



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
