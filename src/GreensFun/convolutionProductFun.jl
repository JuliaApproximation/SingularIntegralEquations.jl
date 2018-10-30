export convolutionProductFun

#
# A ProductFun constructor for bivariate functions on Segments
# defined as the distance of their arguments.
#

convolutionProductFun(f::Function,args...;kwds...) = convolutionProductFun(dynamic(f),args...;kwds...)

function convolutionProductFun(f::DFunction,u::UnivariateSpace,v::UnivariateSpace;tol=eps())
    du,dv = domain(u),domain(v)
    ext = extrema(du,dv)
    if ext[1] == 0
        ff = Fun(z->f(0,z),Chebyshev(Segment(-ext[2]/2,ext[2]/2)))
        fd,T = ff(0),cfstype(ff)
        c = chop(coefficients(ff),norm(coefficients(ff),Inf)*100eps(T))
        N = length(c)
        N1 = isa(du,PeriodicDomain) ? 2N : N
        N2 = isa(dv,PeriodicDomain) ? 2N : N
        return ProductFun((x,y)->x==y ? fd : f(x,y),u⊗v,N1,N2;tol=tol)
    else
        ff = Fun(z->f(0,z),Chebyshev(Segment(ext...)))
        c = chop(coefficients(ff),norm(coefficients(ff),Inf)*100eps(cfstype(ff)))
        N = length(c)
        N1 = isa(du,PeriodicDomain) ? 2N : N
        N2 = isa(dv,PeriodicDomain) ? 2N : N
        return ProductFun(f,u⊗v,N1,N2;tol=tol)
    end
end

convolutionProductFun(f::DFunction,
                      ss::TensorSpace{Tuple{U,V},DD,RR};kwds...) where {U<:UnivariateSpace,V<:UnivariateSpace,DD,RR} =
    convolutionProductFun(f,ss[1],ss[2];kwds...)



#
# ProductFun constructors for functions on periodic intervals.
#

#
# Suppose we are interested in K(ϕ-θ). Then, K(⋅) is periodic
# whether it's viewed as bivariate or univariate.
#

function convolutionProductFun(f::Fun{Fourier{DD,RR},T},u::Fourier{DU,RU},v::Fourier{DV,RV};tol=eps()) where {DD,RR,T,DU,RU,DV,RV}
    df,du,dv = domain(f),domain(u),domain(v)
    @assert df == du == dv && isa(df,PeriodicSegment)
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

function convolutionProductFun(f::Fun{S,T},u::Fourier{DU,RU},v::Fourier{DV,RV};tol=eps()) where {S<:CosSpace,T,DU,RU,DV,RV}
    df,du,dv = domain(f),domain(u),domain(v)
    @assert df == du == dv && isa(df,PeriodicSegment)
    c = coefficients(f)
    N = 2length(c)-1
    X = zeros(T,N,N)
    X[1,1] += c[1]
    @inbounds for i=2:2:N
        X[i,i] += c[div(i,2)+1]
        X[i+1,i+1] += c[div(i,2)+1]
    end
    ProductFun(X,u⊗v)
end

function convolutionProductFun(f::Fun{S,T},u::Fourier{DU,RU},v::Fourier{DV,RV};tol=eps()) where {S<:SinSpace,T,DU,RU,DV,RV}
    df,du,dv = domain(f),domain(u),domain(v)
    @assert df == du == dv && isa(df,PeriodicSegment)
    c = coefficients(f)
    N = 2length(c)+1
    X = zeros(T,N,N)
    @inbounds for i=2:2:N
        X[i+1,i] += c[div(i,2)]
        X[i,i+1] -= c[div(i,2)]
    end
    ProductFun(X,u⊗v)
end

function convolutionProductFun(f::Fun{Laurent{DS,RS},T},u::Laurent{DU,RU},v::Laurent{DV,RV};tol=eps()) where {DS,RS,T,DU,RU,DV,RV}
    df,du,dv = domain(f),domain(u),domain(v)
    @assert df == du == dv && isa(df,PeriodicSegment)
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

function convolutionProductFun(f::Fun{Taylor{DS,RS},T},u::Laurent{DU,RU},v::Laurent{DV,RV};tol=eps()) where {DS,RS,T,DU,RU,DV,RV}
    df,du,dv = domain(f),domain(u),domain(v)
    @assert df == du == dv && isa(df,PeriodicSegment)
    c = coefficients(f)
    N = 2length(c)-1
    X = zeros(T,N-1,N)
    X[1,1] += c[1]
    @inbounds for i=2:2:N-1
        X[i,i+1] += c[div(i,2)+1]
    end
    ProductFun(X,u⊗v)
end

function convolutionProductFun(f::Fun{Hardy{false,DS,RS},T},u::Laurent{DU,RU},v::Laurent{DV,RV};tol=eps()) where {DS,RS,T,DU,RU,DV,RV}
    df,du,dv = domain(f),domain(u),domain(v)
    @assert df == du == dv && isa(df,PeriodicSegment)
    c = coefficients(f)
    N = 2length(c)
    X = zeros(T,N+1,N)
    @inbounds for i=2:2:N
        X[i+1,i] += c[div(i,2)]
    end
    ProductFun(X,u⊗v)
end
