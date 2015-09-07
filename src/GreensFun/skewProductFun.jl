export skewProductFun, skewpoints, skewtransform!, iskewtransform!

#
# These are bivariate constructors that avoid sampling the diagonal y = x
# in a TensorSpace of two of the same spaces.
#

function skewProductFun(f::Function,sp::TensorSpace{@compat(Tuple{Chebyshev,Chebyshev})};tol=100eps())
    for logn = 4:10
        X = coefficients(skewProductFun(f,sp,2^logn,2^logn+1;tol=tol))
        if size(X,1)<2^logn && size(X,2)<2^logn+1
            return ProductFun(X,sp;tol=tol,chopping=true)
        end
    end
    warn("Maximum grid size of ("*string(2^11)*","*string(2^11+1)*") reached")
    skewProductFun(f,sp,2^11,2^11+1;tol=tol)
end

function skewProductFun(f::Function,sp::TensorSpace{@compat(Tuple{Laurent,Laurent})};tol=100eps())
    for logn = 4:10
        X = coefficients(skewProductFun(f,sp,2^logn,2^logn;tol=tol))
        if size(X,1)<2^logn && size(X,2)<2^logn
            return ProductFun(X,sp;tol=tol,chopping=true)
        end
    end
    warn("Maximum grid size of ("*string(2^11)*","*string(2^11)*") reached")
    skewProductFun(f,sp,2^11,2^11;tol=tol)
end

function skewProductFun(f::Function,S::TensorSpace,M::Integer,N::Integer;tol=100eps())
    xy = ApproxFun.checkpoints(S)
    T = promote_type(eltype(f(first(xy)...)),eltype(S))
    ptsx,ptsy=skewpoints(S,M,N)
    vals=T[f(ptsx[k,j],ptsy[k,j]) for k=1:size(ptsx,1), j=1:size(ptsx,2)]
    ProductFun(skewtransform!(S,vals),S;tol=tol,chopping=true)
end

function skewtransform!{T}(S::TensorSpace{@compat(Tuple{Chebyshev,Chebyshev})},M::Matrix{T};kindx::Int=1,kindy::Int=2)
    n=size(M,1)

    planc=plan_chebyshevtransform(M[:,1];kind=kindx)
    for k=1:size(M,2)
        M[:,k]=chebyshevtransform(M[:,k],planc;kind=kindx)
    end

    planr=plan_chebyshevtransform(vec(M[1,:]);kind=kindy)
    for k=1:n
        M[k,:]=chebyshevtransform(vec(M[k,:]),planr;kind=kindy)
    end
    M
end

function iskewtransform!{T}(S::TensorSpace{@compat(Tuple{Chebyshev,Chebyshev})},M::Matrix{T};kindx::Int=1,kindy::Int=2)
    n=size(M,1)

    planc=plan_ichebyshevtransform(M[:,1];kind=kindx)
    for k=1:size(M,2)
        M[:,k]=ichebyshevtransform(M[:,k],planc;kind=kindx)
    end

    planr=plan_ichebyshevtransform(vec(M[1,:]);kind=kindy)
    for k=1:n
        M[k,:]=ichebyshevtransform(vec(M[k,:]),planr;kind=kindy)
    end
    M
end

function skewtransform!{T}(S::TensorSpace{@compat(Tuple{Laurent,Laurent})},M::Matrix{T})
    n=size(M,1)

    planc=plan_transform(S[1],M[:,1])
    for k=1:size(M,2)
        M[:,k]=transform(S[1],M[:,k],planc)
    end

    planr=plan_transform(S[2],vec(M[1,:]))
    for k=1:n
        M[k,:]=process!(transform(S[2],vec(M[k,:]),planr))
    end
    M
end

function iskewtransform!{T}(S::TensorSpace{@compat(Tuple{Laurent,Laurent})},M::Matrix{T})
    n=size(M,1)

    planc=plan_itransform(S[1],M[:,1])
    for k=1:size(M,2)
        M[:,k]=itransform(S[1],M[:,k],planc)
    end

    planr=plan_itransform(S[2],vec(M[1,:]))
    for k=1:n
        M[k,:]=itransform(S[2],iprocess!(vec(M[k,:])),planr)
    end
    M
end

skewpoints(d::TensorSpace,n,m;kwds...) = skewpoints(d,n,m,1;kwds...),skewpoints(d,n,m,2;kwds...)

function skewpoints(d::TensorSpace{@compat(Tuple{Chebyshev,Chebyshev})},n,m,k;kindx::Int=1,kindy::Int=2)
    ptsx=fromcanonical(d[1],chebyshevpoints(n;kind=kindx))
    ptsy=fromcanonical(d[2],chebyshevpoints(m;kind=kindy))
    promote_type(eltype(ptsx),eltype(ptsy))[fromcanonical(d,x,y)[k] for x in ptsx, y in ptsy]
end

function skewpoints(d::TensorSpace{@compat(Tuple{Laurent,Laurent})},n,m,k)
    ptsx=fromcanonical(d[1],fourierpoints(n))
    ptsy=fromcanonical(d[2],fourierpoints(m)+π/m)
    promote_type(eltype(ptsx),eltype(ptsy))[fromcanonical(d,x,y)[k] for x in ptsx, y in ptsy]
end

function process!{T}(c::Vector{T})
    N = length(c)
    twiddle = exp(im*π/N)
    twid = one(T)
    for n=2:2:N-1
        twid *= twiddle
        c[n] *= twid
        c[n+1] /= twid
    end
    if mod(N,2) == 0
        twid *= twiddle
        c[N] *= twid
    end
    c
end

function iprocess!{T}(c::Vector{T})
    N = length(c)
    twiddle = exp(-im*π/N)
    twid = one(T)
    for n=2:2:N-1
        twid *= twiddle
        c[n] *= twid
        c[n+1] /= twid
    end
    if mod(N,2) == 0
        twid *= twiddle
        c[N] *= twid
    end
    c
end
