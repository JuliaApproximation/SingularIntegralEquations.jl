
for (Func,Len) in ((:(Base.sum),:complexlength),(:linesum,:length))
    @eval begin
        function $Func(G::Function,u::Fun{JacobiWeight{Chebyshev}},z)
            d,α,β,n=domain(u),u.space.α,u.space.β,2length(u)
            vals,t = ichebyshevtransform(pad(u.coefficients,n)),points(d,n)
            if α == β == -0.5
                return 0.5*$Len(d)*map(z->mean(G(z,t).*vals),z)*π
            else
                sp,p = space(u),plan_chebyshevtransform(complex(vals))
                map(z->$Func(Fun(chebyshevtransform(G(z,t).*vals,p),sp)),z)
            end
        end
    end
end

function logkernel(G::Function,u::Fun{JacobiWeight{Chebyshev}},z)
    sp,n=space(u),2length(u)
    vals,t = ichebyshevtransform(pad(u.coefficients,n)),points(sp,n)
    p = plan_chebyshevtransform(complex(vals))
    return map(z->logkernel(Fun(chebyshevtransform(G(z,t).*vals,p),sp),z),z)
end

function cauchy(G::Function,u::Fun{JacobiWeight{Chebyshev}},z)
    sp,n=space(u),2length(u)
    vals,t = ichebyshevtransform(pad(u.coefficients,n)),points(sp,n)
    p = plan_chebyshevtransform(vals)#complex(vals))
    return map(z->cauchy(Fun(chebyshevtransform(G(z,t).*vals,p),sp),z),z)
end

function Base.sum{S<:Union(Fourier,Laurent)}(G::Function,u::Fun{S},z)
    d,n=domain(u),2length(u)
    vals,t = values(pad(u,n)),points(d,n)
    if isa(d,Circle)
      return map(z->mean(G(z,t).*vals.*t),z)*2π*im
    else
      return map(z->mean(G(z,t).*vals),z)*length(d)
    end
end

function linesum{S<:Union(Fourier,Laurent)}(G::Function,u::Fun{S},z)
    d,n=domain(u),2length(u)
    vals,t = values(pad(u,n)),points(d,n)
    map(z->mean(G(z,t).*vals),z)*length(d)
end

function logkernel{S<:Union(Fourier,Laurent)}(G::Function,u::Fun{S},z)
    sp,n=space(u),2length(u)
    vals,t = values(pad(u,n)),points(sp,n)
    p = plan_transform(sp,vals)
    return map(z->logkernel(Fun(transform(sp,G(z,t).*vals,p),sp),z),z)
end

for Func in (:(Base.sum),:linesum,:logkernel,:cauchy)
    @eval begin
        $Func{F<:Fun}(G::Function,u::Vector{F},z)=mapreduce(u->$Func(G,u,z),+,u)
        $Func{P<:PiecewiseSpace,T}(G::Function,u::Fun{P,T},z)=$Func(G,vec(u),z)
        $Func{PS<:PolynomialSpace}(G::Function,f::Fun{JacobiWeight{PS}},z)=$Func(G,Fun(f,JacobiWeight(f.space.α,f.space.β,Chebyshev(domain(f)))),z)
    end
end
