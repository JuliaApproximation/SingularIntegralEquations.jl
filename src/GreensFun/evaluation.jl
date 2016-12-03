
for (Func,Len) in ((:(Base.sum),:complexlength),(:linesum,:arclength))
    @eval begin
        function $Func{CC<:Chebyshev,DD}(G::Function,u::Fun{JacobiWeight{CC,DD}},z)
            d,α,β,n=domain(u),u.space.α,u.space.β,2ncoefficients(u)
            vals,t = ichebyshevtransform(pad(u.coefficients,n)),points(d,n)
            if α == β == -0.5
                return 0.5*$Len(d)*map(z->mean(G(z,t).*vals),z)*π
            else
                sp,p = space(u),plan_chebyshevtransform(complex(vals))
                map(z->$Func(Fun(sp,chebyshevtransform(G(z,t).*vals,p))),z)
            end
        end
    end
end

function logkernel{CC<:Chebyshev,DD}(G::Function,u::Fun{JacobiWeight{CC,DD}},z)
    sp,n=space(u),2ncoefficients(u)
    vals,t = ichebyshevtransform(pad(u.coefficients,n)),points(sp,n)
    p = plan_chebyshevtransform(complex(vals))
    return map(z->logkernel(Fun(sp,chebyshevtransform(G(z,t).*vals,p)),z),z)
end

function cauchy{CC<:Chebyshev,DD}(G::Function,u::Fun{JacobiWeight{CC,DD}},z)
    sp,n=space(u),2ncoefficients(u)
    vals,t = ichebyshevtransform(pad(u.coefficients,n)),points(sp,n)
    p = plan_chebyshevtransform(vals)#complex(vals))
    return map(z->cauchy(Fun(sp,chebyshevtransform(G(z,t).*vals,p)),z),z)
end

for TYP in (:Fourier,:Laurent)
    @eval begin
        function Base.sum{DD}(G::Function,u::Fun{$TYP{DD}},z)
            d,n=domain(u),2ncoefficients(u)
            vals,t = values(pad(u,n)),points(d,n)
            if isa(d,Circle)
              return map(z->mean(G(z,t).*vals.*t),z)*2π*im
            else
              return map(z->mean(G(z,t).*vals),z)*arclength(d)
            end
        end

        function linesum{DD}(G::Function,u::Fun{$TYP{DD}},z)
            d,n=domain(u),2ncoefficients(u)
            vals,t = values(pad(u,n)),points(d,n)
            map(z->mean(G(z,t).*vals),z)*arclength(d)
        end
    end
end



function logkernel{DD}(G::Function,sp::Fourier{DD},u,z)
    n=2length(u)
    vals,t = values(pad(Fun(sp,u),n)),points(sp,n)
    p = plan_transform(sp,vals)
    return map(z->logkernel(sp,transform(sp,G(z,t).*vals,p),z),z)
end
logkernel{DD}(G::Function,sp::Laurent{DD},u,z)=logkernel(G,Fun(Fun(sp,u),Fourier),z)

for Func in (:(Base.sum),:linesum,:logkernel,:cauchy)
    @eval begin
        $Func{F<:Fun}(G::Function,u::Vector{F},z)=mapreduce(u->$Func(G,u,z),+,u)
        $Func(G::Function,sp::PiecewiseSpace,u,z)=$Func(G,vec(Fun(sp,u)),z)
        $Func(G::Function,u::Fun,z)=$Func(G,space(u),coefficients(u),z)
        $Func{PS<:PolynomialSpace}(G::Function,sp::JacobiWeight{PS},f,z)=$Func(G,Fun(Fun(sp,f),JacobiWeight(sp.β,sp.α,Chebyshev(domain(sp)))),z)
    end
end

function logkernel{LS,RR<:Arc}(G::Function,sp::Space{LS,RR},u,z)
    n=2length(u)
    vals,t = itransform(sp,pad(u,n)),points(sp,n)
    p = plan_transform(sp,complex(vals))
    return map(z->logkernel(sp,transform(sp,G(z,t).*vals,p),z),z)
end

function linesum{LS,RR<:Arc}(G::Function,sp::Space{LS,RR},u,z)
    n=2length(u)
    vals,t = itransform(sp,pad(u,n)),points(sp,n)
    p = plan_transform(sp,complex(vals))
    return map(z->linesum(sp,transform(sp,G(z,t).*vals,p),z),z)
end
