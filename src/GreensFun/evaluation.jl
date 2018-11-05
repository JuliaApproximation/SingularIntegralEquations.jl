
for (Func,Len) in ((:(Base.sum),:complexlength),(:linesum,:arclength))
    @eval begin
        function $Func(G::Function,u::Fun{<:JacobiWeight{<:Chebyshev}},z)
           d,α,β,n=domain(u),u.space.α,u.space.β,2ncoefficients(u)
           vals,t = ichebyshevtransform(pad(u.coefficients,n)),points(d,n)
           sp,p = space(u),plan_chebyshevtransform(complex(vals))
           map(z->$Func(Fun(sp,p*(G.(z,t).*vals))),z)
       end

       # TODO: remove the following hack
        function $Func(G::Function,u::Fun{<:JacobiWeight{<:Chebyshev,<:IntervalOrSegment}},z)
            d,α,β,n=domain(u),u.space.α,u.space.β,2ncoefficients(u)
            vals,t = ichebyshevtransform(pad(u.coefficients,n)),points(d,n)
            if α == β == -0.5
                return 0.5*$Len(d)*map(z->mean(G.(z,t).*vals),z)*π
            else
                sp,p = space(u),plan_chebyshevtransform(complex(vals))
                map(z->$Func(Fun(sp,p*(G.(z,t).*vals))),z)
            end
        end
    end
end

function logkernel(G::Function,u::Fun{<:JacobiWeight{<:Chebyshev}},z)
    sp,n=space(u),2ncoefficients(u)
    vals,t = ichebyshevtransform(pad(u.coefficients,n)),points(sp,n)
    p = plan_chebyshevtransform(complex(vals))
    return map(z->logkernel(Fun(sp,p*(G.(z,t).*vals)),z),z)
end

function cauchy(G::Function,u::Fun{<:JacobiWeight{<:Chebyshev}},z)
    sp,n=space(u),2ncoefficients(u)
    vals,t = ichebyshevtransform(pad(u.coefficients,n)),points(sp,n)
    p = plan_chebyshevtransform(vals)#complex(vals))
    return map(z->cauchy(Fun(sp,p*(G.(z,t).*vals)),z),z)
end

for TYP in (:Fourier,:Laurent)
    @eval begin
        function Base.sum(G::Function,u::Fun{<:$TYP{<:PeriodicSegment}},z)
            d,n=domain(u),2ncoefficients(u)
            vals,t = values(pad(u,n)),points(d,n)

            return map(z->mean(G.(z,t).*vals),z)*arclength(d)
        end

        function Base.sum(G::Function,u::Fun{<:$TYP{<:Circle}},z)
            d,n=domain(u),2ncoefficients(u)
            vals,t = values(pad(u,n)),points(d,n)
            #TODO: shouldn't this depend on which circle?
            return map(z->mean(G.(z,t).*vals.*t),z)*2π*im
        end

        function linesum(G::Function,u::Fun{<:$TYP{<:PeriodicSegment}},z)
            d,n=domain(u),2ncoefficients(u)
            vals,t = values(pad(u,n)),points(d,n)
            map(z->mean(G.(z,t).*vals),z)*arclength(d)
        end
    end
end



function logkernel(G::Function,sp::Fourier{DD,RR},u,z) where {DD,RR}
    n=2length(u)
    vals,t = values(pad(Fun(sp,u),n)),points(sp,n)
    p = plan_transform(sp,vals)
    return map(z->logkernel(sp,transform(sp,G.(z,t).*vals,p),z),z)
end
logkernel(G::Function,sp::Laurent{DD,RR},u,z) where {DD,RR} =
    logkernel(G,Fun(Fun(sp,u),Fourier),z)

for Func in (:(Base.sum),:linesum,:logkernel,:cauchy)
    @eval begin
        $Func(G::Function,u::Vector{F},z) where {F<:Fun} = mapreduce(u->$Func(G,u,z),+,u)
        $Func(G::Function,sp::PiecewiseSpace,u,z) = $Func(G,vec(Fun(sp,u)),z)
        $Func(G::Function,u::Fun,z) = $Func(G,space(u),coefficients(u),z)
        $Func(G::Function,sp::JacobiWeight{PS},f,z) where {PS<:PolynomialSpace} =
            $Func(G,Fun(Fun(sp,f),JacobiWeight(sp.β,sp.α,Chebyshev(domain(sp)))),z)
    end
end

function logkernel(G::Function,sp::Space{LS,RR},u,z) where {LS,RR<:Arc}
    n=2length(u)
    vals,t = itransform(sp,pad(u,n)),points(sp,n)
    p = plan_transform(sp,complex(vals))
    return map(z->logkernel(sp,transform(sp,G.(z,t).*vals,p),z),z)
end

function linesum(G::Function,sp::Space{LS,RR},u,z) where {LS,RR<:Arc}
    n=2length(u)
    vals,t = itransform(sp,pad(u,n)),points(sp,n)
    p = plan_transform(sp,complex(vals))
    return map(z->linesum(sp,transform(sp,G.(z,t).*vals,p),z),z)
end
