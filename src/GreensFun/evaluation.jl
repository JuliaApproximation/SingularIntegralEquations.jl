
for (Func,Len) in ((:(Base.sum),:complexlength),(:linesum,:length))
    @eval begin
        function $Func(G::Function,u::Fun{JacobiWeight{Chebyshev}},z)
            d,α,β,n=domain(u),u.space.α,u.space.β,length(u)#nextpow2(length(u))
            vals,t = ichebyshevtransform(u.coefficients),points(d,n)
            #vals,t = ichebyshevtransform(pad(u.coefficients,n)),points(d,n)
            #p = plan_chebyshevtransform(complex(vals))
            #X = map(z->chebyshevtransform(G(real(z-t),im*imag(z-t)).*vals,p)[1],z)
            if α == β == -0.5
                return 0.5*$Len(d)*map(z->mean(G(real(z-t),im*imag(z-t)).*vals),z)*π
            end
      #=
            if α ≤ -1.0 || β ≤ -1.0
                fs = Fun(f.coefficients,f.space.space)
                return Inf*0.5*$Len(d)*(sign(fs[d.a])+sign(fs[d.b]))/2
            elseif α == β == -0.5
                return 0.5*$Len(d)*f.coefficients[1]*π
            elseif α == β == 0.5
                return 0.5*$Len(d)*(n ≤ 2 ? f.coefficients[1]/2 : f.coefficients[1]/2 - f.coefficients[3]/4)*π
            elseif α == 0.5 && β == -0.5
                return 0.5*$Len(d)*(n == 1 ? f.coefficients[1] : f.coefficients[1] + f.coefficients[2]/2)*π
            elseif α == -0.5 && β == 0.5
                return 0.5*$Len(d)*(n == 1 ? f.coefficients[1] : f.coefficients[1] - f.coefficients[2]/2)*π
            else
                c = zeros(eltype(f),n)
                c[1] = 2.^(α+β+1)*gamma(α+1)*gamma(β+1)/gamma(α+β+2)
                if n > 1
                    c[2] = c[1]*(α-β)/(α+β+2)
                    for i=1:n-2
                        c[i+2] = (2(α-β)*c[i+1]-(α+β-i+2)*c[i])/(α+β+i+2)
                    end
                end
                return 0.5*$Len(d)*dotu(f.coefficients,c)
            end
        =#
        end
        $Func{F<:Fun}(G::Function,u::Vector{F},z)=mapreduce(u->$Func(G,u,z),+,u)
        $Func{P<:PiecewiseSpace,T}(G::Function,u::Fun{P,T},z)=$Func(G,vec(u),z)
        $Func{PS<:PolynomialSpace}(G::Function,f::Fun{JacobiWeight{PS}},z)=$Func(G,Fun(f,Chebyshev(domain(f))),z)
    end
end

function Base.sum{S<:Union(Fourier,Laurent)}(G::Function,u::Fun{S},z)
    d,n=domain(u),length(u)
    vals,t = values(u),points(d,n)
    if isa(d,Circle)
      return map(z->mean(G(real(z-t),im*imag(z-t)).*vals.*t),z)*2π*im
    else
      return map(z->mean(G(real(z-t),im*imag(z-t)).*vals),z)*length(d)
    end
end

function linesum{S<:Union(Fourier,Laurent)}(G::Function,u::Fun{S},z)
    d,n=domain(u),length(u)
    vals,t = values(u),points(d,n)
    map(z->mean(G(real(z-t),im*imag(z-t)).*vals),z)*length(d)
end
