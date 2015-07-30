export hilbert
##
#  hilbert on JacobiWeight space
#  hilbert is always equal to im*(C^+ + C^-)
#  hilbert(f,z)=im*(cauchy(true,f,z)+cauchy(false,f,z))
##

function hilbert(u::Fun{JacobiWeight{Chebyshev}})
    d=domain(u);sp=space(u)

    if sp.α == sp.β == .5
        # Corollary 5.7 of Olver&Trogdon
        uf=Fun(u.coefficients,d)
        cfs=coefficients(uf,Ultraspherical{1})
        Fun([0.;-cfs],d)
    elseif sp.α == sp.β == -.5
        # Corollary 5.11 of Olver&Trogdon
        uf = Fun(u.coefficients,d)
        cfs= coefficients(uf,ChebyshevDirichlet{1,1})
        if length(cfs)≥2
            Fun([cfs[2];2cfs[3:end]],d)
        else
            Fun(zeros(eltype(cfs),1),d)
        end
    else
        error("hilbert only implemented for Chebyshev weights")
    end
end



## hilbert on JacobiWeight space mapped by open curves

function hilbert{C<:Curve,M,BT,T}(f::Fun{MappedSpace{JacobiWeight{M},C,BT},T})
    #project
    c=space(f).space.domain
    fm=Fun(f.coefficients,JacobiWeight(space(f).α,space(f).β))
    q=hilbert(fm)+2im*Fun(x->sum(cauchy(fm,filter(y->!in(y,Interval()),complexroots(c.curve-c.curve[x])))))
    Fun(q.coefficients,MappedSpace(domain(f),space(q)))
end

function hilbert{C<:Curve,M,T,BT}(f::Fun{MappedSpace{JacobiWeight{M},C,BT},T},x::Number)
    #project
    c=space(f).space.domain
    fm=Fun(f.coefficients,JacobiWeight(space(f).α,space(f).β))
    hilbert(fm,tocanonical(f,x))+2im*sum(cauchy(fm,filter(y->!in(y,Interval()),complexroots(c.curve-c.curve[x]))))
end
