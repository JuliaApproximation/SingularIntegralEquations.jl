export hilbert, hilbertinverse
##
#  hilbert and hilbertinverse on JacobiWeight space
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

function hilbertinverse(u::Fun)
    cfs=coefficients(u,Chebyshev)
    if abs(first(cfs)) < 100eps()
        # no singularity
        # invert Corollary 5.7 of Olver&Trogdon
        cfs=coefficients(cfs[2:end],Ultraspherical{1},Chebyshev)
        Fun(cfs,JacobiWeight(.5,.5,Chebyshev(domain(u))))
    else
        # no singularity
        # invert Corollary 5.11 of Olver&Trogdon
        cfs=[0.,cfs[1],.5*cfs[2:end]]
        cfs=coefficients(cfs,ChebyshevDirichlet{1,1},Chebyshev)
        Fun(cfs,JacobiWeight(-.5,-.5,Chebyshev(domain(u))))
    end
end

## hilbert on JacobiWeight space mapped by open curves

function hilbert{M,BT,T}(f::Fun{CurveSpace{JacobiWeight{M},BT},T})
    #project
    c=space(f).space.domain
    fm=Fun(f.coefficients,JacobiWeight(space(f).α,space(f).β))
    q=hilbert(fm)+2im*Fun(x->sum(cauchy(fm,filter(y->!in(y,Interval()),complexroots(c.curve-c.curve[x])))))
    Fun(q.coefficients,MappedSpace(domain(f),space(q)))
end

function hilbert{M,T,BT}(f::Fun{CurveSpace{JacobiWeight{M},BT},T},x::Number)
    #project
    c=space(f).space.domain
    fm=Fun(f.coefficients,JacobiWeight(space(f).α,space(f).β))
    hilbert(fm,tocanonical(f,x))+2im*sum(cauchy(fm,filter(y->!in(y,Interval()),complexroots(c.curve-c.curve[x]))))
end
