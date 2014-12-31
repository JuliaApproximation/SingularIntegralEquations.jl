export hilbert, hilbertinverse

## hilbert is equal to im*(C^+ + C^-)
## hilbert and hilbertinverse on JacobiWeight space

function hilbert(u::Fun{JacobiWeight{Chebyshev}})
    d=domain(u);sp=space(u)

    if sp.α == sp.β == .5
        uf=Fun(u.coefficients,d)
        Fun([0.,-coefficients(uf,Ultraspherical{1})],d)
    elseif sp.α == sp.β == -.5
        cfs = dirichlettransform(u.coefficients)

        Fun([cfs[2],2cfs[3:end]],d)
    else
        error("hilbert only implemented for Chebyshev weights")
    end
end

function hilbertinverse(u::Fun)
    if abs(u.coefficients[1]) < 100eps()
        ## no singularity
        Fun(ultraiconversion!(u.coefficients[2:end]),JacobiWeight(.5,.5,u.domain))
    else
        Fun(idirichlettransform!([0.,u.coefficients[1],.5*u.coefficients[2:end]]),JacobiWeight(-.5,-.5,u.domain))
    end
end

## hilbert on JacobiWeight space mapped by open curves

function hilbert{M,T}(f::Fun{JacobiWeight{OpenCurveSpace{M}},T})
    #project
    c=space(f).space.domain
    fm=Fun(f.coefficients,JacobiWeight(space(f).α,space(f).β))
    q=hilbert(fm)+2im*Fun(x->sum(cauchy(fm,filter(y->!in(y,Interval()),complexroots(c.curve-c.curve[x])))))
    Fun(q.coefficients,MappedSpace(domain(f),space(q)))
end

function hilbert{M,T}(f::Fun{JacobiWeight{OpenCurveSpace{M}},T},x::Number)
    #project
    c=space(f).space.domain
    fm=Fun(f.coefficients,JacobiWeight(space(f).α,space(f).β))
    hilbert(fm,tocanonical(f,x))+2im*sum(cauchy(fm,filter(y->!in(y,Interval()),complexroots(c.curve-c.curve[x]))))
end
