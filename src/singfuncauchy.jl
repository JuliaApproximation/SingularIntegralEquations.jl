

## SingFun cauchy

intervaloffcircle(s::Bool,z::Complex)=z.-(s?1:-1).*sqrt(z.-1).*sqrt(z.+1)
intervaloffcircle(s::Bool,x::Real)=x<0?x.+(s?1:-1).*sqrt(x.^2-1):x.-(s?1:-1).*sqrt(x.^2-1)
intervaloncircle(s::Bool,x)=x.+1.im*(s?1:-1).*sqrt(1.-x).*sqrt(x.+1)
intervaloffcircle(s::Int,x)=intervaloffcircle(s==1,x)
intervaloncircle(s::Int,x)=intervaloncircle(s==1,x)


function holdersum(cfs,y0)
    ret=zero(y0)
    y=one(y0)

    for k=1:length(cfs)
        y*=y0
        ret+=cfs[k]*y
    end
    
    ret
end

absqrt(a,b,z::Complex)=sqrt(z-a)*sqrt(z-b)
absqrt(a,b,x::Real)=x<a?-sqrt(a-x)*sqrt(b-x):sqrt(x-a)*sqrt(x-b)
absqrt(s::Bool,a,b,z)=(s?1:-1)*im*sqrt(z-a)*sqrt(b-z)
absqrt(s::Int,a,b,z)=absqrt(s==1,a,b,z)


function cauchy(u::Fun{JacobiWeightSpace{ChebyshevSpace}},z::Number)
    d=domain(u);sp=space(u)
    
    if sp.α == sp.β == .5    
        uf=Fun(u.coefficients,ChebyshevSpace(d))
        0.5im*holdersum(coefficients(uf,UltrasphericalSpace{1}),
                        intervaloffcircle(true,tocanonical(u,z)))
    elseif sp.α == sp.β == -.5
        cfs = dirichlettransform(u.coefficients)        
        z=tocanonical(u,z)
        
        
        if length(cfs) >=1
            ret = cfs[1]*0.5im/absqrt(-1,1,z)
        
            if length(cfs) >=2
                ret += cfs[2]*(0.5im*z/absqrt(-1,1,z)-.5im)
            end
        
            ret - 1.im*holdersum(cfs[3:end],intervaloffcircle(true,z))  
        else
            0.0+0.0im
        end
    else
        error("Cauchy only implemented for Chebyshev weights")
    end
end



function cauchy(s::Bool,u::Fun{JacobiWeightSpace{ChebyshevSpace}},x::Number)
    d=domain(u);sp=space(u)

    if sp.α == sp.β == .5
        uf=Fun(u.coefficients,ChebyshevSpace(d))    
        0.5im*holdersum(coefficients(uf,UltrasphericalSpace{1}),
                        intervaloncircle(!s,tocanonical(u,x)))    
    elseif sp.α == sp.β == -.5
        cfs = dirichlettransform(u.coefficients)
        x=tocanonical(u,x)
        
        if length(cfs) >=1
            ret = cfs[1]*0.5*(s?1:-1)/sqrt(1-x^2 )
        
            if length(cfs) >=2
                ret += cfs[2]*(0.5*(s?1:-1)*x/sqrt(1-x^2)-.5im)
            end
            
            ret - 1.im*holdersum(cfs[3:end],intervaloncircle(!s,x))          
        else
            0.0+0.0im
        end    
    else
        error("cauchy only implemented for Chebyshev weights")
    end
end

## hilbert is equal to im*(C^+ + C^-)

function hilbert(u::Fun{JacobiWeightSpace{ChebyshevSpace}})
    d=domain(u);sp=space(u)

    if sp.α == sp.β == .5 
        uf=Fun(u.coefficients,d)        
        Fun([0.,-coefficients(uf,UltrasphericalSpace{1})],d)
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
        Fun(ultraiconversion!(u.coefficients[2:end]),JacobiWeightSpace(.5,.5,u.domain))
    else
        Fun(idirichlettransform!([0.,u.coefficients[1],.5*u.coefficients[2:end]]),JacobiWeightSpace(-.5,-.5,u.domain))
    end
end



## cauchy integral


function divkholdersum(cfs,y0,ys,s)
    ret=zero(y0)
    y=ys

    for k=1:length(cfs)
        y*=y0
        ret+=cfs[k]*y./(k+s)
    end
    
    ret
end

integratejin(cfs,y)=.5*(-cfs[1]*log(y)+divkholdersum(cfs,y,y,1)-divkholdersum(slice(cfs,2:length(cfs)),y,one(y),0))

function cauchyintegral(u::Fun{JacobiWeightSpace{ChebyshevSpace}},z::Number)
    d=domain(u)
    a,b=d.a,d.b
    sp=space(u)
    
    if sp.α == sp.β == .5     
        uf=Fun(u.coefficients,ChebyshevSpace(d))        
        cfs=coefficients(uf,UltrasphericalSpace{1})
        y=intervaloffcircle(true,tocanonical(u,z))
        
        0.25im*(b-a)*integratejin(cfs,y)
    elseif  sp.α == sp.β == -.5     
        cfs = dirichlettransform(u.coefficients)        
        z=tocanonical(u,z)
        y=intervaloffcircle(true,z)
        
        if length(cfs) >=1
            ret = -cfs[1]*0.25im*(b-a)*log(y)
        
            if length(cfs) >=2
                ret += 0.25im*(b-a)*cfs[2]*(absqrt(-1,1,z)-z)
            end
        
            if length(cfs) >= 3
                ret - 0.5im*(b-a)*integratejin(slice(cfs,3:length(cfs)),y)  
            else
                ret
            end
        else
            0.0+0.0im
        end
    end
end








## Mapped

function cauchy{M,T}(f::Fun{JacobiWeightSpace{CurveSpace{M}},T},z::Number)
    #project
    cs=space(f).space
    fm=Fun(f.coefficients,JacobiWeightSpace(space(f).α,space(f).β,cs.space))
    sum(cauchy(fm,complexroots(cs.domain.curve-z)))
end

function cauchy{M,T}(s::Bool,f::Fun{JacobiWeightSpace{CurveSpace{M}},T},z::Number)
    #project
    cs=space(f).space
    fm=Fun(f.coefficients,JacobiWeightSpace(space(f).α,space(f).β,cs.space))
    rts=complexroots(cs.domain.curve-z)
    di=Interval()
    mapreduce(rt->in(rt,di)?cauchy(s,fm,rt):cauchy(fm,rt),+,rts)
end



function hilbert{M,T}(f::Fun{JacobiWeightSpace{CurveSpace{M}},T})
    #project
    c=space(f).space.domain
    fm=Fun(f.coefficients,JacobiWeightSpace(space(f).α,space(f).β))
    q=hilbert(fm)+2im*Fun(x->sum(cauchy(fm,filter(y->!in(y,Interval()),complexroots(c.curve-c.curve[x])))))
    Fun(q.coefficients,MappedSpace(domain(f),space(q)))
end

function hilbert{M,T}(f::Fun{JacobiWeightSpace{CurveSpace{M}},T},x::Number)
    #project
    c=space(f).space.domain    
    fm=Fun(f.coefficients,JacobiWeightSpace(space(f).α,space(f).β))
    hilbert(fm,tocanonical(f,x))+2im*sum(cauchy(fm,filter(y->!in(y,Interval()),complexroots(c.curve-c.curve[x]))))
end


## Operators

function ApproxFun.Hilbert(S::JacobiWeightSpace{CurveSpace{ChebyshevSpace}},k::Integer)
    @assert k==1
    #TODO: choose dimensions
    m,n=40,40
    c=domain(S)
    Sproj=JacobiWeightSpace(S.α,S.β)
    
    rts=[filter(y->!in(y,Interval()),complexroots(c.curve-c.curve[x])) for x in points(Interval(),n)]
     M=hcat(Vector{Complex{Float64}}[transform(ChebyshevSpace(),Complex{Float64}[sum(cauchy(Fun([zeros(k-1),1.0],Sproj),rt)) 
        for rt in rts]) for k=1:m]...)   
          
    A=SpaceOperator(CompactOperator(2im*M),S,S.space)
    H=SpaceOperator(Hilbert(Sproj),S,S.space)
    H+A
end