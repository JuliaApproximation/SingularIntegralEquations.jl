

## SingFun cauchy

intervaloffcircle(s::Bool,x)=x.-(s?1:-1).*sqrt(x.-1).*sqrt(x.+1)
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



function cauchy(u::SingFun,z)
    @assert typeof(u.fun.domain) <: Interval
    
    if u.α == u.β == .5    
        0.5im*holdersum(coefficients(u.fun,1),
                        intervaloffcircle(true,tocanonical(u,z)))
    elseif u.α == u.β == -.5
        cfs = dirichlettransform(u.fun.coefficients)        
        z=tocanonical(u,z)
        
        
        if length(cfs) >=1
            ret = cfs[1]*0.5im/sqrt(z^2 - 1)
        
            if length(cfs) >=2
                ret += cfs[2]*(0.5im*z/sqrt(z^2 - 1)-.5im)
            end
        
            ret - 1.im*holdersum(cfs[3:end],intervaloffcircle(true,z))  
        else
            0.
        end
    else
        error("Cauchy only implemented for Chebyshev weights")
    end
end



function cauchy(s::Bool,u::SingFun,x)
    @assert typeof(u.fun.domain) <: Interval    

    if u.α == u.β == .5
        0.5im*holdersum(coefficients(u.fun,1),
                        intervaloncircle(!s,tocanonical(u,x)))    
    elseif u.α == u.β == -.5
        cfs = dirichlettransform(u.fun.coefficients)
        x=tocanonical(u,x)
        
        if length(cfs) >=1
            ret = cfs[1]*0.5*(s?1:-1)/sqrt(1-x^2 )
        
            if length(cfs) >=2
                ret += cfs[2]*(0.5*(s?1:-1)*x/sqrt(1-x^2)-.5im)
            end
            
            ret - 1.im*holdersum(cfs[3:end],intervaloncircle(!s,x))          
        else
            0.
        end    
    else
        error("Cauchy only implemented for Chebyshev weights")
    end
end

## hilbert is equal to im*(C^+ + C^-)

function hilbert(u::SingFun)
    if u.α == u.β == .5 
        IFun([0.,-coefficients(u.fun,1)],u.fun.domain)
    elseif u.α == u.β == -.5 
        cfs = dirichlettransform(u.fun.coefficients)
        
        IFun([cfs[2],2cfs[3:end]],u.fun.domain)
    else
        error("Cauchy only implemented for Chebyshev weights")    
    end
end


function hilbertinverse(u::IFun)
    if abs(u.coefficients[1]) < 100eps()
        ## no singularity
        SingFun(IFun(ultraiconversion!(u.coefficients[2:end]),u.domain),.5,.5)
    else
        SingFun(IFun(idirichlettransform!([0.,u.coefficients[1],.5*u.coefficients[2:end]]),u.domain),-.5,-.5)
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

function cauchyintegral(u::SingFun,z)
    @assert u.α == u.β == .5     
    
    a,b=u.fun.domain.a,u.fun.domain.b
    
    @assert a==-1.&&b==1.
    
    cfs=coefficients(u.fun,1)
    
    y=intervaloffcircle(true,z)
    
    .25im*(-cfs[1]*log(y)+ divkholdersum(cfs,y,y,1)-divkholdersum(cfs[2:end],y,one(z),0))
end






