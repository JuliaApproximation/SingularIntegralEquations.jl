

## SingFun cauchy

# intervaloffcircle maps the slit plane to the interior(true)/exterior(false) disk
# intervaloncircle maps the interval to the upper(true)/lower(false) half circle

#  analytic continuosation of sqrt(z^2-1)
sqrtx2(z::Complex)=sqrt(z-1).*sqrt(z+1)
sqrtx2(x::Real)=sign(x)*sqrt(x^2-1)
sqrtx2(x::Vector)=map(sqrtx2,x)
function sqrtx2(f::Fun)
    B=Evaluation(space(f),first(domain(f)))
    A=Derivative(space(f))-f*differentiate(f)/(f^2-1)
    linsolve([B,A],sqrtx2(first(f));tolerance=length(f)*10E-15)
end



intervaloffcircle(s::Bool,z)=z-(s?1:-1).*sqrtx2(z)
intervaloncircle(s::Bool,x)=x+1.im*(s?1:-1).*sqrt(1-x).*sqrt(x+1)

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


function cauchy(u::Fun{JacobiWeight{Chebyshev}},z::Number)
    d=domain(u);sp=space(u)

    if sp.α == sp.β == .5
        uf=Fun(u.coefficients,Chebyshev(d))
        0.5im*holdersum(coefficients(uf,Ultraspherical{1}),
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



function cauchy(s::Bool,u::Fun{JacobiWeight{Chebyshev}},x::Number)
    d=domain(u);sp=space(u)

    if sp.α == sp.β == .5
        uf=Fun(u.coefficients,Chebyshev(d))
        0.5im*holdersum(coefficients(uf,Ultraspherical{1}),
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

function cauchyintegral(u::Fun{JacobiWeight{Chebyshev}},z::Number)
    d=domain(u)
    a,b=d.a,d.b
    sp=space(u)

    if sp.α == sp.β == .5
        uf=Fun(u.coefficients,Chebyshev(d))
        cfs=coefficients(uf,Ultraspherical{1})
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

function cauchy{M,T}(f::Fun{JacobiWeight{OpenCurveSpace{M}},T},z::Number)
    #project
    cs=space(f).space
    fm=Fun(f.coefficients,JacobiWeight(space(f).α,space(f).β,cs.space))
    sum(cauchy(fm,complexroots(cs.domain.curve-z)))
end

function cauchy{M,T}(s::Bool,f::Fun{JacobiWeight{OpenCurveSpace{M}},T},z::Number)
    #project
    cs=space(f).space
    fm=Fun(f.coefficients,JacobiWeight(space(f).α,space(f).β,cs.space))
    rts=complexroots(cs.domain.curve-z)
    di=Interval()
    mapreduce(rt->in(rt,di)?cauchy(s,fm,rt):cauchy(fm,rt),+,rts)
end






## Cauchy


function Cauchy(ds::JacobiWeight{Ultraspherical{1}},rs::FunctionSpace)
    @assert ds.α==ds.β==0.5

    x=Fun(identity,rs)
    y=intervaloffcircle(true,tocanonical(ds,x))
    
    ret=Array(typeof(y),300)
    ret[1]=y
    n=1
    l=length(y)-1
    u=0
    
    while norm(ret[n].coefficients)>100eps()
        n+=1
        if n > length(ret)
            # double preallocated ret
            newret=Array(typeof(y),2length(ret))
            newret[1:length(ret)]=ret
            newret=ret
        end
        ret[n]=chop!(y*ret[n-1],100eps())  #will be length 2n-1
        u+=1   # upper bandwidth
        l=max(l,length(ret[n])-n)
    end
    
    M=bazeros(Complex{Float64},n+l,n,l,u)
    for k=1:n,j=1:length(ret[k])
        M[j,k]=0.5im*ret[k].coefficients[j]
    end
    Cauchy(M,ds,rs)
end


