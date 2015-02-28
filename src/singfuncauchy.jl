## SingFun cauchy


# sqrtx2 is analytic continuation of sqrt(z^2-1)
sqrtx2(z::Complex)=sqrt(z-1)*sqrt(z+1)
sqrtx2(x::Real)=sign(x)*sqrt(x^2-1)
@vectorize_1arg Number sqrtx2
function sqrtx2(f::Fun)
    B=Evaluation(first(domain(f)))
    A=Derivative()-f*differentiate(f)/(f^2-1)
    linsolve([B,A],sqrtx2(first(f));tolerance=length(f)*10E-15)
end

logabs(x)=log(abs2(x))/2

# intervaloffcircle maps the slit plane to the interior(true)/exterior(false) disk
# intervaloncircle maps the interval to the upper(true)/lower(false) half circle

intervaloffcircle(s::Bool,z)=z-(s?1:-1).*sqrtx2(z)
intervaloncircle(s::Bool,x)=x+1.im*(s?1:-1).*sqrt(1-x).*sqrt(x+1)

intervaloffcircle(s::Int,x)=intervaloffcircle(s==1,x)
intervaloncircle(s::Int,x)=intervaloncircle(s==1,x)

function hornersum(cfs,y0)
    ret=zero(y0)
    y=zero(y0)+1

    for k=1:length(cfs)
        y.*=y0
        ret+=cfs[k]*y
    end

    ret
end

function divkhornersum(cfs,y0,ys,s)
    ret=zero(y0)
    y=ys

    for k=1:length(cfs)
        y.*=y0
        ret+=cfs[k]*y./(k+s)
    end

    ret
end

function realdivkhornersum(cfs,y0,ys,s)
    ret=zero(real(y0))
    y=ys

    for k=1:length(cfs)
        y.*=y0
        ret+=cfs[k]*real(y)./(k+s)
    end

    ret
end


absqrt(a,b,z::Complex)=sqrt(z-a)*sqrt(z-b)
absqrt(a,b,x::Real)=x<a?-sqrt(a-x)*sqrt(b-x):sqrt(x-a)*sqrt(x-b)
absqrt(s::Bool,a,b,z)=(s?1:-1)*im*sqrt(z-a)*sqrt(b-z)
absqrt(s::Int,a,b,z)=absqrt(s==1,a,b,z)



function cauchy(u::Fun{JacobiWeight{Chebyshev}},z::Number)
    sp=space(u)

    if sp.α == sp.β == .5
        cfs = coefficients(u.coefficients,Chebyshev,Ultraspherical{1})
        0.5im*hornersum(cfs,intervaloffcircle(true,tocanonical(u,z)))
    elseif sp.α == sp.β == -.5
        cfs = coefficients(u.coefficients,Chebyshev,ChebyshevDirichlet{1,1})
        z=tocanonical(u,z)


        if length(cfs) ≥1
            ret = cfs[1]*0.5im/sqrtx2(z)

            if length(cfs) ≥2
                ret += cfs[2]*.5im*(z/sqrtx2(z)-1)
            end

            ret - 1.im*hornersum(cfs[3:end],intervaloffcircle(true,z))
        else
            0.0+0.0im
        end
    else
        error("cauchy only implemented for Chebyshev weights")
    end
end


function cauchy(s::Bool,u::Fun{JacobiWeight{Chebyshev}},x::Number)
    sp=space(u)

    if sp.α == sp.β == .5
        cfs=coefficients(u.coefficients,Chebyshev,Ultraspherical{1})
        0.5im*hornersum(cfs,intervaloncircle(!s,tocanonical(u,x)))
    elseif sp.α == sp.β == -.5
        cfs = coefficients(u.coefficients,Chebyshev,ChebyshevDirichlet{1,1})
        x=tocanonical(u,x)

        if length(cfs) ≥1
            ret = cfs[1]*0.5*(s?1:-1)/sqrt(1-x^2)

            if length(cfs) ≥2
                ret += cfs[2]*(0.5*(s?1:-1)*x/sqrt(1-x^2)-.5im)
            end

            ret - 1.im*hornersum(cfs[3:end],intervaloncircle(!s,x))
        else
            0.0+0.0im
        end
    else
        error("cauchy only implemented for Chebyshev weights")
    end
end





integratejin(cfs,y)=.5*(-cfs[1]*(log(y)+log(2))+divkhornersum(cfs,y,y,1)-divkhornersum(slice(cfs,2:length(cfs)),y,zero(y)+1,0))
realintegratejin(cfs,y)=.5*(-cfs[1]*(logabs(y)+log(2))+realdivkhornersum(cfs,y,y,1)-realdivkhornersum(slice(cfs,2:length(cfs)),y,zero(y)+1,0))


realintervaloffcircle(b,z)=real(intervaloffcircle(b,z))

#########
# stieltjesintegral is an indefinite integral of stieltjes
# normalized so that there is no constant term
# logkernel is the real part of stieljes
#####

for (OP,JIN,LOG,IOC) in ((:stieltjesintegral,:integratejin,:log,:intervaloffcircle),(:logkernel,:realintegratejin,:logabs,:realintervaloffcircle))
    @eval function $OP{S<:PolynomialSpace}(u::Fun{JacobiWeight{S}},z)
        d=domain(u)
        a,b=d.a,d.b     # TODO: type not inferred right now
        sp=space(u)

        if sp.α == sp.β == .5
            cfs=coefficients(u.coefficients,sp.space,Ultraspherical{1})
            y=intervaloffcircle(true,tocanonical(u,z))
            0.5π*(b-a)*$JIN(cfs,y)
        elseif  sp.α == sp.β == -.5
            cfs = coefficients(u.coefficients,sp.space,ChebyshevDirichlet{1,1})
            z=tocanonical(u,z)
            y=intervaloffcircle(true,z)

            if length(cfs) ≥1
                ret = -cfs[1]*0.5π*(b-a)*($LOG(y)+log(2))

                if length(cfs) ≥2
                    ret += -0.5π*(b-a)*cfs[2]*$IOC(true,z)
                end

                if length(cfs) ≥3
                    ret - π*(b-a)*$JIN(slice(cfs,3:length(cfs)),y)
                else
                    ret
                end
            else
                zero(z)
            end
        end
    end
end

cauchyintegral(u,z)=im/(2π)*stieltjesintegral(u,z)


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



