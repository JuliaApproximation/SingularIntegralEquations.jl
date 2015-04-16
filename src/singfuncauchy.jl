## SingFun cauchy

export logabs
logabs(x)=log(abs2(x))/2

# sqrtx2 is analytic continuation of sqrt(z^2-1)
sqrtx2(z::Complex)=sqrt(z-1)*sqrt(z+1)
sqrtx2(x::Real)=sign(x)*sqrt(x^2-1)
@vectorize_1arg Number sqrtx2
function sqrtx2(f::Fun)
    B=Evaluation(first(domain(f)))
    A=Derivative()-f*differentiate(f)/(f^2-1)
    linsolve([B,A],sqrtx2(first(f));tolerance=length(f)*10E-15)
end

# intervaloffcircle maps the slit plane to the interior(true)/exterior(false) disk
# intervaloncircle maps the interval to the upper(true)/lower(false) half circle

intervaloffcircle(s::Bool,z)=z-(s?1:-1).*sqrtx2(z)
intervaloncircle(s::Bool,x)=x+1.im*(s?1:-1).*sqrt(1-x).*sqrt(x+1)

intervaloffcircle(s::Int,x)=intervaloffcircle(s==1,x)
intervaloncircle(s::Int,x)=intervaloncircle(s==1,x)

#TODO: These aren't quite typed correctly but the trouble comes from anticipating the unifying type without checking every element.

updownjoukowskyinverse{T<:Number}(s::Bool,x::T) = in(x,Interval(-one(T),one(T))) ? intervaloncircle(s,x) : intervaloffcircle(s,x)
updownjoukowskyinverse{T<:Number}(s::Bool,x::Vector{T}) = Complex{real(T)}[updownjoukowskyinverse(s,xk) for xk in x]
updownjoukowskyinverse{T<:Number}(s::Bool,x::Array{T,2}) = Complex{real(T)}[updownjoukowskyinverse(s,x[k,j]) for k=1:size(x,1),j=1:size(x,2)]

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

# Mothballed via canonicalization.
#=
absqrt(a,b,z::Complex)=sqrt(z-a)*sqrt(z-b)
absqrt(a,b,x::Real)=x<a?-sqrt(a-x)*sqrt(b-x):sqrt(x-a)*sqrt(x-b)
absqrt(s::Bool,a,b,z)=(s?1:-1)*im*sqrt(z-a)*sqrt(b-z)
absqrt(s::Int,a,b,z)=absqrt(s==1,a,b,z)
=#


function cauchy{S<:PolynomialSpace}(u::Fun{JacobiWeight{S}},z::Number)
    d,sp=domain(u),space(u)

    if sp.α == sp.β == .5
        cfs = coefficients(u.coefficients,sp.space,Ultraspherical{1}(d))
        0.5im*hornersum(cfs,intervaloffcircle(true,tocanonical(u,z)))
    elseif sp.α == sp.β == -.5
        cfs = coefficients(u.coefficients,sp.space,ChebyshevDirichlet{1,1}(d))
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


function cauchy{S<:PolynomialSpace}(s::Bool,u::Fun{JacobiWeight{S}},x::Number)
    d,sp=domain(u),space(u)

    if sp.α == sp.β == .5
        cfs=coefficients(u.coefficients,sp.space,Ultraspherical{1}(d))
        0.5im*hornersum(cfs,intervaloncircle(!s,tocanonical(u,x)))
    elseif sp.α == sp.β == -.5
        cfs = coefficients(u.coefficients,sp.space,ChebyshevDirichlet{1,1}(d))
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





integratejin(c,cfs,y)=.5*(-cfs[1]*(log(y)+log(c))+divkhornersum(cfs,y,y,1)-divkhornersum(slice(cfs,2:length(cfs)),y,zero(y)+1,0))
realintegratejin(c,cfs,y)=.5*(-cfs[1]*(logabs(y)+logabs(c))+realdivkhornersum(cfs,y,y,1)-realdivkhornersum(slice(cfs,2:length(cfs)),y,zero(y)+1,0))

#########
# stieltjesintegral is an indefinite integral of stieltjes
# normalized so that there is no constant term
# logkernel is the real part of stieljes
#####

function logkernel{S<:PolynomialSpace}(u::Fun{JacobiWeight{S}},z)
    d,sp=domain(u),space(u)
    a,b=d.a,d.b     # TODO: type not inferred right now

    if sp.α == sp.β == .5
        cfs=coefficients(u.coefficients,sp.space,Ultraspherical{1}(d))
        z=tocanonical(u,z)
        y = updownjoukowskyinverse(true,z)
        π*length(d)*realintegratejin(4/(b-a),cfs,y)/2
    elseif  sp.α == sp.β == -.5
        cfs = coefficients(u.coefficients,sp.space,ChebyshevDirichlet{1,1}(d))
        z=tocanonical(u,z)
        y = updownjoukowskyinverse(true,z)

        if length(cfs) ≥1
            ret = -cfs[1]*π*length(d)*(logabs(y)+logabs(4/(b-a)))/2

            if length(cfs) ≥2
                ret += -π*length(d)*cfs[2]*real(y)/2
            end

            if length(cfs) ≥3
                ret - π*length(d)*realintegratejin(4/(b-a),slice(cfs,3:length(cfs)),y)
            else
                ret
            end
        else
            zero(z)
        end
    else
        error("logkernel not implemented for parameters "*string(sp.α)*","*string(sp.β))
    end
end

function stieltjesintegral{S<:PolynomialSpace}(u::Fun{JacobiWeight{S}},z)
    d,sp=domain(u),space(u)
    a,b=d.a,d.b     # TODO: type not inferred right now

    if sp.α == sp.β == .5
        cfs=coefficients(u.coefficients,sp.space,Ultraspherical{1}(d))
        y=intervaloffcircle(true,tocanonical(u,z))
        π*complexlength(d)*integratejin(4/(b-a),cfs,y)/2
    elseif  sp.α == sp.β == -.5
        cfs = coefficients(u.coefficients,sp.space,ChebyshevDirichlet{1,1}(d))
        z=tocanonical(u,z)
        y=intervaloffcircle(true,z)

        if length(cfs) ≥1
            ret = -cfs[1]*π*complexlength(d)*(log(y)+log(4/(b-a)))/2

            if length(cfs) ≥2
                ret += -π*complexlength(d)*cfs[2]*intervaloffcircle(true,z)/2
            end

            if length(cfs) ≥3
                ret - π*complexlength(d)*integratejin(4/(b-a),slice(cfs,3:length(cfs)),y)
            else
                ret
            end
        else
            zero(z)
        end
    else
        error("stieltjes integral not implemented for parameters "*string(sp.α)*","*string(sp.β))
    end
end

function stieltjesintegral{S<:PolynomialSpace}(s::Bool,u::Fun{JacobiWeight{S}},z)
    d,sp=domain(u),space(u)
    a,b=d.a,d.b     # TODO: type not inferred right now

    if sp.α == sp.β == .5
        cfs=coefficients(u.coefficients,sp.space,Ultraspherical{1}(d))
        y=intervaloncircle(!s,tocanonical(u,z))
        π*complexlength(d)*integratejin(4/(b-a),cfs,y)/2
    elseif  sp.α == sp.β == -.5
        cfs = coefficients(u.coefficients,sp.space,ChebyshevDirichlet{1,1}(d))
        z=tocanonical(u,z)
        y=intervaloncircle(!s,z)

        if length(cfs) ≥1
            ret = -cfs[1]*π*complexlength(d)*(log(y)+log(4/(b-a)))/2

            if length(cfs) ≥2
                ret += -π*complexlength(d)*cfs[2]*intervaloncircle(!s,z)/2
            end

            if length(cfs) ≥3
                ret - π*complexlength(d)*integratejin(4/(b-a),slice(cfs,3:length(cfs)),y)
            else
                ret
            end
        else
            zero(z)
        end
    else
        error("stieltjes integral not implemented for parameters "*string(sp.α)*","*string(sp.β))
    end
end




## Mapped

function cauchy{M,T,BT}(f::Fun{CurveSpace{JacobiWeight{M},BT},T},z::Number)
    #project
    cs=space(f).space
    fm=Fun(f.coefficients,JacobiWeight(space(f).α,space(f).β,cs.space))
    sum(cauchy(fm,complexroots(cs.domain.curve-z)))
end

function cauchy{M,BT,T}(s::Bool,f::Fun{CurveSpace{JacobiWeight{M},BT},T},z::Number)
    #project
    cs=space(f).space
    fm=Fun(f.coefficients,JacobiWeight(space(f).α,space(f).β,cs.space))
    rts=complexroots(cs.domain.curve-z)
    di=Interval()
    mapreduce(rt->in(rt,di)?cauchy(s,fm,rt):cauchy(fm,rt),+,rts)
end



