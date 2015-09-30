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

# it is more accurate near infinity to do 1/J_- as it avoids round off
intervaloffcircle(s::Bool,z)=s?1./intervaloffcircle(false,z):(z+sqrtx2(z))
intervaloncircle(s::Bool,x)=x+1.im*(s?1:-1).*sqrt(1-x).*sqrt(x+1)

intervaloffcircle(s::Int,x)=intervaloffcircle(s==1,x)
intervaloncircle(s::Int,x)=intervaloncircle(s==1,x)

#TODO: These aren't quite typed correctly but the trouble comes from anticipating the unifying type without checking every element.

updownjoukowskyinverse{T<:Number}(s::Bool,x::T) = in(x,Interval(-one(T),one(T))) ? intervaloncircle(s,x) : intervaloffcircle(s,x)
updownjoukowskyinverse{T<:Number}(s::Bool,x::Vector{T}) = Complex{real(T)}[updownjoukowskyinverse(s,xk) for xk in x]
updownjoukowskyinverse{T<:Number}(s::Bool,x::Array{T,2}) = Complex{real(T)}[updownjoukowskyinverse(s,x[k,j]) for k=1:size(x,1),j=1:size(x,2)]

function hornersum{S<:Number,V<:Number}(cfs::AbstractVector{S},y::V)
    N,P = length(cfs),promote_type(S,V)
    ret = N > 0 ? convert(P,cfs[N]) : zero(P)
    for k=N-1:-1:1
        ret = muladd(y,ret,cfs[k])
    end
    y*ret
end

hornersum{S<:Number,V<:Number}(cfs::AbstractVector{S},y::AbstractVector{V}) = promote_type(S,V)[hornersum(cfs,y[k]) for k=1:length(y)]
hornersum{S<:Number,V<:Number}(cfs::AbstractVector{S},y::AbstractArray{V,2}) = promote_type(S,V)[hornersum(cfs,y[k,j]) for k=1:size(y,1),j=1:size(y,2)]

function divkhornersum{S<:Number,T<:Number,U<:Number,V<:Number}(cfs::AbstractVector{S},y::T,ys::U,s::V)
    N,P = length(cfs),promote_type(S,T,U,V)
    ret = N > 0 ? convert(P,cfs[N]/(N+s)) : zero(P)
    for k=N-1:-1:1
        ret=muladd(y,ret,cfs[k]/(k+s))
    end
    y*ys*ret
end

divkhornersum{S<:Number,T<:Number,U<:Number,V<:Number}(cfs::AbstractVector{S},y::AbstractVector{T},ys::AbstractVector{U},s::V) = promote_type(S,T,U,V)[divkhornersum(cfs,y[k],ys[k],s) for k=1:length(y)]
divkhornersum{S<:Number,T<:Number,U<:Number,V<:Number}(cfs::AbstractVector{S},y::AbstractArray{T,2},ys::AbstractArray{U,2},s::V) = promote_type(S,T,U,V)[divkhornersum(cfs,y[k,j],ys[k,j],s) for k=1:size(y,1),j=1:size(y,2)]

realdivkhornersum{S<:Real}(cfs::AbstractVector{S},y,ys,s) = real(divkhornersum(cfs,y,ys,s))
realdivkhornersum{S<:Complex}(cfs::AbstractVector{S},y,ys,s) = complex(real(divkhornersum(real(cfs),y,ys,s)),real(divkhornersum(imag(cfs),y,ys,s)))


#cauchy{S<:PolynomialSpace}(u::Fun{JacobiWeight{S}},zv::Array)=Complex128[cauchy(u,zv[k,j]) for k=1:size(zv,1), j=1:size(zv,2)]
cauchy{S<:PolynomialSpace,DD}(s::Bool,u::Fun{JacobiWeight{S,DD}},zv::Array)=Complex128[cauchy(s,u,zv[k,j]) for k=1:size(zv,1), j=1:size(zv,2)]

function cauchy{S<:PolynomialSpace,DD}(u::Fun{JacobiWeight{S,DD}},z)
    d,sp=domain(u),space(u)

    if sp.α == sp.β == .5
        cfs = coefficients(u.coefficients,sp.space,Ultraspherical{1}(d))
        0.5im*hornersum(cfs,intervaloffcircle(true,tocanonical(u,z)))
    elseif sp.α == sp.β == -.5
        cfs = coefficients(u.coefficients,sp.space,ChebyshevDirichlet{1,1}(d))
        z=tocanonical(u,z)

        sx2z=sqrtx2(z)
        sx2zi=1./sx2z
        Jm=1./(z+sx2z)  # intervaloffcircle(true,z)


        if length(cfs) ≥1
            ret = cfs[1]*0.5im*sx2zi

            if length(cfs) ≥2
                ret += cfs[2]*.5im*(z.*sx2zi-1)
            end

            ret - 1.im*hornersum(cfs[3:end],Jm)
        else
            zero(z)
        end
    elseif isapproxinteger(sp.α) && isapproxinteger(sp.β)
        cauchy(Fun(u,sp.space),z)
    else
        if domain(u)==Interval()
            cauchyintervalrecurrence(Fun(u,JacobiWeight(sp.α,sp.β,Jacobi(sp.β,sp.α))),z)
        else
            @assert isa(domain(u),Interval)
            cauchy(setdomain(u,Interval()),tocanonical(u,z))
        end
    end
end


function cauchy{SS<:PolynomialSpace,DD}(s::Bool,u::Fun{JacobiWeight{SS,DD}},x::Number)
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
        if domain(u)==Interval()
            S=JacobiWeight(sp.α,sp.β,Jacobi(sp.β,sp.α))
            cfs=coefficients(u,S)
            cf=cauchyforward(s,S,length(cfs),x)
            dotu(cf,cfs)
        else
            @assert isa(domain(u),Interval)
            cauchy(s,setdomain(u,Interval()),tocanonical(u,x))
        end
    end
end



##
#  hilbert on JacobiWeight space
#  hilbert is always equal to im*(C^+ + C^-)
#  hilbert(f,z)=im*(cauchy(true,f,z)+cauchy(false,f,z))
##

function hilbert{DD}(u::Fun{JacobiWeight{Chebyshev{DD},DD}})
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





integratejin(c,cfs,y)=.5*(-cfs[1]*(log(y)+log(c))+divkhornersum(cfs,y,y,1)-divkhornersum(slice(cfs,2:length(cfs)),y,zero(y)+1,0))
realintegratejin(c,cfs,y)=.5*(-cfs[1]*(logabs(y)+logabs(c))+realdivkhornersum(cfs,y,y,1)-realdivkhornersum(slice(cfs,2:length(cfs)),y,zero(y)+1,0))

#########
# stieltjesintegral is an indefinite integral of stieltjes
# normalized so that there is no constant term
# logkernel is the real part of stieljes normalized by π.
#####

function logkernel{S<:PolynomialSpace,DD}(u::Fun{JacobiWeight{S,DD}},z)
    d,sp=domain(u),space(u)
    a,b=d.a,d.b     # TODO: type not inferred right now

    if sp.α == sp.β == .5
        cfs=coefficients(u.coefficients,sp.space,Ultraspherical{1}(d))
        z=tocanonical(u,z)
        y = updownjoukowskyinverse(true,z)
        length(d)*realintegratejin(4/(b-a),cfs,y)/2
    elseif  sp.α == sp.β == -.5
        cfs = coefficients(u.coefficients,sp.space,ChebyshevDirichlet{1,1}(d))
        z=tocanonical(u,z)
        y = updownjoukowskyinverse(true,z)

        if length(cfs) ≥1
            ret = -cfs[1]*length(d)*(logabs(y)+logabs(4/(b-a)))/2

            if length(cfs) ≥2
                ret += -length(d)*cfs[2]*real(y)/2
            end

            if length(cfs) ≥3
                ret - length(d)*realintegratejin(4/(b-a),slice(cfs,3:length(cfs)),y)
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

function stieltjesintegral{S<:PolynomialSpace,DD}(u::Fun{JacobiWeight{S,DD}},z)
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

function stieltjesintegral{S<:PolynomialSpace,DD}(s::Bool,u::Fun{JacobiWeight{S,DD}},z)
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
