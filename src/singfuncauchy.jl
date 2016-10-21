## SingFun stieltjes

export logabs
logabs(x) = log(abs2(x))/2

# sqrtx2 is analytic continuation of sqrt(z^2-1)
sqrtx2(z::Number) = sqrt(z-1)*sqrt(z+1)
sqrtx2(x::Real) = sign(x)*sqrt(x^2-1)
@vectorize_1arg Number sqrtx2
function sqrtx2(f::Fun)
    B = Evaluation(first(domain(f)))
    A = Derivative()-f*differentiate(f)/(f^2-1)
    linsolve([B,A],[sqrtx2(first(f))];tolerance=ncoefficients(f)*10E-15)
end

# intervaloffcircle maps the slit plane to the interior(true)/exterior(false) disk
# intervaloncircle maps the interval to the upper(true)/lower(false) half circle

# it is more accurate near infinity to do 1/J_- as it avoids round off
intervaloffcircle(s::Bool,z)=s?1./intervaloffcircle(false,z):(z+sqrtx2(z))
intervaloncircle(s::Bool,x)=x+1.0im*(s?1:-1).*sqrt(1-x).*sqrt(x+1)

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

divkhornersum{S<:Number,T<:Number,U<:Number,V<:Number}(cfs::AbstractVector{S},y::AbstractVector{T},ys::AbstractVector{U},s::V) =
    promote_type(S,T,U,V)[divkhornersum(cfs,y[k],ys[k],s) for k=1:length(y)]
divkhornersum{S<:Number,T<:Number,U<:Number,V<:Number}(cfs::AbstractVector{S},y::AbstractArray{T,2},ys::AbstractArray{U,2},s::V) =
    promote_type(S,T,U,V)[divkhornersum(cfs,y[k,j],ys[k,j],s) for k=1:size(y,1),j=1:size(y,2)]

realdivkhornersum{S<:Real}(cfs::AbstractVector{S},y,ys,s) = real(divkhornersum(cfs,y,ys,s))
realdivkhornersum{S<:Complex}(cfs::AbstractVector{S},y,ys,s) = complex(real(divkhornersum(real(cfs),y,ys,s)),real(divkhornersum(imag(cfs),y,ys,s)))


function stieltjes{S<:PolynomialSpace,DD<:Interval}(sp::JacobiWeight{S,DD},u,zv::Array,s...)
    ret=similar(zv,Complex128)
    for k=1:length(zv)
        @inbounds ret[k]=stieltjes(sp,u,zv[k],s...)
    end
    ret
end

function stieltjes{S<:PolynomialSpace,DD<:Interval}(sp::JacobiWeight{S,DD},u,z)
    d=domain(sp)

    if sp.α == sp.β == .5
        cfs = coefficients(u,sp.space,Ultraspherical(1,d))
        π*hornersum(cfs,intervaloffcircle(true,mobius(sp,z)))
    elseif sp.α == sp.β == -.5
        cfs = coefficients(u,sp.space,ChebyshevDirichlet{1,1}(d))
        z=mobius(sp,z)

        sx2z=sqrtx2(z)
        sx2zi=1./sx2z
        Jm=1./(z+sx2z)  # intervaloffcircle(true,z)


        if length(cfs) ≥1
            ret = π*cfs[1]*sx2zi

            if length(cfs) ≥2
                ret += cfs[2]*π*(z.*sx2zi-1)
            end

            ret - 2π*hornersum(cfs[3:end],Jm)
        else
            zero(z)
        end
    elseif isapproxinteger(sp.α) && isapproxinteger(sp.β)
        stieltjes(sp.space,coefficients(u,sp,sp.space),z)
    else
        if d==Interval()
            S2=JacobiWeight(sp.α,sp.β,Jacobi(sp.β,sp.α))  # convert and then use recurrence
            stieltjesintervalrecurrence(S2,coefficients(u,sp,S2),z)
        else
            # project to interval
            stieltjes(setdomain(sp,Interval()),u,mobius(sp,z))
        end
    end
end


function stieltjes{SS<:PolynomialSpace,DD<:Interval}(sp::JacobiWeight{SS,DD},u,x::Number,s::Bool)
    d=domain(sp)

    if sp.α == sp.β == .5
        cfs=coefficients(u,sp.space,Ultraspherical(1,d))
        π*hornersum(cfs,intervaloncircle(!s,mobius(sp,x)))
    elseif sp.α == sp.β == -.5
        cfs = coefficients(u,sp.space,ChebyshevDirichlet{1,1}(d))
        x=mobius(sp,x)

        if length(cfs) ≥1
            ret = -π*im*cfs[1]*(s?1:-1)/sqrt(1-x^2)

            if length(cfs) ≥2
                ret += cfs[2]*(-π*im*(s?1:-1)*x/sqrt(1-x^2)-π)
            end

            ret - 2π*hornersum(cfs[3:end],intervaloncircle(!s,x))
        else
            0.0+0.0im
        end
    else
        if d==Interval()
            S=JacobiWeight(sp.α,sp.β,Jacobi(sp.β,sp.α))
            cfs=coefficients(u,sp,S)
            cf=stieltjesforward(S,length(cfs),x,s)
            dotu(cf,cfs)
        else
            @assert isa(d,Interval)
            stieltjes(setdomain(sp,Interval()),u,mobius(sp,x),s)
        end
    end
end



##
#  hilbert on JacobiWeight space
#  hilbert is always equal to im*(C^+ + C^-)
#  hilbert(f,z)=im*(cauchy(true,f,z)+cauchy(false,f,z))
##

function hilbert{DD<:Interval}(sp::JacobiWeight{Chebyshev{DD},DD},u)
    d=domain(u)

    if sp.α == sp.β == .5
        # Corollary 5.7 of Olver&Trogdon
        cfs=coefficients(u,sp.space,Ultraspherical(1))
        Fun([0.;-cfs],d)
    elseif sp.α == sp.β == -.5
        # Corollary 5.11 of Olver&Trogdon
        cfs= coefficients(u,sp.space,ChebyshevDirichlet{1,1})
        if length(cfs)≥2
            Fun([cfs[2];2cfs[3:end]],d)
        else
            Fun(zeros(eltype(cfs),1),d)
        end
    else
        error("hilbert only implemented for Chebyshev weights")
    end
end





integratejin(c,cfs,y)=.5*(-cfs[1]*(log(y)+log(c))+divkhornersum(cfs,y,y,1)-divkhornersum(view(cfs,2:length(cfs)),y,zero(y)+1,0))
realintegratejin(c,cfs,y)=.5*(-cfs[1]*(logabs(y)+logabs(c))+realdivkhornersum(cfs,y,y,1)-realdivkhornersum(view(cfs,2:length(cfs)),y,zero(y)+1,0))

#########
# stieltjesintegral is an indefinite integral of stieltjes
# normalized so that there is no constant term
# logkernel is the real part of stieljes normalized by π.
#####

function logkernel{S<:PolynomialSpace,DD<:Interval}(sp::JacobiWeight{S,DD},u,z)
    d=domain(sp)
    a,b=d.a,d.b

    if sp.α == sp.β == .5
        cfs=coefficients(u,sp.space,Ultraspherical(1,d))
        z=mobius(sp,z)
        y = updownjoukowskyinverse(true,z)
        arclength(d)*realintegratejin(4/(b-a),cfs,y)/2
    elseif  sp.α == sp.β == -.5
        cfs = coefficients(u,sp.space,ChebyshevDirichlet{1,1}(d))
        z=mobius(sp,z)
        y = updownjoukowskyinverse(true,z)

        if length(cfs) ≥1
            ret = -cfs[1]*arclength(d)*(logabs(y)+logabs(4/(b-a)))/2

            if length(cfs) ≥2
                ret += -arclength(d)*cfs[2]*real(y)/2
            end

            if length(cfs) ≥3
                ret - arclength(d)*realintegratejin(4/(b-a),view(cfs,3:length(cfs)),y)
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

function stieltjesintegral{S<:PolynomialSpace,DD<:Interval}(sp::JacobiWeight{S,DD},u,z)
    d=domain(sp)
    a,b=d.a,d.b

    if sp.α == sp.β == .5
        cfs=coefficients(u,sp.space,Ultraspherical(1,d))
        y=intervaloffcircle(true,mobius(sp,z))
        π*complexlength(d)*integratejin(4/(b-a),cfs,y)/2
    elseif  sp.α == sp.β == -.5
        cfs = coefficients(u,sp.space,ChebyshevDirichlet{1,1}(d))
        z=mobius(sp,z)
        y=intervaloffcircle(true,z)

        if length(cfs) ≥1
            ret = -cfs[1]*π*complexlength(d)*(log(y)+log(4/abs(b-a)))/2

            if length(cfs) ≥2
                ret += -π*complexlength(d)*cfs[2]*intervaloffcircle(true,z)/2
            end

            if length(cfs) ≥3
                ret - π*complexlength(d)*integratejin(4/abs(b-a),view(cfs,3:length(cfs)),y)
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

function stieltjesintegral{S<:PolynomialSpace,DD<:Interval}(sp::JacobiWeight{S,DD},u,z,s::Bool)
    d=domain(u)
    a,b=d.a,d.b     # TODO: type not inferred right now

    if sp.α == sp.β == .5
        cfs=coefficients(u,sp.space,Ultraspherical(1,d))
        y=intervaloncircle(!s,mobius(sp,z))
        π*complexlength(d)*integratejin(4/(b-a),cfs,y)/2
    elseif  sp.α == sp.β == -.5
        cfs = coefficients(u,sp.space,ChebyshevDirichlet{1,1}(d))
        z=mobius(sp,z)
        y=intervaloncircle(!s,z)

        if length(cfs) ≥1
            ret = -cfs[1]*π*complexlength(d)*(log(y)+log(4/(b-a)))/2

            if length(cfs) ≥2
                ret += -π*complexlength(d)*cfs[2]*intervaloncircle(!s,z)/2
            end

            if length(cfs) ≥3
                ret - π*complexlength(d)*integratejin(4/(b-a),view(cfs,3:length(cfs)),y)
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
