## SingFun stieltjes

export logabs
logabs(z) = log(abs2(z))/2
sqrtabs(z) = sqrt(abs(z))

# sqrtx2 is analytic continuation of sqrt(z^2-1)
# with the oriented branch cut [-1,1]
# we need to reverseorientation, as sqrt has branch cut [0,∞)
sqrtx2(z::Directed) = sqrt(reverseorientation(z)-1)*sqrt(z.x+1)
sqrtx2(z::Number) = sqrt(z-1)*sqrt(z+1)
sqrtx2(x::Real) = sign(x)*sqrt(x^2-1)

# these mimc logabs.  They are useful because they are continuous functions
# and are used in logkernel for evaluating on the branch cut
# sqrtx2im is omitted as it's discontinuous on the branch cut
sqrtx2abs(z) = sqrtabs(z-1)*sqrtabs(z+1)
sqrtx2real(z) = sqrtx2abs(z)*cos((angle(z-1)+angle(z+1))/2)
#  real(z̄*sqrt(z-1)*sqrt(z+1))
# sqrtx2abs(z)*abs(z)*cos(angle(z-1)+angle(z+1))/2-angle(z))

x̄sqrtx2real(z) = sqrtx2abs(z)*abs(z)*cos((angle(z-1)+angle(z+1))/2-angle(z))


@vectorize_1arg Number sqrtx2
function sqrtx2(f::Fun)
    B = Evaluation(first(domain(f)))
    A = Derivative()-f*differentiate(f)/(f^2-1)
    linsolve([B,A],[sqrtx2(first(f)),0];tolerance=ncoefficients(f)*10E-15)
end

# these are two inverses of the joukowsky map
# the first maps the slit plane to the inner circle, the second to the outer circle
#
# it is more accurate near infinity to do 1/J_- than z - sqrtx2(z) as it avoids round off
joukowskyinverse(::Type{Val{true}},z) = 1./joukowskyinverse(Val{false},z)
joukowskyinverse(::Type{Val{false}},z) = value(z)+sqrtx2(z)

joukowskyinverseabs(::Type{Val{true}},z) = 1./joukowskyinverseabs(Val{false},z)
joukowskyinverseabs(::Type{Val{false}},z) = sqrt(abs2(z)+2x̄sqrtx2real(z)+sqrtx2abs(z)^2)


joukowskyinversereal(::Type{Val{true}},z) =
    joukowskyinversereal(Val{false},z)/joukowskyinverseabs(Val{false},z)^2
joukowskyinversereal(::Type{Val{false}},z) = real(z)+sqrtx2real(z)




function hornersum{S<:Number,V<:Number}(cfs::AbstractVector{S},y::V)
    N,P = length(cfs),Base.promote_op(*,S,V)
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

divkhornersum{S<:Number,T<:Number,U<:Number,V<:Number}(cfs::AbstractVector{S},y::AbstractVector{T},
                                                       ys::AbstractVector{U},s::V) =
    promote_type(S,T,U,V)[divkhornersum(cfs,y[k],ys[k],s) for k=1:length(y)]
divkhornersum{S<:Number,T<:Number,U<:Number,V<:Number}(cfs::AbstractVector{S},y::AbstractArray{T,2},
                                                       ys::AbstractArray{U,2},s::V) =
    promote_type(S,T,U,V)[divkhornersum(cfs,y[k,j],ys[k,j],s) for k=1:size(y,1),j=1:size(y,2)]

realdivkhornersum{S<:Real}(cfs::AbstractVector{S},y,ys,s) = real(divkhornersum(cfs,y,ys,s))
realdivkhornersum{S<:Complex}(cfs::AbstractVector{S},y,ys,s) = complex(real(divkhornersum(real(cfs),y,ys,s)),
                                                                       real(divkhornersum(imag(cfs),y,ys,s)))


function stieltjes{S<:PolynomialSpace,DD<:Segment}(sp::JacobiWeight{S,DD},u,zv::Array)
    ret=similar(zv,Complex128)
    for k=1:length(zv)
        @inbounds ret[k]=stieltjes(sp,u,zv[k])
    end
    ret
end

function stieltjes{S<:PolynomialSpace,DD<:Segment}(sp::JacobiWeight{S,DD},u,z)
    d=domain(sp)

    if sp.α == sp.β == .5
        cfs = coefficients(u,sp.space,Ultraspherical(1,d))
        π*hornersum(cfs,joukowskyinverse(Val{true},mobius(sp,z)))
    elseif sp.α == sp.β == -.5
        cfs = coefficients(u,sp.space,ChebyshevDirichlet{1,1}(d))
        z=mobius(sp,z)

        sx2z=sqrtx2(z)
        sx2zi=1./sx2z
        Jm=1./(value(z)+sx2z)  # joukowskyinverse(true,z)


        if length(cfs) ≥1
            ret = π*cfs[1]*sx2zi

            if length(cfs) ≥2
                ret += cfs[2]*π*(value(z).*sx2zi-1)
            end

            ret - 2π*hornersum(cfs[3:end],Jm)
        else
            zero(value(z))
        end
    elseif isapproxinteger(sp.α) && isapproxinteger(sp.β)
        stieltjes(sp.space,coefficients(u,sp,sp.space),z)
    else
        if d==Segment()
            S2=JacobiWeight(sp.β,sp.α,Jacobi(sp.β,sp.α))  # convert and then use recurrence
            stieltjesintervalrecurrence(S2,coefficients(u,sp,S2),z)
        else
            # project to interval
            stieltjes(setdomain(sp,Segment()),u,mobius(sp,z))
        end
    end
end



##
#  hilbert on JacobiWeight space
#  hilbert is always equal to im*(C^+ + C^-)
#  hilbert(f,z)=im*(cauchy(true,f,z)+cauchy(false,f,z))
##

function hilbert{DD<:Segment}(sp::JacobiWeight{Chebyshev{DD},DD},u)
    d=domain(u)

    if sp.α == sp.β == .5
        # Corollary 5.7 of Olver&Trogdon
        cfs=coefficients(u,sp.space,Ultraspherical(1))
        Fun(d,[0.;-cfs])
    elseif sp.α == sp.β == -.5
        # Corollary 5.11 of Olver&Trogdon
        cfs= coefficients(u,sp.space,ChebyshevDirichlet{1,1})
        if length(cfs)≥2
            Fun(d,[cfs[2];2cfs[3:end]])
        else
            Fun(d,zeros(eltype(cfs),1))
        end
    else
        error("hilbert only implemented for Chebyshev weights")
    end
end





integratejin(c,cfs,y) =
    0.5*(-cfs[1]*(log(y)+log(c))+divkhornersum(cfs,y,y,1)-divkhornersum(view(cfs,2:length(cfs)),y,zero(y)+1,0))
realintegratejin(c,cfs,y) =
    0.5*(-cfs[1]*(logabs(y)+logabs(c))+realdivkhornersum(cfs,y,y,1)-realdivkhornersum(view(cfs,2:length(cfs)),y,zero(y)+1,0))

#########
# stieltjesintegral is an indefinite integral of stieltjes
# normalized so that there is no constant term
# logkernel is the real part of stieljes normalized by π.
#####



logkernel{SS<:PolynomialSpace,DD<:Segment}(S::JacobiWeight{SS,DD},f,z::AbstractArray) = map(x->logkernel(S,f,x),z)

function logkernel{S<:PolynomialSpace,DD<:Segment}(sp::JacobiWeight{S,DD},u,z)
    d=domain(sp)
    a,b=d.a,d.b

    if sp.α == sp.β == .5
        cfs=coefficients(u,sp.space,Ultraspherical(1,d))
        z=mobius(sp,z)
        x,r = joukowskyinversereal(Val{true},z),joukowskyinverseabs(Val{true},z)
        y = r*exp(im*acos(x/r))  # dummy variable, choice of ± in arg doesn't matter
        arclength(d)*realintegratejin(4/(b-a),cfs,y)/2
    elseif  sp.α == sp.β == -.5
        cfs = coefficients(u,sp.space,ChebyshevDirichlet{1,1}(d))
        z=mobius(sp,z)
        x,r = joukowskyinversereal(Val{true},z),joukowskyinverseabs(Val{true},z)
        y = r*exp(im*acos(x/r))

        if length(cfs) ≥1
            ret = -cfs[1]*arclength(d)*(log(r)+logabs(4/(b-a)))/2

            if length(cfs) ≥2
                ret += -arclength(d)*cfs[2]*x/2
            end

            if length(cfs) ≥3
                ret - arclength(d)*realintegratejin(4/(b-a),view(cfs,3:length(cfs)),y)
            else
                ret
            end
        else
            zero(z)
        end
    elseif domain(sp)==Segment()
        DS=WeightedJacobi(sp.β+1,sp.α+1)
        D=Derivative(DS)[2:end,:]

        f=Fun(Fun(sp,u),WeightedJacobi(sp.β,sp.α))  # convert to Legendre expansion
        uu=D\(f|(2:∞))   # find integral, dropping first coefficient of f

        (f.coefficients[1]*real(logjacobimoment(sp.α,sp.β,z)) + real(stieltjes(uu,z)))/π
    else
        error("other intervals not yet implemented")
    end
end

function stieltjesintegral{S<:PolynomialSpace,DD<:Segment}(sp::JacobiWeight{S,DD},u,z)
    d=domain(sp)
    a,b=d.a,d.b

    if sp.α == sp.β == .5
        cfs=coefficients(u,sp.space,Ultraspherical(1,d))
        y=joukowskyinverse(Val{true},mobius(sp,z))
        π*complexlength(d)*integratejin(4/(b-a),cfs,y)/2
    elseif  sp.α == sp.β == -.5
        cfs = coefficients(u,sp.space,ChebyshevDirichlet{1,1}(d))
        z=mobius(sp,z)
        y=joukowskyinverse(Val{true},z)

        if length(cfs) ≥1
            ret = -cfs[1]*π*complexlength(d)*(log(y)+log(4/abs(b-a)))/2

            if length(cfs) ≥2
                ret += -π*complexlength(d)*cfs[2]*joukowskyinverse(Val{true},z)/2
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
        error("stieltjes integral not implemented for parameters β="*string(sp.β)*", α="*string(sp.α))
    end
end
