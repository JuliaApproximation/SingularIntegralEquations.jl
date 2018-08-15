##
# cauchymoment implements the moments of the cauchy transform of a space
#  note that it starts from k==1
##


cauchymoment(S...) = stieltjesmoment(S...)/(-2π*im)



stieltjesmoment(S::WeightedJacobi{T,D},n::Int,z) where {T,D} = stieltjesjacobimoment(S.space.a,S.space.b,n,z)
stieltjesmoment(S::WeightedJacobi{T,D},z) where {T,D} = stieltjesjacobimoment(S.space.a,S.space.b,z)

stieltjesmoment(S::WeightedJacobiQ{T,D},n::Int,z) where {T,D} = stieltjesjacobimoment(S.space.a,S.space.b,n,z)
stieltjesmoment(S::WeightedJacobiQ{T,D},z) where {T,D} = stieltjesjacobimoment(S.space.a,S.space.b,z)

stieltjesmoment(S::Jacobi,n::Int,z) = stieltjesjacobimoment(S.a,S.b,n,z)
stieltjesmoment(S::Jacobi,z) = stieltjesjacobimoment(S.a,S.b,z)
stieltjesmoment(S::JacobiQ,n::Int,z) = stieltjesjacobimoment(S.a,S.b,n,z)
stieltjesmoment(S::JacobiQ,z) = stieltjesjacobimoment(S.a,S.b,z)

normalization(n::Int,α::Real,β::Real) = 2^(α+β)*gamma(n+α+1)*gamma(n+β+1)/gamma(2n+α+β+2)
stieltjesjacobimoment(α::Real,β::Real,n::Int,z) =
    (x = 2/(1-z);normalization(n,α,β)*HypergeometricFunctions.mxa_₂F₁(n+1,n+α+1,2n+α+β+2,x))
stieltjesjacobimoment(α::Real,β::Real,z) = stieltjesjacobimoment(α,β,0,z)


function logjacobimoment(α::Real,β::Real,n::Int,z)
    x = 2/(1-z)
    if n == 0
        2normalization(0,α,β)*(log(z-1)-dualpart(_₂F₁(dual(zero(α)+eps(α+β),one(β)),α+1,α+β+2,x)))
        # For testing purposes only, should be equivalent to above within radius of convergence
        #2normalization(0,α,β)*(log(z-1)-(α+1)/(α+β+2)*x.*_₃F₂(α+2,α+β+3,x))
    else
        -2normalization(n,α,β)/n*(-x)^n*_₂F₁(n,n+α+1,2n+α+β+2,x)
    end
end
logjacobimoment(α::Real,β::Real,z) = logjacobimoment(α,β,0,z)


function logabsjacobimoment(α::Real,β::Real,n::Int,z)
    x = 2/(1-z)
    if n == 0
        2normalization(0,α,β)*(logabs(z-1)-real(dualpart(_₂F₁(dual(zero(α)+eps(α+β),one(β)),α+1,α+β+2,x))))
        # For testing purposes only, should be equivalent to above within radius of convergence
        #2normalization(0,α,β)*(log(z-1)-(α+1)/(α+β+2)*x.*_₃F₂(α+2,α+β+3,x))
    else
        -2normalization(n,α,β)/n*real((-x)^n*_₂F₁(n,n+α+1,2n+α+β+2,x))
    end
end

logabsjacobimoment(α::Real,β::Real,z) = logabsjacobimoment(α,β,0,z)



stieltjeslegendremoment(n::Int,z) = stieltjesjacobimoment(zero(real(eltype(z))),zero(real(eltype(z))),n,z)
stieltjeslegendremoment(z) = stieltjeslegendremoment(0,z)

logabslegendremoment(z) = real(z)*logabs((z+1)/(z-1))-imag(z)*angle((z+1)/(z-1))+logabs(z^2-1)-2
                        # real(z*log((z+1)/(z-1)))+logabs(z^2-1)-2

#=
#TODO: this is for x^k but we probably want P_k

#These formulae are from mathematica
function stieltjesjacobimoment(α,β,k::Integer,z)
    if α == β == 0
        return stieltjeslegendremoment(k,z)
    elseif isapprox(α,0) && isapprox(β,0.5)
        if k==1
            return -2*sqrt(2)*(sqrtatansqrt(2/(z-1))-1)
        elseif k==2
            return 2*sqrt(2)*(1/3*(3z-2)-z*sqrtatansqrt(2/(z-1)))
        end
    elseif isapprox(α,0) && isapprox(β,-0.5)
        if k==1
            return 2sqrt(1/(z-1))*atan(sqrt(2/(z-1)))
        elseif k==2
            return z/sqrt(1-z)*log((1+1/sqrt(1-z))*(1-sqrt(2)/sqrt(1-z))/((1-1/sqrt(1-z))*(1+sqrt(2)/sqrt(1-z))))-
                2*sqrt(2)-2z/sqrt(1-z)*asinh(sqrt(-1/z))
        end
    elseif !isapprox(α,0) && isapprox(β,0.)
        return (-1)^k*stieltjesjacobimoment(β,α,k,-z)
    end
    error("stieltjesmoment not implemented for JacobiWeight "*string(α)*string(β))
end




stieltjesmoment{DD}(S::JacobiWeight{Chebyshev{DD},DD},k::Integer,z)=stieltjesjacobimoment(S.β,S.α,k,mobius(S,z))


function stieltjesmoment{T,DD}(S::JacobiWeight{Jacobi{T,DD},DD},k::Integer,z)
    z=mobius(S,z)
    J=S.space

    if k==1
        return stieltjesjacobimoment(S.β,S.α,k,z)
    elseif k==2
        if J.a==J.b
            return (1+J.a)*stieltjesjacobimoment(S.α,S.β,k,z)
        else
            return (J.a-J.b)/2*stieltjesjacobimoment(S.α,S.β,1,z) + (2+J.a+J.b)/2*stieltjesjacobimoment(S.α,S.β,2,z)
        end
    end
    error("stieltjesmoment not implemented for JacobiWeight "*string(S.α)*string(S.β))
end
=#


hilbertmoment(S::Space,k::Integer,x) = -real(stieltjesmoment(S,k,(x)⁺)+stieltjesmoment(S,k,(x)⁻))/(2π)
