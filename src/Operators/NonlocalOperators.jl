export NonlocalLaplacian

#############
# NonlocalLaplacian implements the nonlocal integral operator:
#
#        𝓛 u(x) := \int_{B_δ(x)} ρ_δ(|x-y|) [u(y) - u(x)] dy
#
# where B_δ(x) = {y ∈ ℝᵈ : |x-y| ≤ δ} is the closed ball of radius 0 < δ < ∞,
# and the spherically symmetric kernel ρ_δ is defined by:
#
#       ρ_δ(r) = 2*Γ(d/2+1)*(d+2-α)/(π^{d/2} δ^{d+2-α}) χ_[0,δ](r)/r^α,
#
# where χ_[0,δ](⋅) is the indicator function, and 0 ≤ α < d + 2.
#############

abstract type AbstractNonlocalCalculusOperator{S, T} <: Operator{T} end

abstract type NonlocalLaplacian{S, T} <: AbstractNonlocalCalculusOperator{S, T} end

struct ConcreteNonlocalLaplacian{S, T} <: NonlocalLaplacian{S, T}
    space::S
    α::T
    δ::T
end

ConcreteNonlocalLaplacian(sp::Space, α::T, δ::T) where T = ConcreteNonlocalLaplacian{typeof(sp), T}(sp, α, δ)

ConcreteNonlocalLaplacian(sp::Space, α, δ) = ConcreteNonlocalLaplacian{typeof(sp), promote_type(typeof(α), typeof(δ))}(sp, α, δ)

NonlocalLaplacian(sp::Space, α, δ) = ConcreteNonlocalLaplacian(sp, α, δ)
NonlocalLaplacian(d::Domain, x...) = NonlocalLaplacian(Space(d), x...)

domain(L::ConcreteNonlocalLaplacian) = domain(L.space)

domainspace(L::ConcreteNonlocalLaplacian) = L.space

getindex(::ConcreteNonlocalLaplacian{UnsetSpace,T}, k::Integer, j::Integer) where T =
    error("Spaces cannot be inferred for operator")
rangespace(L::ConcreteNonlocalLaplacian{UnsetSpace,T}) where T = UnsetSpace()

rangespace(L::ConcreteNonlocalLaplacian{Hardy{s,DD,RR}}) where {s,DD,RR} = L.space
rangespace(L::ConcreteNonlocalLaplacian{Laurent{DD,RR}}) where {DD,RR} = L.space
rangespace(L::ConcreteNonlocalLaplacian{Fourier{DD,RR}}) where {DD,RR} = L.space
rangespace(L::ConcreteNonlocalLaplacian{CosSpace{DD,RR}}) where {DD,RR} = L.space
rangespace(L::ConcreteNonlocalLaplacian{SinSpace{DD,RR}}) where {DD,RR} = L.space

bandwidths(::ConcreteNonlocalLaplacian{Hardy{s,DD,RR}}) where {s,DD,RR} = 0,0
bandwidths(::ConcreteNonlocalLaplacian{Laurent{DD,RR}}) where {DD,RR} = 0,0
bandwidths(::ConcreteNonlocalLaplacian{Fourier{DD,RR}}) where {DD,RR} = 0,0
bandwidths(::ConcreteNonlocalLaplacian{CosSpace{DD,RR}}) where {DD,RR} = 0,0
bandwidths(::ConcreteNonlocalLaplacian{SinSpace{DD,RR}}) where {DD,RR} = 0,0

##TODO: Add scale for different periods.
function getindex(L::ConcreteNonlocalLaplacian{Hardy{true, DD, RR}, T},
                  k::Integer, j::Integer) where {DD <: PeriodicSegment, RR, T}
    if k == j
        fourier_lambda(k-1, L.α, L.δ, 1)
    else
        zero(T)
    end
end

function getindex(L::ConcreteNonlocalLaplacian{Hardy{false, DD, RR}, T},
                  k::Integer, j::Integer) where {DD <: PeriodicSegment, RR, T}
    if k == j
        fourier_lambda(k, L.α, L.δ, 1)
    else
        zero(T)
    end
end

for SP in (:Laurent, :Fourier)
    @eval begin
        function getindex(L::ConcreteNonlocalLaplacian{$SP{DD, RR}, T},
                          k::Integer, j::Integer) where {DD <: PeriodicSegment, RR, T}
            if k == j
                fourier_lambda(k÷2, L.α, L.δ, 1)
            else
                zero(T)
            end
        end
    end
end

function getindex(L::ConcreteNonlocalLaplacian{CosSpace{DD, RR}, T},
                  k::Integer, j::Integer) where {DD <: PeriodicSegment, RR, T}
    if k == j
        fourier_lambda(k-1, L.α, L.δ, 1)
    else
        zero(T)
    end
end

function getindex(L::ConcreteNonlocalLaplacian{SinSpace{DD, RR}, T},
                  k::Integer, j::Integer) where {DD <: PeriodicSegment, RR, T}
    if k == j
        fourier_lambda(k, L.α, L.δ, 1)
    else
        zero(T)
    end
end


function fourier_lambda(k, α, δ, d::Integer)
    d < 1 && error("The $d-dimensional Fourier spectrum is absurd.")
    0 ≤ α < d + 2 || error("The algebraic singularity strength α = $α is not permissible.")
    if abs(k*δ) < 6
        fourier_lambda_small(k, α, δ, d)
    else
        fourier_lambda_large(k, α, δ, d)
    end
end

fourier_lambda_small(k::Real, α::Real, δ::Real, d::Integer) = fourier_lambda_small(promote(k, α, δ)..., d)

function fourier_lambda_small(k::T, α::T, δ::T, d::Integer) where T <: Real
    z = -(k*δ/2)^2
    C = (d+2-α)/2
    D = 1+T(d)/2
    S₀, S₁, err, j = one(T), one(T) + C/(C+one(T))*z/(2*D), one(T), 1
    while err > 10*abs(S₀)*eps(T)
        rⱼ = ((C+j)/(C+j+1))/((D+j)*(j+2))
        S₀, S₁ = S₁, S₁ + (S₁ - S₀)*rⱼ*z
        err = abs(S₁-S₀)
        j += 1
    end
    return -k^2*S₁
end

fourier_lambda_large(k::Real, α::Real, δ::Real, d::Integer) = fourier_lambda_large(promote(k, α, δ)..., d)

function fourier_lambda_large(k::T, α::T, δ::T, d::Integer) where T <: Real
    if norm(d+2-α) < eps(T)
        return T(-k^2)
    else
        kδ = k*δ
        Td = T(d)
        scl = (d+2-α)*gamma(Td/2+1)*2/δ^2
        cst = multif((d-α)/2, 2/kδ, Td/2)/gamma(Td/2)
        tail = 2^(Td/2)*(kδ)^(α+1-d)*((d-2-α)*besselj((Td-2)/2, kδ)*lommelS2((d-2-2α)/2, (Td-4)/2, kδ) - besselj((Td-4)/2, kδ)*lommelS2((d-2α)/2, (Td-2)/2, kδ))
        return scl*(cst+tail)
    end
end

function multif(x, y, z)
    (y^(2x)*gamma(x+1)*gamma(z)/gamma(z-x)-1)/x
end

function multif(x::Float64, y::Float64, z::Float64)
    if x ≠ 0.0
        w = 2x*log(y) + HypergeometricFunctions.lanczosapprox(1.0, x) - HypergeometricFunctions.lanczosapprox(z, -x)
        return expm1(w)/x
    else
        return 2*log(y) - MathConstants.γ + digamma(z)
    end
end

function lommelS2(μ, ν, z)
    z^(μ-1)*drummond((1-μ+ν)/2, (1-μ-ν)/2, (z/2)^2)
end

function drummond(α::T, β::T, z::T) where T
    Nlo = one(T)
    Dlo = one(T)
    Tlo = Nlo/Dlo

    cst = (α+1)*(β+1)
    if norm(cst) < eps(real(T))
        return Tlo
    end
    Nmid = (z+α+β+1)/cst
    Dmid = (z+cst)/cst
    Tmid = Nmid/Dmid

    cst = (α+2)*(β+2)
    if norm(cst) < eps(real(T))
        return Tmid
    end
    Dhi = ((z+α+β+3+cst)*Dmid - (α+β+3)*Dlo)/cst
    Nhi = ((z+α+β+3+cst)*Nmid - (α+β+3)*Nlo)/cst
    Thi = Nhi/Dhi

    k = 0
    while (abs(Thi-Tmid) > 10*abs(Thi)*eps(real(T)) || abs(Tmid-Tlo) > 10*abs(Tmid)*eps(real(T))) && k < 10_000
        cst = (α+k+3)*(β+k+3)
        Nhi, Nmid, Nlo = ((z+(k+2)*(α+β+2*k+5)+cst)*Nhi - (k+2)*(α+β+3*k+6)*Nmid + (k+2)*(k+1)*Nlo)/cst, Nhi, Nmid
        Dhi, Dmid, Dlo = ((z+(k+2)*(α+β+2*k+5)+cst)*Dhi - (k+2)*(α+β+3*k+6)*Dmid + (k+2)*(k+1)*Dlo)/cst, Dhi, Dmid
        Thi, Tmid, Tlo = Nhi/Dhi, Thi, Tmid
        k += 1
    end
    return isnan(Thi) ? isnan(Tmid) ? Tlo : Tmid : Thi
end
