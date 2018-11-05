export Convolution

#############
# Convolution implements the convolution operator:
#
# \int_Γ G(|x-y|) u(y) dy or dΓ(y), for x ∈ Γ,
#
# where op qualifies the integral operator, and
# where G is supplied as a Fun.
#############

abstract type Convolution{S,BT,T} <: Operator{T} end

struct ConcreteConvolution{S<:Space,BT,T} <: Convolution{S,BT,T}
    op::BT
    G::Fun{S,T}
end

Convolution(op::BT,G::Fun{S,T}) where {S,BT,T} = ConcreteConvolution(op,G)

domain(C::ConcreteConvolution) = domain(C.op)
domainspace(C::ConcreteConvolution) = domainspace(C.op)
rangespace(C::ConcreteConvolution) = UnsetSpace()
bandwidths(C::ConcreteConvolution) = error("No range space attached to Convolution")
getindex(C::ConcreteConvolution,k::Integer,j::Integer) = error("No range space attached to Convolution")

rangespace(C::ConcreteConvolution{Laurent{D},BT}) where {D,BT} = space(C.G)
rangespace(C::ConcreteConvolution{Fourier{D},BT}) where {D,BT} = space(C.G)
rangespace(C::ConcreteConvolution{CosSpace{D},BT}) where {D,BT} = Fourier(domain(C.G))

bandwidths(C::ConcreteConvolution{Laurent{D},BT}) where {D,BT} = 0,0
bandwidths(C::ConcreteConvolution{Fourier{D},BT}) where {D,BT} = 0,0
bandwidths(C::ConcreteConvolution{CosSpace{D},BT}) where {D,BT} = 0,0

function getindex(C::ConcreteConvolution{Fourier{D},ConcreteDefiniteLineIntegral{Fourier{D},T1},T2},k::Integer,j::Integer) where {D,T1,T2}
    T = promote_type(T1,T2)
    if k == j
        if k == 1 && k ≤ ncoefficients(C.G)
            (C.op[1]*C.G.coefficients[k])::T
        elseif isodd(k) && k ≤ ncoefficients(C.G)
            (C.op[1]*C.G.coefficients[k]/2)::T
        elseif iseven(k) && k ≤ ncoefficients(C.G)-1
            (C.op[1]*C.G.coefficients[k+1]/2)::T
        else
            zero(T)
        end
    else
        zero(T)
    end
end

function getindex(C::ConcreteConvolution{CosSpace{D},ConcreteDefiniteLineIntegral{Fourier{D},T1},T2},k::Integer,j::Integer) where {D,T1,T2}
    T = promote_type(T1,T2)
    if k == j
        if k == 1 && k ≤ ncoefficients(C.G)
            (C.op[1]*C.G.coefficients[k])::T
        elseif k ≤ 2ncoefficients(C.G)-1
            (C.op[1]*C.G.coefficients[k÷2+1]/2)::T
        else
            zero(T)
        end
    else
        zero(T)
    end
end

function getindex(C::ConcreteConvolution{Laurent{D},ConcreteDefiniteLineIntegral{Laurent{D},T1},T2},k::Integer,j::Integer) where {D,T1,T2}
    T = promote_type(T1,T2)
    if k == j && k ≤ ncoefficients(C.G)
        (C.op[1]*C.G.coefficients[k])::T
    else
        zero(T)
    end
end

function getindex(C::ConcreteConvolution{Laurent{D},ConcreteSingularIntegral{Laurent{D},OT,T1},T2},k::Integer,j::Integer) where {D,OT,T1,T2}
    T = promote_type(T1,T2)
    c = C.G.coefficients
    Op = C.op
    N = ncoefficients(C.G)
    ret = zero(T)
    if k == j
        if isodd(k)
            for ℓ = 1:2:N
                ret += c[ℓ]*Op[ℓ+k-1,ℓ+k-1]
            end
            for ℓ = 2:2:min(k-1,N)
                ret += c[ℓ]*Op[k-ℓ,k-ℓ]
            end
            for ℓ = (k+1):2:N
                ret += c[ℓ]*Op[ℓ-k+1,ℓ-k+1]
            end
        else
            for ℓ = 2:2:N
                ret += c[ℓ]*Op[ℓ+k,ℓ+k]
            end
            for ℓ = 1:2:min(k-1,N)
                ret += c[ℓ]*Op[k-ℓ+1,k-ℓ+1]
            end
            for ℓ = (k+1):2:N
                ret += c[ℓ]*Op[ℓ-k,ℓ-k]
            end
        end
    end
    ret
end
