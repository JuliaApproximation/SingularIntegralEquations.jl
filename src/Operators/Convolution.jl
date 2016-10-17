export Convolution

#############
# Convolution implements the convolution operator:
#
# \int_Γ G(|x-y|) u(y) dy or dΓ(y), for x ∈ Γ,
#
# where op qualifies the integral operator, and
# where G is supplied as a Fun.
#############

abstract Convolution{S,BT,T} <: Operator{T}

immutable ConcreteConvolution{S<:Space,BT,T} <: Convolution{S,BT,T}
    op::BT
    G::Fun{S,T}
end

Convolution{S,BT,T}(op::BT,G::Fun{S,T}) = ConcreteConvolution(op,G)

domain(C::ConcreteConvolution) = domain(C.op)
domainspace(C::ConcreteConvolution) = domainspace(C.op)
rangespace(C::ConcreteConvolution) = UnsetSpace()
bandinds(C::ConcreteConvolution) = error("No range space attached to Convolution")
getindex(C::ConcreteConvolution,k::Integer,j::Integer) = error("No range space attached to Convolution")

rangespace{D,BT}(C::ConcreteConvolution{Laurent{D},BT}) = space(C.G)
rangespace{D,BT}(C::ConcreteConvolution{Fourier{D},BT}) = space(C.G)
rangespace{D,BT}(C::ConcreteConvolution{CosSpace{D},BT}) = Fourier(domain(C.G))

bandinds{D,BT}(C::ConcreteConvolution{Laurent{D},BT}) = 0,0
bandinds{D,BT}(C::ConcreteConvolution{Fourier{D},BT}) = 0,0
bandinds{D,BT}(C::ConcreteConvolution{CosSpace{D},BT}) = 0,0

function getindex{D,T1,T2}(C::ConcreteConvolution{Fourier{D},ConcreteDefiniteLineIntegral{Fourier{D},T1},T2},k::Integer,j::Integer)
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

function getindex{D,T1,T2}(C::ConcreteConvolution{CosSpace{D},ConcreteDefiniteLineIntegral{Fourier{D},T1},T2},k::Integer,j::Integer)
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

function getindex{D,T1,T2}(C::ConcreteConvolution{Laurent{D},ConcreteDefiniteLineIntegral{Laurent{D},T1},T2},k::Integer,j::Integer)
    T = promote_type(T1,T2)
    if k == j && k ≤ ncoefficients(C.G)
        (C.op[1]*C.G.coefficients[k])::T
    else
        zero(T)
    end
end
