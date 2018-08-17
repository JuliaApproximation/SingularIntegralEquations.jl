#
# PivotLDU is a block LDU decomposition of
# [I B;
#  C I]
# where
#
# size(B) = (r1,r2)
# size(C) = (r2,r1)
#

import LinearAlgebra: Factorization, LU

struct PivotLDU{T,S<:AbstractMatrix} <: Factorization{T}
    B::S
    C::S
    factor::LU{T,S} # LU factorization of I-C*B
    r1::Int
    r2::Int
end

function pivotldufact(A::AbstractMatrix,r1,r2)
    @assert size(A) == (r1+r2,r1+r2)
    B = A[1:r1,r1+1:end]
    C = A[r1+1:end,1:r1]
    PivotLDU(B,C,lu(I-C*B),r1,r2)
end

function ldiv1B2!(P::PivotLDU{T,S},b1::AbstractArray{T},b2::AbstractArray{T}) where {T<:Number,S}
    b2[:] = b2 - P.C*b1
    ldiv!(P.factor,b2)
    b1[:] = b1 - P.B*b2
    b1,b2
end

function ldiv!(P::PivotLDU{T,S},b::AbstractVector{T}) where {T<:Number,S}
    b[1+P.r1:end] -= P.C*b[1:P.r1]
    b[1+P.r1:end] = P.factor\b[1+P.r1:end]
    b[1:P.r1] -= P.B*b[1+P.r1:end]
    b
end

function ldiv!(P::PivotLDU{T,S},b::AbstractMatrix{T}) where {T<:Number,S}
    b[1+P.r1:end,:] .-= P.C*b[1:P.r1,:]
    b[1+P.r1:end,:] = P.factor\b[1+P.r1:end,:]
    b[1:P.r1,:] .-= P.B*b[1+P.r1:end,:]
    b
end
