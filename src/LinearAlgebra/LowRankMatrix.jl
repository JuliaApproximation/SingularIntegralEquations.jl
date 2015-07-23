

##
# Represent an m x n rank-r matrix
# A = U V^T
##

export LowRankMatrix

type LowRankMatrix{T} <: AbstractMatrix{T}
    U::Matrix{T} # m x r Matrix
    V::Matrix{T} # n x r Matrix
    m::Int
    n::Int
    r::Int

    function LowRankMatrix(U::Matrix{T},V::Matrix{T},m::Int,n::Int,r::Int)
        mu,ru = size(U)
        nv,rv = size(V)
        @assert ru == rv == r
        @assert mu ≤ m
        @assert nv ≤ n
        new(pad(U,m,r),pad(V,n,r),m,n,r)
    end
    LowRankMatrix(U::Matrix{T},V::Matrix{T}) = new(U,V,size(U)[1],size(V)[1],size(V)[2])
end

LowRankMatrix{T}(U::Matrix{T},V::Matrix{T},m::Int,n::Int)=LowRankMatrix{T}(U,V,m,n,size(V)[2])
LowRankMatrix{T}(U::Matrix{T},V::Matrix{T})=LowRankMatrix{T}(U,V)

function LowRankMatrix{S,T}(U::Matrix{S},V::Matrix{T})
    R = promote_type(S,T)
    LowRankMatrix(convert(Matrix{R},U),convert(Matrix{R},V))
end

function LowRankMatrix{T}(A::Matrix{T})
    U,Σ,V=svd(A)
    r=max(1,count(s->s>10eps(T),Σ))
    sqrtΣ = sqrt(Σ)
    for k=1:r
        U[:,k] .*= sqrtΣ[k]
        V[:,k] = conj(V[:,k]).*sqrtΣ[k]
    end
    LowRankMatrix(U[:,1:r],V[:,1:r])
end



Base.eltype{T}(::LowRankMatrix{T})=T
Base.convert{T}(::Type{LowRankMatrix{T}},L::LowRankMatrix) = LowRankMatrix{T}(convert(Matrix{T},L.U),convert(Matrix{T},L.V))
Base.convert{T}(::Type{Matrix{T}},L::LowRankMatrix) = convert(Matrix{T},full(L))
Base.promote_rule{T,V}(::Type{LowRankMatrix{T}},::Type{LowRankMatrix{V}})=LowRankMatrix{promote_type(T,V)}

Base.size(L::LowRankMatrix) = L.m,L.n
Base.rank(L::LowRankMatrix) = L.r
Base.transpose(L::LowRankMatrix) = LowRankMatrix(L.V,L.U)
Base.ctranspose{T<:Real}(L::LowRankMatrix{T}) = LowRankMatrix(L.V,L.U)
Base.ctranspose(L::LowRankMatrix) = LowRankMatrix(conj(L.V),conj(L.U))

Base.getindex(L::LowRankMatrix,i::Int,j::Int) = mapreduce(k->L.U[i,k]*L.V[j,k],+,1:L.r)
Base.getindex{T}(L::LowRankMatrix{T},i::Int,jr::Range) = T[L[i,j] for j=jr].'#'
Base.getindex{T}(L::LowRankMatrix{T},ir::Range,j::Int) = T[L[i,j] for i=ir]
Base.getindex{T}(L::LowRankMatrix{T},ir::Range,jr::Range) = T[L[i,j] for i=ir,j=jr]
Base.full(L::LowRankMatrix)=L[1:L.m,1:L.n]

*{S,T}(L::LowRankMatrix{S},b::AbstractVecOrMat{T}) = L.U*(L.V.'*b)#'
\{S,T}(L::LowRankMatrix{S},b::AbstractVecOrMat{T}) = full(L)\b
