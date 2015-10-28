

##
# Represent an m x n rank-r matrix
# A = U V^T
##

import ApproxFun: ⊕
export LowRankMatrix, ⊖

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
LowRankMatrix(a::Number,m::Int,n::Int)=fill!(LowRankMatrix(zeros(eltype(a),m,1),zeros(eltype(a),n,1)),a)

function LowRankMatrix{S,T}(U::Matrix{S},V::Matrix{T})
    R = promote_type(S,T)
    LowRankMatrix(convert(Matrix{R},U),convert(Matrix{R},V))
end

function LowRankMatrix{T}(A::Matrix{T})
    U,Σ,V = svd(A)
    r = refactorsvd!(U,Σ,V)
    LowRankMatrix(U[:,1:r],V[:,1:r])
end

function refactorsvd!{S,T}(U::Matrix{S},Σ::Vector{T},V::Matrix{S})
    conj!(V)
    σmax = Σ[1]
    r=max(1,count(s->s>10eps(T),Σ))
    m,n = size(U,1),size(V,1)
    for k=1:r
        σk = sqrt(Σ[k])
        for i=1:m
            @inbounds U[i,k] *= σk
        end
        for j=1:n
            @inbounds V[j,k] *= σk
        end
    end
    r
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
Base.fill!{T}(L::LowRankMatrix{T}, x::T) = (fill!(L.U, sqrt(abs(x)/L.r));fill!(L.V,sqrt(abs(x)/L.r)/sign(x)); L)

function Base.getindex(L::LowRankMatrix,i::Int,j::Int)
    if 1 ≤ i ≤ L.m && 1 ≤ j ≤ L.n
        ret = zero(eltype(L))
        for k=1:rank(L)
            ret = muladd(L.U[i,k],L.V[j,k],ret)
        end
        return ret
    else
        throw(BoundsError())
    end
end
Base.getindex(L::LowRankMatrix,i::Int,jr::Range) = eltype(L)[L[i,j] for j=jr].'
Base.getindex(L::LowRankMatrix,ir::Range,j::Int) = eltype(L)[L[i,j] for i=ir]
Base.getindex(L::LowRankMatrix,ir::Range,jr::Range) = eltype(L)[L[i,j] for i=ir,j=jr]
Base.full(L::LowRankMatrix)=L[1:size(L,1),1:size(L,2)]

for (op,opformatted) in ((:+,:⊕),(:-,:⊖))
    @eval begin
        $op(L::LowRankMatrix) = LowRankMatrix($op(L.U),L.V)
        function $op(L::LowRankMatrix,M::LowRankMatrix)
            @assert size(L) == size(M)
            LowRankMatrix(hcat(L.U,$op(M.U)),hcat(L.V,M.V))
        end
        $op(L::LowRankMatrix,A::AbstractMatrix) = $op(full(L),A)
        $op(A::AbstractMatrix,L::LowRankMatrix) = $op(L,A)

        $op(a::Bool,L::LowRankMatrix{Bool}) = error("Not callable")
        $op(L::LowRankMatrix{Bool},a::Bool) = error("Not callable")
        $op(a::Number,L::LowRankMatrix) = $op(LowRankMatrix(a,size(L)...),L)
        $op(L::LowRankMatrix,a::Number) = $op(L,LowRankMatrix(a,size(L)...))

        $opformatted(L::LowRankMatrix) = LowRankMatrix($op(L.U),L.V)
        function $opformatted(L::LowRankMatrix,M::LowRankMatrix)
            N = $op(L,M)
            T = eltype(N)
            QU,RU = qr(N.U)
            QV,RV = qr(N.V)
            U,Σ,V = svd(RU*RV.')
            r = refactorsvd!(U,Σ,V)
            LowRankMatrix(QU[:,1:r]*U[1:r,1:r],QV[:,1:r]*V[1:r,1:r])
        end
        $opformatted(L::LowRankMatrix,A::AbstractMatrix) = $op(L,A)
        $opformatted(A::AbstractMatrix,L::LowRankMatrix) = $op(L,A)

        $opformatted(a::Number,L::LowRankMatrix) = $opformatted(LowRankMatrix(a,size(L)...),L)
        $opformatted(L::LowRankMatrix,a::Number) = $opformatted(L,LowRankMatrix(a,size(L)...))
    end
end


function Base.A_mul_B!(b::AbstractVector,L::LowRankMatrix,x::AbstractVector)
    m,n = size(L)
    r = rank(L)
    temp = zeros(promote_type(eltype(L),eltype(x)),r)
    At_mul_B!(temp,L.V,x)
    A_mul_B!(b,L.U,temp)
    b
end

*(a::Number,L::LowRankMatrix) = LowRankMatrix(a*L.U,L.V)
*(L::LowRankMatrix,a::Number) = LowRankMatrix(L.U,L.V*a)
.*(a::Number,L::LowRankMatrix) = LowRankMatrix(a*L.U,L.V)
.*(L::LowRankMatrix,a::Number) = LowRankMatrix(L.U,L.V*a)

function *(L::LowRankMatrix,M::LowRankMatrix)
    m1,n1 = size(L)
    r1 = rank(L)
    m2,n2 = size(M)
    r2 = rank(M)
    @assert m2 == n1
    T = promote_type(eltype(L),eltype(M))
    temp = zeros(T,r1,r2)
    At_mul_B!(temp,L.V,M.U)
    V = zeros(T,n2,r1)
    A_mul_Bt!(V,M.V,temp)
    LowRankMatrix(L.U,V)
end

function *(L::LowRankMatrix,A::AbstractMatrix)
    m,n = size(L)
    r = rank(L)
    @assert size(A,1) == n
    V = zeros(promote_type(eltype(L),eltype(A)),size(A,2),r)
    At_mul_B!(V,A,L.V)
    LowRankMatrix(L.U,V)
end

function *(A::AbstractMatrix,L::LowRankMatrix)
    m,n = size(L)
    r = rank(L)
    @assert size(A,2) == m
    U = zeros(promote_type(eltype(A),eltype(L)),size(A,1),r)
    At_mul_B!(U,A,L.U)
    LowRankMatrix(U,L.V)
end

\(L::LowRankMatrix,b::AbstractVecOrMat) = full(L)\b
