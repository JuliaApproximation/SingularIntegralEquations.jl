

##
# Represent an m x n rank-r matrix
# A = U V^T
##

export LowRankMatrix

type LowRankMatrix{T} <: AbstractMatrix{T}
    U::Matrix{T} # m x r Matrix
    V::Matrix{T} # n x r Matrix

    function LowRankMatrix(U::Matrix{T},V::Matrix{T})
        mu,ru = size(U)
        nv,rv = size(V)
        @assert ru == rv
        new(U,V)
    end
end

LowRankMatrix{S,T}(U::Matrix{S},V::Matrix{T})=LowRankMatrix{promote_type(S,T)}(promote(U,V)...)

LowRankMatrix(U::Vector,V::Matrix)=LowRankMatrix(reshape(U,length(U),1),V)
LowRankMatrix(U::Matrix,V::Vector)=LowRankMatrix(U,reshape(V,length(V),1))
LowRankMatrix(U::Vector,V::Vector)=LowRankMatrix(reshape(U,length(U),1),reshape(V,length(V),1))
LowRankMatrix(a::Number,m::Int,n::Int)=fill!(LowRankMatrix(zeros(eltype(a),m),zeros(eltype(a),n)),a)

function LowRankMatrix(A::Matrix)
    U,Σ,V = svd(A)
    r = refactorsvd!(U,Σ,V)
    LowRankMatrix(U[:,1:r],V[:,1:r])
end

function balance!(L::LowRankMatrix)
    m,n = size(L)
    for k=1:rank(L)
        uk = zero(eltype(L))
        for i=1:m
            @inbounds uk += abs2(L.U[i,k])
        end
        vk = zero(eltype(L))
        for j=1:n
            @inbounds vk += abs2(L.V[j,k])
        end
        uk,vk = sqrt(uk),sqrt(vk)
        σk = sqrt(uk*vk)
        uk,vk = σk/uk,σk/vk
        for i=1:m
            @inbounds L.U[i,k] *= uk
        end
        for j=1:n
            @inbounds L.V[j,k] *= vk
        end
    end
    L
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

Base.size(L::LowRankMatrix) = size(L.U,1),size(L.V,1)
Base.rank(L::LowRankMatrix) = size(L.U,2)
Base.transpose(L::LowRankMatrix) = LowRankMatrix(L.V,L.U)
Base.ctranspose{T<:Real}(L::LowRankMatrix{T}) = LowRankMatrix(L.V,L.U)
Base.ctranspose(L::LowRankMatrix) = LowRankMatrix(conj(L.V),conj(L.U))
Base.fill!{T}(L::LowRankMatrix{T}, x::T) = (fill!(L.U, sqrt(abs(x)/rank(L)));fill!(L.V,sqrt(abs(x)/rank(L))/sign(x)); L)

function Base.getindex(L::LowRankMatrix,i::Int,j::Int)
    m,n = size(L)
    if 1 ≤ i ≤ m && 1 ≤ j ≤ n
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

# constructors

for op in (:zeros,:eye,:ones,:rand)
    lrop = parse("lr"*string(op))
    @eval begin
        export $lrop
        $lrop(T::Type,m::Int,n::Int) = LowRankMatrix($op(T,m,n))
        $lrop(T::Type,n::Int) = $lrop(T,n,n)
        $lrop(m::Int,n::Int) = LowRankMatrix($op(Float64,m,n))
        $lrop(n::Int) = $lrop(n,n)
    end
end
export lrrandn
lrrandn(::Type{Float64},m::Int,n::Int) = LowRankMatrix(randn(m,n))
lrrandn(::Type{Float64},n::Int) = lrrandn(n,n)
lrrandn(m::Int,n::Int) = lrrandn(m,n)
lrrandn(n::Int) = lrrandn(n,n)

# algebra

for op in (:+,:-,:.+,:.-)
    @eval begin
        $op(L::LowRankMatrix) = LowRankMatrix($op(L.U),L.V)

        $op(a::Bool,L::LowRankMatrix{Bool}) = error("Not callable")
        $op(L::LowRankMatrix{Bool},a::Bool) = error("Not callable")
        $op(a::Number,L::LowRankMatrix) = $op(LowRankMatrix(a,size(L)...),L)
        $op(L::LowRankMatrix,a::Number) = $op(L,LowRankMatrix(a,size(L)...))

        function $op(L::LowRankMatrix,M::LowRankMatrix)
            @assert size(L) == size(M)
            LowRankMatrix(hcat(L.U,$op(M.U)),hcat(L.V,M.V))
        end
        $op(L::LowRankMatrix,A::AbstractMatrix) = $op(full(L),A)
        $op(A::AbstractMatrix,L::LowRankMatrix) = $op(L,A)
    end
end

*(a::Number,L::LowRankMatrix) = LowRankMatrix(a*L.U,L.V)
*(L::LowRankMatrix,a::Number) = LowRankMatrix(L.U,L.V*a)
.*(a::Number,L::LowRankMatrix) = a*L
.*(L::LowRankMatrix,a::Number) = L*a

function Base.A_mul_B!(b::AbstractVector,L::LowRankMatrix,x::AbstractVector)
    m,n = size(L)
    r = rank(L)
    temp = zeros(promote_type(eltype(L),eltype(x)),r)
    At_mul_B!(temp,L.V,x)
    A_mul_B!(b,L.U,temp)
    b
end

function *(L::LowRankMatrix,M::LowRankMatrix)
    m1,n1 = size(L)
    r1 = rank(L)
    m2,n2 = size(M)
    r2 = rank(M)
    T = promote_type(eltype(L),eltype(M))
    temp = zeros(T,r1,r2)
    At_mul_B!(temp,L.V,M.U)
    V = zeros(T,n2,r1)
    A_mul_Bt!(V,M.V,temp)
    balance!(LowRankMatrix(copy(L.U),V))
end

function *(L::LowRankMatrix,A::AbstractMatrix)
    m,n = size(L)
    r = rank(L)
    V = zeros(promote_type(eltype(L),eltype(A)),size(A,2),r)
    At_mul_B!(V,A,L.V)
    balance!(LowRankMatrix(copy(L.U),V))
end

function *(A::AbstractMatrix,L::LowRankMatrix)
    m,n = size(L)
    r = rank(L)
    U = zeros(promote_type(eltype(A),eltype(L)),size(A,1),r)
    At_mul_B!(U,A,L.U)
    balance!(LowRankMatrix(U,copy(L.V)))
end

\(L::LowRankMatrix,b::AbstractVecOrMat) = full(L)\b
