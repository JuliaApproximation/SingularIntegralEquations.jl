

##
# Represent a power-of-two hierarchical matrix
# [ x x \ /
#   x x / \
#   \ / x x
#   / \ x x ]
# where the diagonal blocks are of the same type.
##

export HierarchicalMatrix, partitionmatrix

type HierarchicalMatrix{T} <: AbstractSparseMatrix{T,Int}
    diagonaldata::Union(@compat(Tuple{HierarchicalMatrix{T},HierarchicalMatrix{T}}),@compat(Tuple{T,T})) # n ≥ 2 ? Tuple of two on-diagonal HierarchicalMatrix{T} : Tuple of two on-diagonal T
    offdiagonaldata::@compat(Tuple{T,T}) # Tuple of two off-diagonal T
    n::Int # Power of hierarchy (i.e. 2^n)
    function HierarchicalMatrix(data::Vector{T},n::Int)
        @assert length(data) == 3*2^n-2
        if n == 0
            return data[1]
        elseif n == 1
            return new((data[1],data[2]),(data[3],data[4]),n)
        elseif n ≥ 2
            p1,p2 = setpartition(n)
            return new((HierarchicalMatrix(data[p1],n-1),HierarchicalMatrix(data[p2],n-1)),(data[end-1],data[end]),n)
        end
    end
end

function setpartition(n::Int)
    p1,p2 = Array(Int,3*2^(n-1)-2),Array(Int,3*2^(n-1)-2)
    k,k1,k2 = 1,1,1
    for j=1:2^(n-1)
        p1[k1] = k
        k1+=1
        k+=1
    end
    for j=1:2^(n-1)
        p2[k2] = k
        k2+=1
        k+=1
    end
    for i = n-1:-1:1
        for j=1:2^i
            p1[k1] = k
            k1+=1
            k+=1
        end
        for j=1:2^i
            p2[k2] = k
            k2+=1
            k+=1
        end
    end
    p1,p2
end

HierarchicalMatrix{T}(data::Vector{T},n::Integer)=HierarchicalMatrix{T}(data,n)
HierarchicalMatrix{T}(data::Vector{T})=HierarchicalMatrix(data,round(Int,log2(div(length(data)+2,3))))


Base.eltype{T}(::HierarchicalMatrix{T})=T
#Base.convert{V}(::Type{HierarchicalMatrix{V}},M::HierarchicalMatrix) = HierarchicalMatrix{V}(convert(Vector{V},M.data),M.n)

Base.promote_rule{T,V}(::Type{HierarchicalMatrix{T}},::Type{HierarchicalMatrix{V}})=HierarchicalMatrix{promote_type(T,V)}

partitionmatrix{T}(H::HierarchicalMatrix{T}) = (H.diagonaldata),(H.offdiagonaldata)



Base.size{T}(H::HierarchicalMatrix{AbstractArray{T,2}},k) = k==1? size(H.diagonaldata[1],k)+size(H.offdiagonaldata[1],k) : k == 2? size(H.diagonaldata[1],k)+size(H.offdiagonaldata[2],k) : Nothing
Base.size{T}(H::HierarchicalMatrix{AbstractArray{T,2}}) = size(H,1),size(H,2)

function Base.getindex{T}(H::HierarchicalMatrix{AbstractArray{T,2}},i::Int,j::Int)
    m1,n1 = size(H.diagonaldata[1])
    m2,n2 = size(H.diagonaldata[2])
    if 1 ≤ i ≤ m1
        if 1 ≤ j ≤ n1
            return getindex(H.diagonaldata[1],i,j)
        elseif n1 < j ≤ n1+n2
            return getindex(H.offdiagonaldata[2],i,j-n1)
        end
    elseif m1 < i ≤ m1+m2
        if 1 ≤ j ≤ n1
            return getindex(H.offdiagonaldata[1],i-m1,j)
        elseif n1 < j ≤ n1+n2
            return getindex(H.diagonaldata[2],i-m1,j-n1)
        end
    end
end
Base.getindex{T}(H::HierarchicalMatrix{AbstractArray{T,2}},i::Int,jr::Range) = transpose(T[H[i,j] for j=jr])
Base.getindex{T}(H::HierarchicalMatrix{AbstractArray{T,2}},ir::Range,j::Int) = T[H[i,j] for i=ir]
Base.getindex{T}(H::HierarchicalMatrix{AbstractArray{T,2}},ir::Range,jr::Range) = T[H[i,j] for i=ir,j=jr]
Base.full{T}(H::HierarchicalMatrix{AbstractArray{T,2}})=H[1:size(H,1),1:size(H,2)]

function *{S,T}(H::HierarchicalMatrix{AbstractArray{S,2}},b::Vector{T})
    m1,m2 = size(H.diagonaldata[1],1),size(H.offdiagonaldata[1],1)
    (b1,b2) = (b[1:m1],b[1+m1:m1+m2])
    vcat(H.diagonaldata[1]*b1+H.offdiagonaldata[2]*b2,H.offdiagonaldata[1]*b1+H.diagonaldata[2]*b2)
end
\{S,T}(H::HierarchicalMatrix{AbstractArray{S,2}},b::VecOrMat{T}) = full(H)\b
