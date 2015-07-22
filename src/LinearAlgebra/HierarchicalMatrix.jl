

##
# Represent a power-of-two hierarchical matrix
# [ x x \ /
#   x x / \
#   \ / x x
#   / \ x x ]
# where the diagonal blocks are of the same type.
# To have simple recursive construction, we use the ordering:
# [ 5   4
#     ⋯      2
#   3   6
#          9   8
#     1      ⋯
#          7   10 ]
##

export HierarchicalMatrix, partitionmatrix

type HierarchicalMatrix{T} <: AbstractSparseMatrix{T,Int}
    offdiagonaldata::@compat(Tuple{T,T}) # Tuple of two off-diagonal T
    diagonaldata::Union(@compat(Tuple{HierarchicalMatrix{T},HierarchicalMatrix{T}}),@compat(Tuple{T,T})) # n ≥ 2 ? Tuple of two on-diagonal HierarchicalMatrix{T} : Tuple of two on-diagonal T
    n::Int # Power of hierarchy (i.e. 2^n)

    function HierarchicalMatrix(data::Vector{T},n::Int)
        @assert length(data) == 3*2^n-2
        if n == 0
            return data[1] ## TODO: not type-stable
        elseif n == 1
            return new((data[1],data[2]),(data[3],data[4]),n)
        elseif n ≥ 2
            dldp2 = div(length(data)+2,2)
            return new((data[1],data[2]),(HierarchicalMatrix(data[3:dldp2],n-1),HierarchicalMatrix(data[dldp2+1:end],n-1)),n)
        end
    end
end

HierarchicalMatrix{T}(data::Vector{T},n::Integer)=HierarchicalMatrix{T}(data,n)
HierarchicalMatrix{T}(data::Vector{T})=HierarchicalMatrix(data,round(Int,log2(div(length(data)+2,3))))

function collectdata{T}(H::HierarchicalMatrix{T})
    data = collect(H.offdiagonaldata)
    if H.n == 1
        push!(data,H.diagonaldata...)
    elseif H.n ≥ 2
        append!(data,mapreduce(collectdata,vcat,H.diagonaldata))
    end
    data
end

Base.eltype{T}(::HierarchicalMatrix{T})=T
Base.convert{V}(::Type{HierarchicalMatrix{V}},M::HierarchicalMatrix) = HierarchicalMatrix{V}(convert(Vector{V},collectdata(M)),M.n)
Base.promote_rule{T,V}(::Type{HierarchicalMatrix{T}},::Type{HierarchicalMatrix{V}})=HierarchicalMatrix{promote_type(T,V)}

function Base.size{T<:AbstractMatrix}(H::HierarchicalMatrix{T})
    m1,n1 = size(H.offdiagonaldata[1])
    m2,n2 = size(H.offdiagonaldata[2])
    m1+m2,n1+n2
end

function Base.getindex{T<:AbstractMatrix}(H::HierarchicalMatrix{T},i::Int,j::Int)
    m1,n1 = size(H.offdiagonaldata[1])
    m2,n2 = size(H.offdiagonaldata[2])
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
Base.getindex{T<:AbstractMatrix}(H::HierarchicalMatrix{T},i::Int,jr::Range) = transpose(eltype(T)[H[i,j] for j=jr])
Base.getindex{T<:AbstractMatrix}(H::HierarchicalMatrix{T},ir::Range,j::Int) = eltype(T)[H[i,j] for i=ir]
Base.getindex{T<:AbstractMatrix}(H::HierarchicalMatrix{T},ir::Range,jr::Range) = eltype(T)[H[i,j] for i=ir,j=jr]
Base.full{T<:AbstractMatrix}(H::HierarchicalMatrix{T})=H[1:size(H,1),1:size(H,2)]

partitionmatrix{T}(H::HierarchicalMatrix{T}) = H.diagonaldata,H.offdiagonaldata

function *{S,T}(H::HierarchicalMatrix{S},b::AbstractVecOrMat{T})
    m1,m2 = size(H.diagonaldata[1],1),size(H.offdiagonaldata[1],1)
    (b1,b2) = (b[1:m1],b[1+m1:m1+m2])
    vcat(H.diagonaldata[1]*b1+H.offdiagonaldata[2]*b2,H.offdiagonaldata[1]*b1+H.diagonaldata[2]*b2)
end
\{S,T}(H::HierarchicalMatrix{S},b::AbstractVecOrMat{T}) = full(H)\b
