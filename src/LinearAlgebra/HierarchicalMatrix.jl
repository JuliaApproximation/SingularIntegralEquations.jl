

##
# Represent a power-of-two hierarchical matrix
# [ x x \ /
#   x x / \
#   \ / x x
#   / \ x x ]
# ordering the data from the main diagonal outward
# starting with the lower block
##

export HierarchicalMatrix

type HierarchicalMatrix{T} <: AbstractSparseMatrix{T,Int}
    diagonaldata::Union(@compat(Tuple{HierarchicalMatrix{T},HierarchicalMatrix{T}}),@compat(Tuple{T,T})) # n ≥ 2 ? Tuple of two on-diagonal HierarchicalMatrix{T} : Tuple of two on-diagonal T
    offdiagonaldata::@compat(Tuple{T,T}) # Tuple of two off-diagonal T
    n::Int # Power of hierarchy (i.e. 2^n)
    function HierarchicalMatrix(data::Vector{T},n::Int)
        @assert length(data) == 3*2^n-2
        if n == 1
            return new((data[1],data[2]),(data[3],data[4]),n)
        elseif n ≥ 2
            data1,data2 = dividedata(data,n)
            return new((HierarchicalMatrix(data1,n-1),HierarchicalMatrix(data2,n-1)),(data[end-1],data[end]),n)
        end
    end
end

function dividedata{T}(data::Vector{T},n::Int)
    data1,data2 = Array(T,3*2^(n-1)-2),Array(T,3*2^(n-1)-2)
    k,k1,k2 = 1,1,1
    for j=1:2^(n-1)
        data1[k1] = data[k]
        k1+=1
        k+=1
    end
    for j=1:2^(n-1)
        data2[k2] = data[k]
        k2+=1
        k+=1
    end
    for i = n-1:-1:1
        for j=1:2^i
            data1[k1] = data[k]
            k1+=1
            k+=1
        end
        for j=1:2^i
            data2[k2] = data[k]
            k2+=1
            k+=1
        end
    end
    data1,data2
end

HierarchicalMatrix{T}(data::Vector{T},n::Integer)=HierarchicalMatrix{T}(data,n)
HierarchicalMatrix{T}(data::Vector{T})=HierarchicalMatrix(data,round(Int,log2(div(length(data)+2,3))))


Base.eltype{T}(::HierarchicalMatrix{T})=T
Base.convert{V}(::Type{HierarchicalMatrix{V}},M::HierarchicalMatrix) = HierarchicalMatrix{V}(convert(Vector{V},M.data),M.n)

Base.promote_rule{T,V}(::Type{HierarchicalMatrix{T}},::Type{HierarchicalMatrix{V}})=HierarchicalMatrix{promote_type(T,V)}
