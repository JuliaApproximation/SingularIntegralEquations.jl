

##
# Represent a power-of-two hierarchical matrix
# [ x x \ /
#   x x / \
#   \ / x x
#   / \ x x ]
# where the diagonal blocks are of one type, and
# the off-diagonal blocks are of another type.
#
# To have simple recursive construction, we use the ordering:
# [ D₁   L₄
#      ⋯      L₂
#   L₃   D₂
#          D₃   L₆
#      L₁     ⋯
#          L₅   D₄ ]
# where it can be seen that the diagonal blocks are ordered along the diagonal,
# and the off-diagonal blocks are ordered from bottom left, top right, then
# followed recursively by top left and bottom right.
##

export HierarchicalMatrix, partitionmatrix

# TODO: how to extract the element-types of S and T into the Abstract supertype?

type HierarchicalMatrix{S,T} <: AbstractMatrix{promote_type(S,T)}
    diagonaldata::Union(@compat(Tuple{HierarchicalMatrix{S,T},HierarchicalMatrix{S,T}}),@compat(Tuple{S,S})) # n ≥ 2 ? Tuple of two on-diagonal HierarchicalMatrix{S,T} : Tuple of two on-diagonal S
    offdiagonaldata::@compat(Tuple{T,T}) # Tuple of two off-diagonal T
    n::Int # Power of hierarchy (i.e. 2^n)

    function HierarchicalMatrix(diagonaldata::Vector{S},offdiagonaldata::Vector{T},n::Int)
        @assert length(diagonaldata) == 2^n
        @assert length(offdiagonaldata) == 2^(n+1)-2
        if n == 1
            return new(tuple(diagonaldata...),tuple(offdiagonaldata...),n)
        elseif n ≥ 2
            dldp2 = div(length(offdiagonaldata)+2,2)
            return new((HierarchicalMatrix(diagonaldata[1:2^(n-1)],offdiagonaldata[3:dldp2],n-1),HierarchicalMatrix(diagonaldata[1+2^(n-1):end],offdiagonaldata[dldp2+1:end],n-1)),(offdiagonaldata[1],offdiagonaldata[2]),n)
        end
    end
    HierarchicalMatrix(diagonaldata::Union(@compat(Tuple{HierarchicalMatrix{S,T},HierarchicalMatrix{S,T}}),@compat(Tuple{S,S})),offdiagonaldata::@compat(Tuple{T,T}),n::Int) = new(diagonaldata,offdiagonaldata,n)
end

HierarchicalMatrix{S,T}(diagonaldata::Vector{S},offdiagonaldata::Vector{T},n::Int)=HierarchicalMatrix{S,T}(diagonaldata,offdiagonaldata,n)
HierarchicalMatrix{S,T}(diagonaldata::Vector{S},offdiagonaldata::Vector{T})=HierarchicalMatrix(diagonaldata,offdiagonaldata,round(Int,log2(length(diagonaldata))))

HierarchicalMatrix{S,T}(diagonaldata::Union(@compat(Tuple{HierarchicalMatrix{S,T},HierarchicalMatrix{S,T}}),@compat(Tuple{S,S})),offdiagonaldata::@compat(Tuple{T,T}),n::Int) = HierarchicalMatrix{S,T}(diagonaldata,offdiagonaldata,n)
HierarchicalMatrix{S,T}(diagonaldata::Union(@compat(Tuple{HierarchicalMatrix{S,T},HierarchicalMatrix{S,T}}),@compat(Tuple{S,S})),offdiagonaldata::@compat(Tuple{T,T})) = HierarchicalMatrix(diagonaldata,offdiagonaldata,round(Int,log2(length(diagonaldata))))


function collectoffdiagonaldata{S,T}(H::HierarchicalMatrix{S,T})
    data = collect(H.offdiagonaldata)
    if H.n ≥ 2
        append!(data,mapreduce(collectoffdiagonaldata,vcat,H.diagonaldata))
    end
    data
end

function collectdiagonaldata{S,T}(H::HierarchicalMatrix{S,T})
    data = S[]
    if H.n == 1
        push!(data,H.diagonaldata...)
    elseif H.n ≥ 2
        append!(data,mapreduce(collectdiagonaldata,vcat,H.diagonaldata))
    end
    data
end


Base.eltype{S,T}(::HierarchicalMatrix{S,T})=promote_type(eltype(S),eltype(T))
Base.convert{U,V}(::Type{HierarchicalMatrix{U,V}},M::HierarchicalMatrix) = HierarchicalMatrix{U,V}(convert(Vector{U},collectdiagonaldata(M)),convert(Vector{V},collectoffdiagonaldata(M)),M.n)
Base.promote_rule{S,T,U,V}(::Type{HierarchicalMatrix{S,T}},::Type{HierarchicalMatrix{U,V}})=HierarchicalMatrix{promote_type(S,U),promote_type(T,V)}

Base.transpose(H::HierarchicalMatrix) = HierarchicalMatrix(map(transpose,H.diagonaldata),map(transpose,reverse(H.offdiagonaldata)),H.n)
Base.ctranspose(H::HierarchicalMatrix) = HierarchicalMatrix(map(ctranspose,H.diagonaldata),map(ctranspose,reverse(H.offdiagonaldata)),H.n)

function Base.size{S<:AbstractMatrix,T<:AbstractMatrix}(H::HierarchicalMatrix{S,T})
    m1,n1 = size(H.offdiagonaldata[2])
    m2,n2 = size(H.offdiagonaldata[1])
    m1+m2,n1+n2
end

function Base.getindex{S<:AbstractMatrix,T<:AbstractMatrix}(H::HierarchicalMatrix{S,T},i::Int,j::Int)
    m1,n1 = size(H.offdiagonaldata[2])
    m2,n2 = size(H.offdiagonaldata[1])
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
Base.getindex{S<:AbstractMatrix,T<:AbstractMatrix}(H::HierarchicalMatrix{S,T},i::Int,jr::Range) = eltype(H)[H[i,j] for j=jr].'#'
Base.getindex{S<:AbstractMatrix,T<:AbstractMatrix}(H::HierarchicalMatrix{S,T},ir::Range,j::Int) = eltype(H)[H[i,j] for i=ir]
Base.getindex{S<:AbstractMatrix,T<:AbstractMatrix}(H::HierarchicalMatrix{S,T},ir::Range,jr::Range) = eltype(H)[H[i,j] for i=ir,j=jr]
Base.full{S<:AbstractMatrix,T<:AbstractMatrix}(H::HierarchicalMatrix{S,T})=H[1:size(H,1),1:size(H,2)]

partitionmatrix{S,T}(H::HierarchicalMatrix{S,T}) = H.diagonaldata,H.offdiagonaldata

function *{S,T,V}(H::HierarchicalMatrix{S,T},b::AbstractVecOrMat{V})
    m1,m2 = size(H.offdiagonaldata[2],1),size(H.offdiagonaldata[1],1)
    n = size(b,2)
    (b1,b2) = (b[1:m1,1:n],b[1+m1:m1+m2,1:n])
    vcat(H.diagonaldata[1]*b1+H.offdiagonaldata[2]*b2,H.offdiagonaldata[1]*b1+H.diagonaldata[2]*b2)
end

\{S,T,V}(H::HierarchicalMatrix{S,T},b::AbstractVecOrMat{V}) = full(H)\b
\{S<:AbstractMatrix,T<:LowRankMatrix,V}(H::HierarchicalMatrix{S,T},f::AbstractVecOrMat{V}) = woodburysolve(H,f)
