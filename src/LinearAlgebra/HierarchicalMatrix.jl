

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
# The diagonal blocks are ordered along the diagonal,
# and the off-diagonal blocks are ordered from: bottom left,
# top right, then followed recursively by top left and bottom right.
##

export HierarchicalMatrix, partitionmatrix, isfactored, ishierarchical

typealias AbstractHierarchicalMatrix{S,V,T} AbstractHierarchicalArray{S,V,T,2}

type HierarchicalMatrix{S,V,T} <: AbstractHierarchicalMatrix{S,V,T}
    diagonaldata::Tuple{S,S}   # Tuple of two on-diagonal S
    offdiagonaldata::Tuple{V,V} # Tuple of two off-diagonal V
    hierarchicaldata::Tuple{HierarchicalMatrix{S,V,T},HierarchicalMatrix{S,V,T}} # Tuple of two on-diagonal HierarchicalMatrix{S,V,T}
    hierarchical::Bool

    A::Matrix{T} # Cache of matrix for pivot computation
    factorization::Factorization{T} # Cache of factorization of A for pivot computation
    factored::Bool

    function HierarchicalMatrix(diagonaldata::Tuple{S,S},offdiagonaldata::Tuple{V,V})
        H = new()
        H.diagonaldata = diagonaldata
        H.offdiagonaldata = offdiagonaldata
        H.hierarchical = false

        r1,r2 = rank(offdiagonaldata[1]),rank(offdiagonaldata[2])
        H.A = eye(T,r1+r2,r1+r2)
        H.factorization = pivotldufact(H.A,r1,r2)
        H.factored = false

        H
    end

    function HierarchicalMatrix(hierarchicaldata::Tuple{HierarchicalMatrix{S,V,T},HierarchicalMatrix{S,V,T}},offdiagonaldata::Tuple{V,V})
        H = new()
        H.hierarchicaldata = hierarchicaldata
        H.offdiagonaldata = offdiagonaldata
        H.hierarchical = true

        r1,r2 = rank(offdiagonaldata[1]),rank(offdiagonaldata[2])
        H.A = eye(T,r1+r2,r1+r2)
        H.factorization = pivotldufact(H.A,r1,r2)
        H.factored = false

        H
    end
end

HierarchicalMatrix{S,V}(diagonaldata::Tuple{S,S},offdiagonaldata::Tuple{V,V}) = HierarchicalMatrix{S,V,promote_type(eltype(S),eltype(V))}(diagonaldata,offdiagonaldata)
HierarchicalMatrix{S,V,T}(hierarchicaldata::Tuple{HierarchicalMatrix{S,V,T},HierarchicalMatrix{S,V,T}},offdiagonaldata::Tuple{V,V}) = HierarchicalMatrix{S,V,T}(hierarchicaldata,offdiagonaldata)

HierarchicalMatrix(diagonaldata::Vector,offdiagonaldata::Vector)=HierarchicalMatrix(diagonaldata,offdiagonaldata,round(Int,log2(length(diagonaldata))))

function HierarchicalMatrix(diagonaldata::Vector,offdiagonaldata::Vector,n::Int)
    @assert length(diagonaldata) == 2^n
    @assert length(offdiagonaldata) == 2^(n+1)-2

    if n == 1
        return HierarchicalMatrix(tuple(diagonaldata...),tuple(offdiagonaldata...))
    elseif n ≥ 2
        dldp2 = div(length(offdiagonaldata)+2,2)
        return HierarchicalMatrix((HierarchicalMatrix(diagonaldata[1:2^(n-1)],offdiagonaldata[3:dldp2],n-1),
                                   HierarchicalMatrix(diagonaldata[1+2^(n-1):end],offdiagonaldata[dldp2+1:end],n-1)),
                                 (offdiagonaldata[1],offdiagonaldata[2]))
    end
end


isfactored(H::HierarchicalMatrix) = H.factored
ishierarchical(H::HierarchicalMatrix) = H.hierarchical
degree(H::HierarchicalMatrix) = ishierarchical(H) ? 1+degree(H.hierarchicaldata[1]) : 1
partitionmatrix(H::HierarchicalMatrix) = ishierarchical(H) ? (H.hierarchicaldata,H.offdiagonaldata) : (H.diagonaldata,H.offdiagonaldata)

function collectoffdiagonaldata(H::HierarchicalMatrix)
    data = collect(H.offdiagonaldata)
    ishierarchical(H) && append!(data,mapreduce(collectoffdiagonaldata,vcat,H.hierarchicaldata))
    data
end

function collectdiagonaldata{S}(H::HierarchicalMatrix{S})
    data = S[]
    if ishierarchical(H)
        append!(data,mapreduce(collectdiagonaldata,vcat,H.hierarchicaldata))
    else
        push!(data,H.diagonaldata...)
    end
    data
end

Base.convert{S,V,T}(::Type{HierarchicalMatrix{S,V,T}},M::HierarchicalMatrix) = HierarchicalMatrix(convert(Vector{S},collectdiagonaldata(M)),convert(Vector{V},collectoffdiagonaldata(M)))
Base.promote_rule{S,V,T,SS,VV,TT}(::Type{HierarchicalMatrix{S,V,T}},::Type{HierarchicalMatrix{SS,VV,TT}})=HierarchicalMatrix{promote_type(S,SS),promote_type(V,VV),promote_type(T,TT)}

Base.transpose(H::HierarchicalMatrix) = HierarchicalMatrix(map(transpose,H.diagonaldata),map(transpose,reverse(H.offdiagonaldata)))
Base.ctranspose(H::HierarchicalMatrix) = HierarchicalMatrix(map(ctranspose,H.diagonaldata),map(ctranspose,reverse(H.offdiagonaldata)))

function Base.size{S<:AbstractMatrix,V<:AbstractMatrix}(H::HierarchicalMatrix{S,V})
    m1,n1 = size(H.offdiagonaldata[2])
    m2,n2 = size(H.offdiagonaldata[1])
    m1+m2,n1+n2
end

function Base.getindex{S<:AbstractMatrix,V<:AbstractMatrix}(H::HierarchicalMatrix{S,V},i::Int,j::Int)
    m1,n1 = size(H.offdiagonaldata[2])
    m2,n2 = size(H.offdiagonaldata[1])
    if 1 ≤ i ≤ m1
        if 1 ≤ j ≤ n1
            return getindex(ishierarchical(H) ? H.hierarchicaldata[1] : H.diagonaldata[1],i,j)
        elseif n1 < j ≤ n1+n2
            return getindex(H.offdiagonaldata[2],i,j-n1)
        else
            throw(BoundsError())
        end
    elseif m1 < i ≤ m1+m2
        if 1 ≤ j ≤ n1
            return getindex(H.offdiagonaldata[1],i-m1,j)
        elseif n1 < j ≤ n1+n2
            return getindex(ishierarchical(H) ? H.hierarchicaldata[2] : H.diagonaldata[2],i-m1,j-n1)
        else
            throw(BoundsError())
        end
    else
        throw(BoundsError())
    end
end
Base.getindex{S<:AbstractMatrix,V<:AbstractMatrix}(H::HierarchicalMatrix{S,V},i::Int,jr::Range) = eltype(H)[H[i,j] for j=jr].'
Base.getindex{S<:AbstractMatrix,V<:AbstractMatrix}(H::HierarchicalMatrix{S,V},ir::Range,j::Int) = eltype(H)[H[i,j] for i=ir]
Base.getindex{S<:AbstractMatrix,V<:AbstractMatrix}(H::HierarchicalMatrix{S,V},ir::Range,jr::Range) = eltype(H)[H[i,j] for i=ir,j=jr]
Base.full{S<:AbstractMatrix,V<:AbstractMatrix}(H::HierarchicalMatrix{S,V})=H[1:size(H,1),1:size(H,2)]


function Base.rank(H::HierarchicalMatrix)
    n = degree(H)
    A = Array{Int}(2^n,2^n)
    r1,r2 = map(rank,H.offdiagonaldata)
    for j=1:2^(n-1),i=1:2^(n-1)
        A[i+2^(n-1),j] = r1
        A[i,j+2^(n-1)] = r2
    end
    A11,A22 = map(rank,ishierarchical(H) ? H.hierarchicaldata : H.diagonaldata)
    for j=1:2^(n-1),i=1:2^(n-1)
        A[i,j] = A11[i,j]
        A[i+2^(n-1),j+2^(n-1)] = A22[i,j]
    end
    A
end

for op in (:+,:-)
    @eval begin
        function $op(H::HierarchicalMatrix,J::HierarchicalMatrix)
            Hd,Ho = collectdiagonaldata(H),collectoffdiagonaldata(H)
            Jd,Jo = collectdiagonaldata(J),collectoffdiagonaldata(J)
            HierarchicalMatrix($op(Hd,Jd),$op(Ho,Jo))
        end
        $op(H::HierarchicalMatrix) = HierarchicalMatrix($op(collectdiagonaldata(H)),$op(collectoffdiagonaldata(H)))
    end
end

function *(H::HierarchicalMatrix,b::AbstractVecOrMat)
    m1,m2 = size(H.offdiagonaldata[2],1),size(H.offdiagonaldata[1],1)
    n = size(b,2)
    (b1,b2) = (b[1:m1,1:n],b[1+m1:m1+m2,1:n])
    if ishierarchical(H)
        vcat(H.hierarchicaldata[1]*b1+H.offdiagonaldata[2]*b2,H.offdiagonaldata[1]*b1+H.hierarchicaldata[2]*b2)
    else
        vcat(H.diagonaldata[1]*b1+H.offdiagonaldata[2]*b2,H.offdiagonaldata[1]*b1+H.diagonaldata[2]*b2)
    end
end

\(H::HierarchicalMatrix,b::AbstractVecOrMat) = full(H)\b
