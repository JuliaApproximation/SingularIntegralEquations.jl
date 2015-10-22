export HierarchicalMatrix, partitionmatrix, isfactored

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

typealias AbstractHierarchicalMatrix{S,V,T,B} AbstractHierarchicalArray{Tuple{S,V},T,B,2}

type HierarchicalMatrix{S,V,T,B} <: AbstractHierarchicalMatrix{S,V,T,B}
    diagonaldata::NTuple{2,S}    # Tuple of two on-diagonal S
    offdiagonaldata::NTuple{2,V} # Tuple of two off-diagonal V
    hierarchicaldata::NTuple{2,HierarchicalMatrix{S,V,T}} # Tuple of two on-diagonal HierarchicalMatrix{S,V,T}

    A::Matrix{T} # Cache of matrix for pivot computation
    factorization::Factorization{T} # Cache of factorization of A for pivot computation
    factored::Bool

    function HierarchicalMatrix(diagonaldata::NTuple{2,S},offdiagonaldata::NTuple{2,V})
        H = new()
        H.diagonaldata = diagonaldata
        H.offdiagonaldata = offdiagonaldata

        r1,r2 = rank(offdiagonaldata[1]),rank(offdiagonaldata[2])
        H.A = eye(T,r1+r2,r1+r2)
        H.factorization = pivotldufact(H.A,r1,r2)
        H.factored = false

        H
    end

    function HierarchicalMatrix(hierarchicaldata::NTuple{2,HierarchicalMatrix{S,V,T}},offdiagonaldata::NTuple{2,V})
        H = new()
        H.hierarchicaldata = hierarchicaldata
        H.offdiagonaldata = offdiagonaldata

        r1,r2 = rank(offdiagonaldata[1]),rank(offdiagonaldata[2])
        H.A = eye(T,r1+r2,r1+r2)
        H.factorization = pivotldufact(H.A,r1,r2)
        H.factored = false

        H
    end
end

HierarchicalMatrix{S,V}(diagonaldata::NTuple{2,S},offdiagonaldata::NTuple{2,V}) = HierarchicalMatrix{S,V,promote_type(eltype(S),eltype(V)),false}(diagonaldata,offdiagonaldata)
HierarchicalMatrix{S,V,T}(hierarchicaldata::NTuple{2,HierarchicalMatrix{S,V,T}},offdiagonaldata::NTuple{2,V}) = HierarchicalMatrix{S,V,T,true}(hierarchicaldata,offdiagonaldata)

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


diagonaldata{S,V,T}(H::HierarchicalMatrix{S,V,T,false}) = H.diagonaldata
diagonaldata{S,V,T}(H::HierarchicalMatrix{S,V,T,true}) = H.hierarchicaldata
offdiagonaldata(H::HierarchicalMatrix) = H.offdiagonaldata

degree{S,V,T}(H::HierarchicalMatrix{S,V,T,true}) = 1+degree(first(diagonaldata(H)))

partitionmatrix(H::HierarchicalMatrix) = diagonaldata(H),offdiagonaldata(H)

collectoffdiagonaldata{S,V,T}(H::HierarchicalMatrix{S,V,T,false}) = collect(offdiagonaldata(H))
function collectoffdiagonaldata{S,V,T}(H::HierarchicalMatrix{S,V,T,true})
    data = collect(offdiagonaldata(H))
    append!(data,mapreduce(collectoffdiagonaldata,vcat,diagonaldata(H)))
    data
end

collectdiagonaldata{S,V,T}(H::HierarchicalMatrix{S,V,T,false}) = collect(diagonaldata(H))
function collectdiagonaldata{S,V,T}(H::HierarchicalMatrix{S,V,T,true})
    data = S[]
    append!(data,mapreduce(collectdiagonaldata,vcat,diagonaldata(H)))
    data
end

isfactored(H::HierarchicalMatrix) = H.factored


Base.convert{S,V,T,B}(::Type{HierarchicalMatrix{S,V,T,B}},M::HierarchicalMatrix) = HierarchicalMatrix(convert(Vector{S},collectdiagonaldata(M)),convert(Vector{V},collectoffdiagonaldata(M)))
Base.promote_rule{S,V,T,SS,VV,TT}(::Type{HierarchicalMatrix{S,V,T,true}},::Type{HierarchicalMatrix{SS,VV,TT,true}})=HierarchicalMatrix{promote_type(S,SS),promote_type(V,VV),promote_type(T,TT),true}
Base.promote_rule{S,V,T,SS,VV,TT}(::Type{HierarchicalMatrix{S,V,T,false}},::Type{HierarchicalMatrix{SS,VV,TT,false}})=HierarchicalMatrix{promote_type(S,SS),promote_type(V,VV),promote_type(T,TT),false}

Base.transpose(H::HierarchicalMatrix) = HierarchicalMatrix(map(transpose,diagonaldata(H)),map(transpose,reverse(offdiagonaldata(H))))
Base.ctranspose(H::HierarchicalMatrix) = HierarchicalMatrix(map(ctranspose,diagonaldata(H)),map(ctranspose,reverse(offdiagonaldata(H))))

function Base.size{S<:AbstractMatrix,V<:AbstractMatrix}(H::HierarchicalMatrix{S,V})
    H21,H12 = offdiagonaldata(H)
    m1,n1 = size(H12)
    m2,n2 = size(H21)
    m1+m2,n1+n2
end

function Base.getindex{S<:AbstractMatrix,V<:AbstractMatrix}(H::HierarchicalMatrix{S,V},i::Int,j::Int)
    H11,H22 = diagonaldata(H)
    H21,H12 = offdiagonaldata(H)
    m1,n1 = size(H12)
    m2,n2 = size(H21)
    if 1 ≤ i ≤ m1
        if 1 ≤ j ≤ n1
            return getindex(H11,i,j)
        elseif n1 < j ≤ n1+n2
            return getindex(H12,i,j-n1)
        else
            throw(BoundsError())
        end
    elseif m1 < i ≤ m1+m2
        if 1 ≤ j ≤ n1
            return getindex(H21,i-m1,j)
        elseif n1 < j ≤ n1+n2
            return getindex(H22,i-m1,j-n1)
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
    r1,r2 = map(rank,offdiagonaldata(H))
    for j=1:2^(n-1),i=1:2^(n-1)
        A[i+2^(n-1),j] = r1
        A[i,j+2^(n-1)] = r2
    end
    A11,A22 = map(rank,diagonaldata(H))
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
    H11,H22 = diagonaldata(H)
    H21,H12 = offdiagonaldata(H)
    m1,n1 = size(H12)
    m2,n2 = size(H21)
    n = size(b,2)
    (b1,b2) = (b[1:m1,1:n],b[1+m1:m1+m2,1:n])
    vcat(H11*b1+H12*b2,H21*b1+H22*b2)
end

\(H::HierarchicalMatrix,b::AbstractVecOrMat) = full(H)\b
