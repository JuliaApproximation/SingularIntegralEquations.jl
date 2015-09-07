

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

export HierarchicalMatrix, partitionmatrix, isfactored

type HierarchicalMatrix{S,T,P} <: AbstractMatrix{P}
    diagonaldata::Union(@compat(Tuple{HierarchicalMatrix{S,T},HierarchicalMatrix{S,T}}),@compat(Tuple{S,S})) # n ≥ 2 ? Tuple of two on-diagonal HierarchicalMatrix{S,T} : Tuple of two on-diagonal S
    offdiagonaldata::@compat(Tuple{T,T}) # Tuple of two off-diagonal T
    A::Matrix{P} # Cache of matrix for pivot computation
    factorization::Factorization{P} # Cache of factorization of A for pivot computation
    factored::Bool
    n::Int # Power of hierarchy (i.e. 2^n)

    function HierarchicalMatrix(diagonaldata::Vector{S},offdiagonaldata::Vector{T},A::Matrix{P},factorization::Factorization{P},factored::Bool,n::Int)
        @assert length(diagonaldata) == 2^n
        @assert length(offdiagonaldata) == 2^(n+1)-2
        if n == 1
            return new(tuple(diagonaldata...),tuple(offdiagonaldata...),A,factorization,factored,n)
        elseif n ≥ 2
            dldp2 = div(length(offdiagonaldata)+2,2)
            return new((HierarchicalMatrix(diagonaldata[1:2^(n-1)],offdiagonaldata[3:dldp2],n-1),HierarchicalMatrix(diagonaldata[1+2^(n-1):end],offdiagonaldata[dldp2+1:end],n-1)),(offdiagonaldata[1],offdiagonaldata[2]),A,factorization,factored,n)
        end
    end
    HierarchicalMatrix(diagonaldata::Union(@compat(Tuple{HierarchicalMatrix{S,T},HierarchicalMatrix{S,T}}),@compat(Tuple{S,S})),offdiagonaldata::@compat(Tuple{T,T}),A::Matrix{P},factorization::Factorization{P},factored::Bool,n::Int) = new(diagonaldata,offdiagonaldata,A,factorization,factored,n)
end

#HierarchicalMatrix{S,T,P}(diagonaldata::Vector{S},offdiagonaldata::Vector{T},A::AbstractMatrix{P},n::Int)=HierarchicalMatrix{S,T,P}(diagonaldata,offdiagonaldata,A,n)
#HierarchicalMatrix{S,T}(diagonaldata::Vector{S},offdiagonaldata::Vector{T},n::Int)=HierarchicalMatrix{S,T,promote_type(eltype(S),eltype(T))}(diagonaldata,offdiagonaldata,zeros(promote_type(eltype(S),eltype(T)),mapreduce(rank,+,offdiagonaldata[1:2]),mapreduce(rank,+,offdiagonaldata[1:2])),n)

function HierarchicalMatrix{S,T}(diagonaldata::Vector{S},offdiagonaldata::Vector{T},n::Int)
    P = promote_type(eltype(S),eltype(T))
    r1,r2 = rank(offdiagonaldata[1]),rank(offdiagonaldata[2])
    A = eye(P,r1+r2,r1+r2)
    factorization = lufact(A)
    HierarchicalMatrix{S,T,P}(diagonaldata,offdiagonaldata,A,factorization,false,n)
end
HierarchicalMatrix{S,T}(diagonaldata::Vector{S},offdiagonaldata::Vector{T})=HierarchicalMatrix(diagonaldata,offdiagonaldata,round(Int,log2(length(diagonaldata))))


HierarchicalMatrix{S,T,P}(diagonaldata::Union(@compat(Tuple{HierarchicalMatrix{S,T},HierarchicalMatrix{S,T}}),@compat(Tuple{S,S})),offdiagonaldata::@compat(Tuple{T,T}),A::Matrix{P},factorization::Factorization{P},n::Int) = HierarchicalMatrix{S,T,P}(diagonaldata,offdiagonaldata,A,factorization,false,n)
function HierarchicalMatrix{S,T}(diagonaldata::@compat(Tuple{S,S}),offdiagonaldata::@compat(Tuple{T,T}),n::Int)
    P = promote_type(eltype(diagonaldata[1]),eltype(diagonaldata[2]),eltype(offdiagonaldata[1]),eltype(offdiagonaldata[2]))
    r1,r2 = rank(offdiagonaldata[1]),rank(offdiagonaldata[2])
    A = eye(P,r1+r2,r1+r2)
    factorization = lufact(A)
    HierarchicalMatrix{S,T,P}(diagonaldata,offdiagonaldata,A,factorization,false,n)
end
function HierarchicalMatrix{S,T,U,V}(diagonaldata::@compat(Tuple{HierarchicalMatrix{S,T,U},HierarchicalMatrix{S,T,U}}),offdiagonaldata::@compat(Tuple{V,V}),n::Int)
    P = promote_type(U,eltype(offdiagonaldata[1]),eltype(offdiagonaldata[2]))
    r1,r2 = rank(offdiagonaldata[1]),rank(offdiagonaldata[2])
    A = eye(P,r1+r2,r1+r2)
    factorization = lufact(A)
    HierarchicalMatrix{S,promote_type(T,V),P}(diagonaldata,offdiagonaldata,A,factorization,false,n)
end

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
        else
            throw(BoundsError())
        end
    elseif m1 < i ≤ m1+m2
        if 1 ≤ j ≤ n1
            return getindex(H.offdiagonaldata[1],i-m1,j)
        elseif n1 < j ≤ n1+n2
            return getindex(H.diagonaldata[2],i-m1,j-n1)
        else
            throw(BoundsError())
        end
    else
        throw(BoundsError())
    end
end
Base.getindex{S<:AbstractMatrix,T<:AbstractMatrix}(H::HierarchicalMatrix{S,T},i::Int,jr::Range) = eltype(H)[H[i,j] for j=jr].'#'
Base.getindex{S<:AbstractMatrix,T<:AbstractMatrix}(H::HierarchicalMatrix{S,T},ir::Range,j::Int) = eltype(H)[H[i,j] for i=ir]
Base.getindex{S<:AbstractMatrix,T<:AbstractMatrix}(H::HierarchicalMatrix{S,T},ir::Range,jr::Range) = eltype(H)[H[i,j] for i=ir,j=jr]
Base.full{S<:AbstractMatrix,T<:AbstractMatrix}(H::HierarchicalMatrix{S,T})=H[1:size(H,1),1:size(H,2)]

partitionmatrix(H::HierarchicalMatrix) = H.diagonaldata,H.offdiagonaldata

isfactored(H::HierarchicalMatrix) = H.factored

for op in (:+,:-)
    @eval begin
        function $op(H::HierarchicalMatrix,J::HierarchicalMatrix)
            @assert (n = H.n) == J.n
            Hd,Ho = collectdiagonaldata(H),collectoffdiagonaldata(H)
            Jd,Jo = collectdiagonaldata(J),collectoffdiagonaldata(J)
            HierarchicalMatrix($op(Hd,Jd),$op(Ho,Jo),n)
        end
        $op(H::HierarchicalMatrix) = HierarchicalMatrix($op(collectdiagonaldata(H)),$op(collectoffdiagonaldata(H)),H.n)
    end
end

function *{S,T,V}(H::HierarchicalMatrix{S,T},b::AbstractVecOrMat{V})
    m1,m2 = size(H.offdiagonaldata[2],1),size(H.offdiagonaldata[1],1)
    n = size(b,2)
    (b1,b2) = (b[1:m1,1:n],b[1+m1:m1+m2,1:n])
    vcat(H.diagonaldata[1]*b1+H.offdiagonaldata[2]*b2,H.offdiagonaldata[1]*b1+H.diagonaldata[2]*b2)
end

\{S,T,V}(H::HierarchicalMatrix{S,T},b::AbstractVecOrMat{V}) = full(H)\b
