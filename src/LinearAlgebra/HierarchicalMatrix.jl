

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
# SS is Union(@compat(Tuple{HierarchicalMatrix{S,T},HierarchicalMatrix{S,T}}),@compat(Tuple{S,S}))
    diagonaldata::@compat(Tuple{S,S})   # n ≥ 2 ? Tuple of two on-diagonal HierarchicalMatrix{S,T} : Tuple of two on-diagonal S
    offdiagonaldata::@compat(Tuple{T,T}) # Tuple of two off-diagonal T
    A::Matrix{P} # Cache of matrix for pivot computation
    factorization::Factorization{P} # Cache of factorization of A for pivot computation
    factored::Bool
    n::Int # Power of hierarchy (i.e. 2^n)

end


function HierarchicalMatrix(diagonaldata::Vector,offdiagonaldata::Vector,A::Matrix,factorization::Factorization,factored::Bool,n::Int)
    @assert length(diagonaldata) == 2^n
    @assert length(offdiagonaldata) == 2^(n+1)-2
    
    if n == 1
        return HierarchicalMatrix(tuple(diagonaldata...),tuple(offdiagonaldata...),A,factorization,factored,n)
    elseif n ≥ 2
        dldp2 = div(length(offdiagonaldata)+2,2)
        return HierarchicalMatrix((HierarchicalMatrix(diagonaldata[1:2^(n-1)],offdiagonaldata[3:dldp2],n-1),
                                   HierarchicalMatrix(diagonaldata[1+2^(n-1):end],offdiagonaldata[dldp2+1:end],n-1)),
                                 (offdiagonaldata[1],offdiagonaldata[2]),A,factorization,factored,n)
    end
end

#     HierarchicalMatrix(diagonaldata::Union(@compat(Tuple{HierarchicalMatrix{S,T},HierarchicalMatrix{S,T}}),@compat(Tuple{S,S})),offdiagonaldata::@compat(Tuple{T,T}),A::Matrix{P},factorization::Factorization{P},factored::Bool,n::Int) = new(diagonaldata,offdiagonaldata,A,factorization,factored,n)



#HierarchicalMatrix{S,T,P}(diagonaldata::Vector{S},offdiagonaldata::Vector{T},A::AbstractMatrix{P},n::Int)=HierarchicalMatrix{S,T,P}(diagonaldata,offdiagonaldata,A,n)
#HierarchicalMatrix{S,T}(diagonaldata::Vector{S},offdiagonaldata::Vector{T},n::Int)=HierarchicalMatrix{S,T,promote_type(eltype(S),eltype(T))}(diagonaldata,offdiagonaldata,zeros(promote_type(eltype(S),eltype(T)),mapreduce(rank,+,offdiagonaldata[1:2]),mapreduce(rank,+,offdiagonaldata[1:2])),n)

function HierarchicalMatrix(diagonaldata::Vector,offdiagonaldata::Vector,n::Int)
    P = promote_type(eltype(first(diagonaldata)),eltype(first(offdiagonaldata)))
    r1,r2 = rank(offdiagonaldata[1]),rank(offdiagonaldata[2])
    A = eye(P,r1+r2,r1+r2)
    factorization = pivotldufact(A,r1,r2)#lufact(A)
    HierarchicalMatrix(diagonaldata,offdiagonaldata,A,factorization,false,n)
end
HierarchicalMatrix(diagonaldata::Vector,offdiagonaldata::Vector)=HierarchicalMatrix(diagonaldata,offdiagonaldata,round(Int,log2(length(diagonaldata))))

HierarchicalMatrix(diagonaldata,offdiagonaldata,A::Matrix,factorization::Factorization,n::Int) = HierarchicalMatrix(diagonaldata,offdiagonaldata,A,factorization,false,n)


function HierarchicalMatrix(diagonaldata,offdiagonaldata,n::Int)
    P = promote_type(eltype(diagonaldata[1]),eltype(diagonaldata[2]),eltype(offdiagonaldata[1]),eltype(offdiagonaldata[2]))
    r1,r2 = rank(offdiagonaldata[1]),rank(offdiagonaldata[2])
    A = eye(P,r1+r2,r1+r2)
    factorization = pivotldufact(A,r1,r2)#lufact(A)
    HierarchicalMatrix(diagonaldata,offdiagonaldata,A,factorization,false,n)
end

function HierarchicalMatrix{S,T,U,V}(diagonaldata::@compat(Tuple{HierarchicalMatrix{S,T,U},HierarchicalMatrix{S,T,U}}),offdiagonaldata::@compat(Tuple{V,V}),n::Int)
    P = promote_type(U,eltype(offdiagonaldata[1]),eltype(offdiagonaldata[2]))
    r1,r2 = rank(offdiagonaldata[1]),rank(offdiagonaldata[2])
    A = eye(P,r1+r2,r1+r2)
    factorization = pivotldufact(A,r1,r2)#lufact(A)
    HierarchicalMatrix{typeof(first(diagonaldata)),promote_type(T,V),P}(diagonaldata,offdiagonaldata,A,factorization,false,n)
end

function collectoffdiagonaldata(H::HierarchicalMatrix)
    data = collect(H.offdiagonaldata)
    if H.n ≥ 2
        append!(data,mapreduce(collectoffdiagonaldata,vcat,H.diagonaldata))
    end
    data
end


#determine the type of the diagonal
diagonalmatrixtype(H::AbstractMatrix)=typeof(H)
diagonalmatrixtype(H::HierarchicalMatrix)=diagonalmatrixtype(first(H.diagonaldata))


function collectdiagonaldata(H::HierarchicalMatrix)
    S=diagonalmatrixtype(H)
    data = S[]
    if H.n == 1
        push!(data,H.diagonaldata...)
    elseif H.n ≥ 2
        append!(data,mapreduce(collectdiagonaldata,vcat,H.diagonaldata))
    end
    data
end


#TODO: shouldn't this come from P?
Base.eltype{S,T}(::HierarchicalMatrix{S,T})=promote_type(eltype(S),eltype(T))
Base.convert{U,V,P}(::Type{HierarchicalMatrix{U,V,P}},M::HierarchicalMatrix) = HierarchicalMatrix(convert(Vector{U},collectdiagonaldata(M)),convert(Vector{V},collectoffdiagonaldata(M)),M.n)
Base.promote_rule{S,T,U,V,P,PP}(::Type{HierarchicalMatrix{S,T,P}},::Type{HierarchicalMatrix{U,V,PP}})=HierarchicalMatrix{promote_type(S,U),promote_type(T,V),promote_type(P,PP)}

Base.transpose(H::HierarchicalMatrix) = HierarchicalMatrix(map(transpose,H.diagonaldata),map(transpose,reverse(H.offdiagonaldata)),H.n)
Base.ctranspose(H::HierarchicalMatrix) = HierarchicalMatrix(map(ctranspose,H.diagonaldata),map(ctranspose,reverse(H.offdiagonaldata)),H.n)

function Base.size(H::HierarchicalMatrix)
    m1,n1 = size(H.offdiagonaldata[2])
    m2,n2 = size(H.offdiagonaldata[1])
    m1+m2,n1+n2
end

function Base.getindex(H::HierarchicalMatrix,i::Int,j::Int)
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
Base.getindex(H::HierarchicalMatrix,i::Int,jr::Range) = eltype(H)[H[i,j] for j=jr].'#'
Base.getindex(H::HierarchicalMatrix,ir::Range,j::Int) = eltype(H)[H[i,j] for i=ir]
Base.getindex(H::HierarchicalMatrix,ir::Range,jr::Range) = eltype(H)[H[i,j] for i=ir,j=jr]
Base.full(H::HierarchicalMatrix)=H[1:size(H,1),1:size(H,2)]

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

function *(H::HierarchicalMatrix,b::AbstractVecOrMat)
    m1,m2 = size(H.offdiagonaldata[2],1),size(H.offdiagonaldata[1],1)
    n = size(b,2)
    (b1,b2) = (b[1:m1,1:n],b[1+m1:m1+m2,1:n])
    vcat(H.diagonaldata[1]*b1+H.offdiagonaldata[2]*b2,H.offdiagonaldata[1]*b1+H.diagonaldata[2]*b2)
end

\(H::HierarchicalMatrix,b::AbstractVecOrMat) = full(H)\b
