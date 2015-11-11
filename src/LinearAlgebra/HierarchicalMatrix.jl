export HierarchicalMatrix, isfactored, condest, blockrank

condest(A)=1
blocksize(A) = (1,1)
blockrank(A)=rank(A)

##
# Represent a binary hierarchical matrix
# [ x x \ /
#   x x / \
#   \ / x x
#   / \ x x ]
# where the diagonal blocks are of one type, and
# the off-diagonal blocks are of a type with low-rank structure.
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

typealias AbstractHierarchicalMatrix{S,V,T,HS,HV} AbstractHierarchicalArray{Tuple{S,V},T,Tuple{HS,HV},2}

type HierarchicalMatrix{S,V,T,HS,HV} <: AbstractHierarchicalMatrix{S,V,T,HS,HV}
    diagonaldata::HS
    offdiagonaldata::HV

    A::Matrix{T} # Cache of matrix for pivot computation
    factorization::PivotLDU{T,Matrix{T}} # Cache of factorization of A for pivot computation
    factored::Bool

    function HierarchicalMatrix(diagonaldata::HS,offdiagonaldata::HV)
        H = new()
        H.diagonaldata = diagonaldata
        H.offdiagonaldata = offdiagonaldata

        r = mapreduce(rank,+,offdiagonaldata)
        H.A = eye(T,r,r)
        H.factored = false

        H
    end
end


HierarchicalMatrix{S1,S2,V1,V2}(diagonaldata::Tuple{S1,S2},offdiagonaldata::Tuple{V1,V2}) = HierarchicalMatrix{promote_type(S1,S2),promote_type(V1,V2),promote_type(eltype(S1),eltype(S2),eltype(V1),eltype(V2)),Tuple{S1,S2},Tuple{V1,V2}}(diagonaldata,offdiagonaldata)
HierarchicalMatrix{S1,S2,V1,V2,V3,V4,T1,T2,HS1,HS2,HV1,HV2}(diagonaldata::Tuple{HierarchicalMatrix{S1,V1,T1,HS1,HV1},HierarchicalMatrix{S2,V2,T2,HS2,HV2}},offdiagonaldata::Tuple{V3,V4}) = HierarchicalMatrix{promote_type(S1,S2),promote_type(V1,V2,V3,V4),promote_type(eltype(S1),eltype(S2),eltype(V1),eltype(V2),eltype(V3),eltype(V4),T1,T2),Tuple{HierarchicalMatrix{S1,V1,T1,HS1,HV1},HierarchicalMatrix{S2,V2,T2,HS2,HV2}},Tuple{V3,V4}}(diagonaldata,offdiagonaldata)
HierarchicalMatrix{S,S1,V,V1,V2,T,HS,HV}(diagonaldata::Tuple{S1,HierarchicalMatrix{S,V,T,HS,HV}},offdiagonaldata::Tuple{V1,V2}) = HierarchicalMatrix{promote_type(S,S1),promote_type(V,V1,V2),promote_type(eltype(S),eltype(S1),eltype(V),eltype(V1),eltype(V2),T),Tuple{S1,HierarchicalMatrix{S,V,T,HS,HV}},Tuple{V1,V2}}(diagonaldata,offdiagonaldata)
HierarchicalMatrix{S,S1,V,V1,V2,T,HS,HV}(diagonaldata::Tuple{HierarchicalMatrix{S,V,T,HS,HV},S1},offdiagonaldata::Tuple{V1,V2}) = HierarchicalMatrix{promote_type(S,S1),promote_type(V,V1,V2),promote_type(eltype(S),eltype(S1),eltype(V),eltype(V1),eltype(V2),T),Tuple{HierarchicalMatrix{S,V,T,HS,HV},S1},Tuple{V1,V2}}(diagonaldata,offdiagonaldata)


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

Base.similar(H::HierarchicalMatrix) = HierarchicalMatrix(map(similar,diagonaldata(H)),map(similar,offdiagonaldata(H)))
Base.similar{SS,V,T}(H::HierarchicalMatrix{SS,V,T}, S) = HierarchicalMatrix(map(A->similar(A,S),diagonaldata(H)),map(A->similar(A,S),offdiagonaldata(H)))

diagonaldata(H::HierarchicalMatrix) = H.diagonaldata
offdiagonaldata(H::HierarchicalMatrix) = H.offdiagonaldata

nlevels(H::HierarchicalMatrix) = 1+mapreduce(nlevels,max,diagonaldata(H))

partition(H::HierarchicalMatrix) = diagonaldata(H),offdiagonaldata(H)

collectoffdiagonaldata{S,V,T}(H::HierarchicalMatrix{S,V,T,NTuple{2,S}}) = collect(offdiagonaldata(H))
function collectoffdiagonaldata{S,V,T,HS}(H::HierarchicalMatrix{S,V,T,HS})
    data = collect(offdiagonaldata(H))
    append!(data,mapreduce(collectoffdiagonaldata,vcat,diagonaldata(H)))
    data
end

collectdiagonaldata{S,V,T}(H::HierarchicalMatrix{S,V,T,NTuple{2,S}}) = collect(diagonaldata(H))
function collectdiagonaldata{S,V,T,HS}(H::HierarchicalMatrix{S,V,T,HS})
    data = S[]
    append!(data,mapreduce(collectdiagonaldata,vcat,diagonaldata(H)))
    data
end

isfactored(H::HierarchicalMatrix) = H.factored

function condest(H::HierarchicalMatrix)
    !isfactored(H) && factorize!(H)
    return cond(H.A)*mapreduce(condest,+,diagonaldata(H))
end

Base.convert{S,V,T,HS,HV}(::Type{HierarchicalMatrix{S,V,T,HS,HV}},M::HierarchicalMatrix) = HierarchicalMatrix(convert(Vector{S},collectdiagonaldata(M)),convert(Vector{V},collectoffdiagonaldata(M)))
Base.convert{T}(::Type{Matrix{T}},M::HierarchicalMatrix) = full(M)
Base.promote_rule{S,V,T,HS,HV,SS,VV,TT,HSS,HVV}(::Type{HierarchicalMatrix{S,V,T,HS,HV}},::Type{HierarchicalMatrix{SS,VV,TT,HSS,HVV}})=HierarchicalMatrix{promote_type(S,SS),promote_type(V,VV),promote_type(T,TT),promote_type(HS,HSS),promote_type(HV,HVV)}
Base.promote_rule{T,SS,VV,TT,HSS,HVV}(::Type{Matrix{T}},::Type{HierarchicalMatrix{SS,VV,TT,HSS,HVV}})=Matrix{promote_type(T,TT)}
Base.promote_rule{T,SS,VV,TT,HSS,HVV}(::Type{LowRankMatrix{T}},::Type{HierarchicalMatrix{SS,VV,TT,HSS,HVV}})=Matrix{promote_type(T,TT)}

Base.transpose(H::HierarchicalMatrix) = HierarchicalMatrix(map(transpose,diagonaldata(H)),map(transpose,reverse(offdiagonaldata(H))))
Base.ctranspose(H::HierarchicalMatrix) = HierarchicalMatrix(map(ctranspose,diagonaldata(H)),map(ctranspose,reverse(offdiagonaldata(H))))

function Base.size{S<:AbstractMatrix,V<:AbstractMatrix}(H::HierarchicalMatrix{S,V})
    H11,H22 = diagonaldata(H)
    H21,H12 = offdiagonaldata(H)
    m1,n2 = size(H12)
    m2,n1 = size(H21)
    @assert (m1,n1) == size(H11)
    @assert (m2,n2) == size(H22)
    m1+m2,n1+n2
end

function Base.getindex{S<:AbstractMatrix,V<:AbstractMatrix}(H::HierarchicalMatrix{S,V},i::Int,j::Int)
    H11,H22 = diagonaldata(H)
    H21,H12 = offdiagonaldata(H)
    m1,n2 = size(H12)
    m2,n1 = size(H21)
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

Base.copy(H::HierarchicalMatrix) = HierarchicalMatrix(map(copy,diagonaldata(H)),map(copy,offdiagonaldata(H)))
Base.copy!(H::HierarchicalMatrix,J::HierarchicalMatrix) = (map(copy!,diagonaldata(H),diagonaldata(J));map(copy!,offdiagonaldata(H),offdiagonaldata(J));H)

Base.rank(H::HierarchicalMatrix) = rank(full(H))
Base.cond(H::HierarchicalMatrix) = cond(full(H))

function blockrank(H::HierarchicalMatrix)
    m,n = blocksize(H)
    A = Array{Int64}(m,n)
    (m1,n1),(m2,n2) = map(blocksize,diagonaldata(H))
    r1,r2 = map(rank,offdiagonaldata(H))
    for j=1:n1,i=1:m2
        A[i+m1,j] = r1
    end
    for j=1:n2,i=1:m1
        A[i,j+n1] = r2
    end
    A11,A22 = map(blockrank,diagonaldata(H))
    for j=1:n1,i=1:m1
        A[i,j] = A11[i,j]
    end
    for j=1:n2,i=1:m2
        A[i+m1,j+n1] = A22[i,j]
    end
    A
end

for op in (:+,:-,:.+,:.-)
    @eval begin
        $op(H::HierarchicalMatrix) = HierarchicalMatrix(map($op,diagonaldata(H)),map($op,offdiagonaldata(H)))
        $op(H::HierarchicalMatrix,J::HierarchicalMatrix) = HierarchicalMatrix(map($op,diagonaldata(H),diagonaldata(J)),map($op,offdiagonaldata(H),offdiagonaldata(J)))
        $op(H::HierarchicalMatrix,L::LowRankMatrix) = $op(promote(H,L)...)
        $op(L::LowRankMatrix,H::HierarchicalMatrix) = $op(promote(L,H)...)
        $op(H::HierarchicalMatrix,A::Matrix) = $op(promote(H,A)...)
        $op(A::Matrix,H::HierarchicalMatrix) = $op(promote(A,H)...)
    end
end

function *(H::HierarchicalMatrix, x::HierarchicalVector)
    T = promote_type(eltype(H),eltype(x))
    A_mul_B!(similar(x,T),H,x)
end

function Base.A_mul_B!(b::AbstractVector,D::Base.LinAlg.Diagonal,x::AbstractVector)
    d = D.diag
    for i=1:min(length(b),length(x))
        @inbounds b[i] += d[i]*x[i]
    end
    b
end

function Base.A_mul_B!(b::HierarchicalVector,H::HierarchicalMatrix,h::HierarchicalVector)
    (H11,H22),(H21,H12) = partition(H)
    h1,h2 = partition(h)
    b1,b2 = partition(b)
    A_mul_B!(b1,H12,h2)
    A_mul_B!(b1,H11,h1)
    A_mul_B!(b2,H21,h1)
    A_mul_B!(b2,H22,h2)
    b
end

function *(H::HierarchicalMatrix,b::Vector)
    (H11,H22),(H21,H12) = partition(H)
    m1,m2 = size(H12,1),size(H21,1)
    (b1,b2) = (b[1:m1],b[1+m1:m1+m2])
    HierarchicalVector((H11*b1⊕H12*b2,H21*b1⊕H22*b2))
end


\(H::HierarchicalMatrix,b::AbstractVecOrMat) = full(H)\b
