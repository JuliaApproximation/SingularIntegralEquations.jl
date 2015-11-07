export HierarchicalOperator

##
# Represent a binary hierarchical operator
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

type HierarchicalOperator{S,V,T,HS} <: BandedOperator{T}
    diagonaldata::HS
    offdiagonaldata::NTuple{2,V} # Tuple of two off-diagonal V

    A::Matrix{T} # Cache of matrix for pivot computation
    factorization::PivotLDU{T,Matrix{T}} # Cache of factorization of A for pivot computation
    factored::Bool

    function HierarchicalOperator(diagonaldata::HS,offdiagonaldata::NTuple{2,V})
        H = new()
        H.diagonaldata = diagonaldata
        H.offdiagonaldata = offdiagonaldata

        r1,r2 = map(rank,offdiagonaldata)
        H.A = eye(T,r1+r2,r1+r2)
        H.factorization = pivotldufact(H.A,r1,r2)
        H.factored = false

        H
    end
end

HierarchicalOperator{S,V}(diagonaldata::NTuple{2,S},offdiagonaldata::NTuple{2,V}) = HierarchicalOperator{S,V,promote_type(eltype(S),eltype(V)),NTuple{2,S}}(diagonaldata,offdiagonaldata)

HierarchicalOperator{S,V,V1,T,HS}(diagonaldata::Tuple{S,HierarchicalOperator{S,V1,T,HS}},offdiagonaldata::NTuple{2,V}) = HierarchicalOperator{S,V,promote_type(eltype(S),eltype(V),T),Tuple{S,HierarchicalOperator{S,V1,T,HS}}}(diagonaldata,offdiagonaldata)
HierarchicalOperator{S,V,V1,T,HS}(diagonaldata::Tuple{HierarchicalOperator{S,V1,T,HS},S},offdiagonaldata::NTuple{2,V}) = HierarchicalOperator{S,V,promote_type(eltype(S),eltype(V),T),Tuple{HierarchicalOperator{S,V1,T,HS},S}}(diagonaldata,offdiagonaldata)

HierarchicalOperator{S,V,V1,T,HS}(diagonaldata::NTuple{2,HierarchicalOperator{S,V1,T,HS}},offdiagonaldata::NTuple{2,V}) = HierarchicalOperator{S,V,promote_type(eltype(S),eltype(V),T),NTuple{2,HierarchicalOperator{S,V1,T,HS}}}(diagonaldata,offdiagonaldata)
HierarchicalOperator{S,V,V1,V2,T,HS1,HS2}(diagonaldata::Tuple{HierarchicalOperator{S,V1,T,HS1},HierarchicalOperator{S,V2,T,HS2}},offdiagonaldata::NTuple{2,V}) = HierarchicalOperator{S,V,promote_type(eltype(S),eltype(V),T),Tuple{HierarchicalOperator{S,V1,T,HS1},HierarchicalOperator{S,V2,T,HS2}}}(diagonaldata,offdiagonaldata)

HierarchicalOperator(diagonaldata::Vector,offdiagonaldata::Vector)=HierarchicalOperator(diagonaldata,offdiagonaldata,round(Int,log2(length(diagonaldata))))

function HierarchicalOperator(diagonaldata::Vector,offdiagonaldata::Vector,n::Int)
    @assert length(diagonaldata) == 2^n
    @assert length(offdiagonaldata) == 2^(n+1)-2

    if n == 1
        return HierarchicalOperator(tuple(diagonaldata...),tuple(offdiagonaldata...))
    elseif n ≥ 2
        dldp2 = div(length(offdiagonaldata)+2,2)
        return HierarchicalOperator((HierarchicalOperator(diagonaldata[1:2^(n-1)],offdiagonaldata[3:dldp2],n-1),
                                   HierarchicalOperator(diagonaldata[1+2^(n-1):end],offdiagonaldata[dldp2+1:end],n-1)),
                                   (offdiagonaldata[1],offdiagonaldata[2]))
    end
end

Base.similar(H::HierarchicalOperator) = HierarchicalOperator(map(similar,diagonaldata(H)),map(similar,offdiagonaldata(H)))
Base.similar{SS,V,T}(H::HierarchicalOperator{SS,V,T}, S) = HierarchicalOperator(map(A->similar(A,S),diagonaldata(H)),map(A->similar(A,S),offdiagonaldata(H)))

diagonaldata(H::HierarchicalOperator) = H.diagonaldata
offdiagonaldata(H::HierarchicalOperator) = H.offdiagonaldata

nlevels(H::HierarchicalOperator) = 1+mapreduce(nlevels,max,diagonaldata(H))

partition(H::HierarchicalOperator) = diagonaldata(H),offdiagonaldata(H)

collectoffdiagonaldata{S,V,T}(H::HierarchicalOperator{S,V,T,NTuple{2,S}}) = collect(offdiagonaldata(H))
function collectoffdiagonaldata{S,V,T,HS}(H::HierarchicalOperator{S,V,T,HS})
    data = collect(offdiagonaldata(H))
    append!(data,mapreduce(collectoffdiagonaldata,vcat,diagonaldata(H)))
    data
end

collectdiagonaldata{S,V,T}(H::HierarchicalOperator{S,V,T,NTuple{2,S}}) = collect(diagonaldata(H))
function collectdiagonaldata{S,V,T,HS}(H::HierarchicalOperator{S,V,T,HS})
    data = S[]
    append!(data,mapreduce(collectdiagonaldata,vcat,diagonaldata(H)))
    data
end

isfactored(H::HierarchicalOperator) = H.factored

function condest(H::HierarchicalOperator)
    !isfactored(H) && factorize!(H)
    return cond(H.A)*mapreduce(condest,+,diagonaldata(H))
end

Base.convert{S,V,T,HS}(::Type{HierarchicalOperator{S,V,T,HS}},M::HierarchicalOperator) = HierarchicalOperator(convert(Vector{S},collectdiagonaldata(M)),convert(Vector{V},collectoffdiagonaldata(M)))
Base.promote_rule{S,V,T,HS,SS,VV,TT,HSS}(::Type{HierarchicalOperator{S,V,T,HS}},::Type{HierarchicalOperator{SS,VV,TT,HSS}})=HierarchicalOperator{promote_type(S,SS),promote_type(V,VV),promote_type(T,TT),promote_type(HS,HSS)}

Base.transpose(H::HierarchicalOperator) = HierarchicalOperator(map(transpose,diagonaldata(H)),map(transpose,reverse(offdiagonaldata(H))))
Base.ctranspose(H::HierarchicalOperator) = HierarchicalOperator(map(ctranspose,diagonaldata(H)),map(ctranspose,reverse(offdiagonaldata(H))))

Base.copy(H::HierarchicalOperator) = HierarchicalOperator(map(copy,diagonaldata(H)),map(copy,offdiagonaldata(H)))
Base.copy!(H::HierarchicalOperator,J::HierarchicalOperator) = (map(copy!,diagonaldata(H),diagonaldata(J));map(copy!,offdiagonaldata(H),offdiagonaldata(J));H)

Base.rank(A::Operator)=Inf

function blockrank(H::HierarchicalOperator)
    n = nlevels(H)
    A = Array{Union{Float64,Int64}}(2^n,2^n)
    r1,r2 = map(rank,offdiagonaldata(H))
    for j=1:2^(n-1),i=1:2^(n-1)
        A[i+2^(n-1),j] = r1
        A[i,j+2^(n-1)] = r2
    end
    A11,A22 = map(blockrank,diagonaldata(H))
    for j=1:2^(n-1),i=1:2^(n-1)
        A[i,j] = A11[i,j]
        A[i+2^(n-1),j+2^(n-1)] = A22[i,j]
    end
    A
end

for op in (:+,:-,:.+,:.-)
    @eval begin
        $op(H::HierarchicalOperator) = HierarchicalOperator(map($op,diagonaldata(H)),map($op,offdiagonaldata(H)))
        $op(H::HierarchicalOperator,J::HierarchicalOperator) = HierarchicalOperator(map($op,diagonaldata(H),diagonaldata(J)),map($op,offdiagonaldata(H),offdiagonaldata(J)))
        #$op(H::HierarchicalOperator,L::LowRankMatrix) = $op(promote(H,L)...)
        #$op(L::LowRankMatrix,H::HierarchicalOperator) = $op(promote(L,H)...)
        #$op(H::HierarchicalOperator,A::Operator) = $op(promote(H,A)...)
        #$op(A::Operator,H::HierarchicalOperator) = $op(promote(A,H)...)
    end
end

#=
function *(H::HierarchicalOperator, x::HierarchicalVector)
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

function Base.A_mul_B!(b::HierarchicalVector,H::HierarchicalOperator,h::HierarchicalVector)
    (H11,H22),(H21,H12) = partition(H)
    h1,h2 = partition(h)
    b1,b2 = partition(b)
    A_mul_B!(b1,H12,h2)
    A_mul_B!(b1,H11,h1)
    A_mul_B!(b2,H21,h1)
    A_mul_B!(b2,H22,h2)
    b
end

function *(H::HierarchicalOperator,b::Vector)
    (H11,H22),(H21,H12) = partition(H)
    m1,m2 = size(H12,1),size(H21,1)
    (b1,b2) = (b[1:m1],b[1+m1:m1+m2])
    HierarchicalVector((H11*b1⊕H12*b2,H21*b1⊕H22*b2))
end
=#
