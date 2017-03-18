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

type HierarchicalOperator{S,V,T,HS,HV} <: Operator{T}
    diagonaldata::HS
    offdiagonaldata::HV

    A::Matrix{T} # Cache of matrix for pivot computation
    factorization::PivotLDU{T,Matrix{T}} # Cache of factorization of A for pivot computation
    factored::Bool

    function HierarchicalOperator{S,V,T,HS,HV}(diagonaldata::HS,offdiagonaldata::HV) where {S,V,T,HS,HV}
        H = new()
        H.diagonaldata = diagonaldata
        H.offdiagonaldata = offdiagonaldata

        r = mapreduce(rank,+,offdiagonaldata)
        H.A = eye(T,r,r)
        H.factored = false

        H
    end
end


HierarchicalOperator{S1,S2,V1,V2}(diagonaldata::Tuple{S1,S2},offdiagonaldata::Tuple{V1,V2}) = HierarchicalOperator{promote_type(S1,S2),promote_type(V1,V2),promote_type(eltype(S1),eltype(S2),eltype(V1),eltype(V2)),Tuple{S1,S2},Tuple{V1,V2}}(diagonaldata,offdiagonaldata)
HierarchicalOperator{S1,S2,V1,V2,V3,V4,T1,T2,HS1,HS2,HV1,HV2}(diagonaldata::Tuple{HierarchicalOperator{S1,V1,T1,HS1,HV1},HierarchicalOperator{S2,V2,T2,HS2,HV2}},offdiagonaldata::Tuple{V3,V4}) = HierarchicalOperator{promote_type(S1,S2),promote_type(V1,V2,V3,V4),promote_type(eltype(S1),eltype(S2),eltype(V1),eltype(V2),eltype(V3),eltype(V4),T1,T2),Tuple{HierarchicalOperator{S1,V1,T1,HS1,HV1},HierarchicalOperator{S2,V2,T2,HS2,HV2}},Tuple{V3,V4}}(diagonaldata,offdiagonaldata)
HierarchicalOperator{S,S1,V,V1,V2,T,HS,HV}(diagonaldata::Tuple{S1,HierarchicalOperator{S,V,T,HS,HV}},offdiagonaldata::Tuple{V1,V2}) = HierarchicalOperator{promote_type(S,S1),promote_type(V,V1,V2),promote_type(eltype(S),eltype(S1),eltype(V),eltype(V1),eltype(V2),T),Tuple{S1,HierarchicalOperator{S,V,T,HS,HV}},Tuple{V1,V2}}(diagonaldata,offdiagonaldata)
HierarchicalOperator{S,S1,V,V1,V2,T,HS,HV}(diagonaldata::Tuple{HierarchicalOperator{S,V,T,HS,HV},S1},offdiagonaldata::Tuple{V1,V2}) = HierarchicalOperator{promote_type(S,S1),promote_type(V,V1,V2),promote_type(eltype(S),eltype(S1),eltype(V),eltype(V1),eltype(V2),T),Tuple{HierarchicalOperator{S,V,T,HS,HV},S1},Tuple{V1,V2}}(diagonaldata,offdiagonaldata)


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

Base.convert{S,V,T,HS,HV}(::Type{HierarchicalOperator{S,V,T,HS,HV}},M::HierarchicalOperator) = HierarchicalOperator(convert(Vector{S},collectdiagonaldata(M)),convert(Vector{V},collectoffdiagonaldata(M)))
Base.promote_rule{S,V,T,HS,HV,SS,VV,TT,HSS,HVV}(::Type{HierarchicalOperator{S,V,T,HS,HV}},::Type{HierarchicalOperator{SS,VV,TT,HSS,HVV}})=HierarchicalOperator{promote_type(S,SS),promote_type(V,VV),promote_type(T,TT),promote_type(HS,HSS),promote_type(HV,HVV)}

Base.transpose(H::HierarchicalOperator) = HierarchicalOperator(map(transpose,diagonaldata(H)),map(transpose,reverse(offdiagonaldata(H))))
Base.ctranspose(H::HierarchicalOperator) = HierarchicalOperator(map(ctranspose,diagonaldata(H)),map(ctranspose,reverse(offdiagonaldata(H))))

Base.copy(H::HierarchicalOperator) = HierarchicalOperator(map(copy,diagonaldata(H)),map(copy,offdiagonaldata(H)))
Base.copy!(H::HierarchicalOperator,J::HierarchicalOperator) = (map(copy!,diagonaldata(H),diagonaldata(J));map(copy!,offdiagonaldata(H),offdiagonaldata(J));H)

Base.rank(A::Operator) = ∞

function blockrank(H::HierarchicalOperator)
    m,n = blocksize(H)
    A = Array{Number}(m,n)
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

for OP in (:domainspace,:rangespace)
    @eval $OP{S<:Operator,V<:AbstractLowRankOperator}(H::HierarchicalOperator{S,V})=PiecewiseSpace(map($OP,diagonaldata(H))...)
end

blocksize{U<:Operator,V<:AbstractLowRankOperator}(H::HierarchicalOperator{U,V}) = length(domainspace(H)),length(rangespace(H)) # TODO: check it's not rangespace...domainspace

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
