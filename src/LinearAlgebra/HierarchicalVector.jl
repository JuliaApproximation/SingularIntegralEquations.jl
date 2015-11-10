export HierarchicalVector, partition, nlevels

partition(x)=x
nlevels(A)=0

##
# Represent a binary hierarchical vector
##

abstract AbstractHierarchicalArray{SV,T,HS,N} <: AbstractArray{T,N}

typealias AbstractHierarchicalVector{S,T,HS} AbstractHierarchicalArray{S,T,HS,1}

type HierarchicalVector{S,T,HS} <: AbstractHierarchicalVector{S,T,HS}
    data::HS
    HierarchicalVector(data::HS) = new(data)
end

HierarchicalVector{S}(data::NTuple{2,S}) = HierarchicalVector{S,eltype(S),NTuple{2,S}}(data)

HierarchicalVector{S,T,HS}(data::Tuple{S,HierarchicalVector{S,T,HS}}) = HierarchicalVector{S,T,Tuple{S,HierarchicalVector{S,T,HS}}}(data)
HierarchicalVector{S,T,HS}(data::Tuple{HierarchicalVector{S,T,HS},S}) = HierarchicalVector{S,T,Tuple{HierarchicalVector{S,T,HS},S}}(data)

HierarchicalVector{S,T,HS}(data::NTuple{2,HierarchicalVector{S,T,HS}}) = HierarchicalVector{S,T,NTuple{2,HierarchicalVector{S,T,HS}}}(data)
HierarchicalVector{S,T,HS1,HS2}(data::Tuple{HierarchicalVector{S,T,HS1},HierarchicalVector{S,T,HS2}}) = HierarchicalVector{S,T,Tuple{HierarchicalVector{S,T,HS1},HierarchicalVector{S,T,HS2}}}(data)

HierarchicalVector(data::Vector) = HierarchicalVector(data,round(Int,log2(length(data))))

function HierarchicalVector(data::Vector,n::Int)
    @assert length(data) == 2^n
    if n == 1
        return HierarchicalVector(tuple(data...))
    elseif n ≥ 2
        return HierarchicalVector((HierarchicalVector(data[1:2^(n-1)],n-1),HierarchicalVector(data[1+2^(n-1):end],n-1)))
    end
end

Base.similar(H::HierarchicalVector) = HierarchicalVector(map(similar,data(H)))
Base.similar{SS,V,T}(H::HierarchicalVector{SS,V,T}, S) = HierarchicalVector(map(A->similar(A,S),data(H)))

data(H::HierarchicalVector) = H.data

nlevels(H::HierarchicalVector) = 1+mapreduce(nlevels,max,data(H))

partition(H::HierarchicalVector) = data(H)

function partition(H::Vector{HierarchicalVector})
    n = length(H)
    H11,H12 = partition(H[1])
    H1,H2 = fill(H11,n),fill(H12,n)
    for i=1:n
        H1[i],H2[i] = partition(H[i])
    end
    H1,H2
end

collectdata{S,T}(H::HierarchicalVector{S,T,NTuple{2,S}}) = collect(data(H))
collectdata{S,T,HS}(H::HierarchicalVector{S,T,Tuple{S,HierarchicalVector{S,T,HS}}}) = vcat(H.data[1],collectdata(H.data[2]))
collectdata{S,T,HS}(H::HierarchicalVector{S,T,Tuple{HierarchicalVector{S,T,HS},S}}) = vcat(collectdata(H.data[1]),H.data[2])
function collectdata{S}(H::HierarchicalVector{S})
    ret = S[]
    append!(ret,mapreduce(collectdata,vcat,data(H)))
    ret
end


Base.convert{S,T,HS}(::Type{HierarchicalVector{S,T,HS}},M::HierarchicalVector) = HierarchicalVector(convert(Vector{S},collectdata(M)))
Base.promote_rule{S,T,HS,SS,TT,HSS}(::Type{HierarchicalVector{S,T,HS}},::Type{HierarchicalVector{SS,TT,HSS}})=HierarchicalVector{promote_type(S,SS),promote_type(T,TT),promote_type(HS,HSS)}

function Base.size{S<:AbstractVector}(H::HierarchicalVector{S})
    H1,H2 = data(H)
    (size(H1)[1]+size(H2)[1],)
end

function Base.size{S<:Number}(H::HierarchicalVector{S})
    H1,H2 = data(H)
    if typeof(H1) <: S && typeof(H2) <: S
        return (2,)
    elseif typeof(H1) <: S
        return (1+size(H2)[1],)
    elseif typeof(H2) <: S
        return (size(H1)[1]+1,)
    else
        return (size(H1)[1]+size(H2)[1],)
    end
end

function Base.getindex{S<:Union{Number,AbstractVector}}(H::HierarchicalVector{S},i::Int)
    H1,H2 = data(H)
    m1,m2 = length(H1),length(H2)
    if 1 ≤ i ≤ m1
        return getindex(H1,i)
    elseif m1 < i ≤ m1+m2
        return getindex(H2,i-m1)
    else
        throw(BoundsError())
    end
end
Base.getindex{S<:Union{Number,AbstractVector}}(H::HierarchicalVector{S},ir::Range) = eltype(H)[H[i] for i=ir]
Base.full{S<:Union{Number,AbstractVector}}(H::HierarchicalVector{S})=H[1:size(H,1)]

# algebra

for op in (:+,:-,:.+,:.-,:.*)
    @eval begin
        $op{S}(a::Bool,H::HierarchicalVector{S,Bool}) = error("Not callable")
        $op{S}(H::HierarchicalVector{S,Bool},a::Bool) = error("Not callable")

        $op(H::HierarchicalVector) = HierarchicalVector(map($op,data(H)))
        $op(H::HierarchicalVector,a::Number) = HierarchicalVector(($op(H.data[1],a),$op(H.data[2],a)))
        $op(a::Number,H::HierarchicalVector) = HierarchicalVector(($op(a,H.data[1]),$op(a,H.data[2])))

        $op(H::HierarchicalVector,J::HierarchicalVector) = HierarchicalVector(map($op,data(H),data(J)))
        $op(H::HierarchicalVector,J::Vector) = $op(full(H),J)
        $op(J::Vector,H::HierarchicalVector) = $op(J,full(H))
    end
end

*(H::HierarchicalVector,a::Number) = HierarchicalVector((H.data[1]*a,H.data[2]*a))
*(a::Number,H::HierarchicalVector) = H*a

for op in (:(Base.dot),:dotu)
    @eval begin
        $op(H::HierarchicalVector,J::HierarchicalVector) = $op(H.data[1],J.data[1])+$op(H.data[2],J.data[2])
        $op(H::HierarchicalVector,J::Vector) = $op(full(H),J)
        $op(J::Vector,H::HierarchicalVector) = $op(J,full(H))
    end
end

Base.cumsum(H::HierarchicalVector) = HierarchicalVector((cumsum(H.data[1]),sum(H.data[1])+cumsum(H.data[2])))
Base.conj!(H::HierarchicalVector) = (map(conj!,data(H));H)
Base.copy!(H::HierarchicalVector,J::HierarchicalVector) = (map(copy!,data(H),data(J));H)

for op in (:(Base.zero),:(Base.ones),:(Base.abs),:(Base.abs2),:(Base.conj),:(Base.copy),:.^)
    @eval begin
        $op(H::HierarchicalVector) = HierarchicalVector(map($op,data(H)))
    end
end

Base.A_mul_B!(b::Vector,A::Matrix,H::HierarchicalVector) = A_mul_B!(b,A,full(H))
Base.A_mul_B!(b::Vector,A::LowRankMatrix,H::HierarchicalVector) = A_mul_B!(b,A,full(H))


#=
        function $op{S,T,HS,SS,TT,HSS}(H::Array{HierarchicalVector{S,T,HS}}, J::Array{HierarchicalVector{SS,TT,HSS}})
            ret = similar(H)
            for i in eachindex(H,J)
                @inbounds ret[i] = $op(H[i], J[i])
            end
            ret
        end
        function $op{S,T,HS,V}(H::Array{HierarchicalVector{S,T,HS}}, J::Array{V})
            ret = similar(H)
            for i in eachindex(H,J)
                @inbounds ret[i] = $op(H[i], J[i])
            end
            ret
        end
        $op{S,T,HS,V}(J::Array{V}, H::Array{HierarchicalVector{S,T,HS}}) = $op(H,J)
=#
