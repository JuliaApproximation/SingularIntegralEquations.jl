export HierarchicalVector, partitionvector, ishierarchical, degree

##
# Represent a binary hierarchical vector
##

abstract AbstractHierarchicalArray{SV,T,HS,N} <: AbstractArray{T,N}

degree{S,T,N}(::AbstractHierarchicalArray{S,T,NTuple{2,S},N}) = 1
degree{S,V,T,N}(::AbstractHierarchicalArray{Tuple{S,V},T,NTuple{2,S},N}) = 1

degree{S,T,HS,N}(::AbstractHierarchicalArray{S,T,NTuple{2,HS},N}) = 1+degree(super(HS))
degree{S,V,T,HS,N}(::AbstractHierarchicalArray{Tuple{S,V},T,NTuple{2,HS},N}) = 1+degree(super(HS))

degree{S,T,N}(::Type{AbstractHierarchicalArray{S,T,NTuple{2,S},N}}) = 1
degree{S,V,T,N}(::Type{AbstractHierarchicalArray{Tuple{S,V},T,NTuple{2,S},N}}) = 1

degree{S,T,HS,N}(::Type{AbstractHierarchicalArray{S,T,NTuple{2,HS},N}}) = 1+degree(super(HS))
degree{S,V,T,HS,N}(::Type{AbstractHierarchicalArray{Tuple{S,V},T,NTuple{2,HS},N}}) = 1+degree(super(HS))

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


data(H::HierarchicalVector) = H.data

partitionvector(H::HierarchicalVector) = data(H)

function partitionvector(H::Vector{HierarchicalVector})
    n = length(H)
    H11,H12 = partitionvector(H[1])
    H1,H2 = fill(H11,n),fill(H12,n)
    for i=1:n
        H1[i],H2[i] = partitionvector(H[i])
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

function Base.getindex{S<:AbstractVector}(H::HierarchicalVector{S},i::Int)
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
Base.getindex{S<:AbstractVector}(H::HierarchicalVector{S},ir::Range) = eltype(H)[H[i] for i=ir]
Base.full{S<:AbstractVector}(H::HierarchicalVector{S})=H[1:size(H,1)]

# algebra

for op in (:+,:-,:.+,:.-,:.*)
    @eval begin
        $op{S}(a::Bool,H::HierarchicalVector{S,Bool}) = error("Not callable")
        $op{S}(H::HierarchicalVector{S,Bool},a::Bool) = error("Not callable")

        $op(H::HierarchicalVector) = HierarchicalVector(map($op,data(H)))
        $op(H::HierarchicalVector,a::Number) = HierarchicalVector(($op(H.data[1],a),$op(H.data[2],a)))
        $op(a::Number,H::HierarchicalVector) = $op(H,a)

        $op(H::HierarchicalVector,J::HierarchicalVector) = HierarchicalVector(map($op,data(H),data(J)))
        $op(H::HierarchicalVector,J::Vector) = $op(full(H),J)
        $op(J::Vector,H::HierarchicalVector) = $op(H,J)
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

for op in (:(Base.zero),:(Base.ones),:(Base.abs),:(Base.abs2),:(Base.conj),:(Base.copy),:.^)
    @eval begin
        $op(H::HierarchicalVector) = HierarchicalVector(map($op,data(H)))
    end
end

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
