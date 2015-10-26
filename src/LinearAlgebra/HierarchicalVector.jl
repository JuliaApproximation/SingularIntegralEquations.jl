export HierarchicalVector, partitionvector, ishierarchical, degree

##
# Represent a binary hierarchical vector
##

abstract AbstractHierarchicalArray{SV,T,HS,N} <: AbstractArray{T,N}

degree{S,T,N}(::AbstractHierarchicalArray{S,T,S,N}) = 1
degree{S,V,T,N}(::AbstractHierarchicalArray{Tuple{S,V},T,S,N}) = 1

degree{S,T,HS,N}(::AbstractHierarchicalArray{S,T,HS,N}) = 1+degree(super(HS))
degree{S,V,T,HS,N}(::AbstractHierarchicalArray{Tuple{S,V},T,HS,N}) = 1+degree(super(HS))

degree{S,T,N}(::Type{AbstractHierarchicalArray{S,T,S,N}}) = 1
degree{S,V,T,N}(::Type{AbstractHierarchicalArray{Tuple{S,V},T,S,N}}) = 1

degree{S,T,HS,N}(::Type{AbstractHierarchicalArray{S,T,HS,N}}) = 1+degree(super(HS))
degree{S,V,T,HS,N}(::Type{AbstractHierarchicalArray{Tuple{S,V},T,HS,N}}) = 1+degree(super(HS))

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


for op in (:+,:-)
    @eval begin
        function $op(H::HierarchicalVector,J::Vector)
            Hd = collectdata(H)
            nd = cumsum(map(length,Hd))
            ret = similar(Hd)
            ret[1] = $op(Hd[1],J[1:nd[1]])
            for i=2:length(Hd)
                @inbounds ret[i] = $op(Hd[i],J[1+nd[i-1]:nd[i]])
            end
            HierarchicalVector(ret)
        end
        $op(J::Vector,H::HierarchicalVector) = $op(H,J)

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

        function $op(H::HierarchicalVector,J::HierarchicalVector)
            Hd,Jd = collectdata(H),collectdata(J)
            HierarchicalVector($op(Hd,Jd))
        end
        $op(H::HierarchicalVector) = HierarchicalVector($op(collectdata(H)))
    end
end
