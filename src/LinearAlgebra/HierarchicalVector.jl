

##
# Represent a power-of-two hierarchical vector
# in a binary tree.
##

export HierarchicalVector, partitionvector

type HierarchicalVector{S,T} <: AbstractVector{T}
    data::Union(@compat(Tuple{HierarchicalVector{S,T},HierarchicalVector{S,T}}),@compat(Tuple{S,S})) # n ≥ 2 ? Tuple of two HierarchicalVector{S,T} : Tuple of two S
    n::Int # Power of hierarchy (i.e. 2^n)

    function HierarchicalVector(data::Vector{S},n::Int)
        @assert length(data) == 2^n
        if n == 1
            return new(tuple(data...),n)
        elseif n ≥ 2
            return new((HierarchicalVector(data[1:2^(n-1)],n-1),HierarchicalVector(data[1+2^(n-1):end],n-1)),n)
        end
    end
    HierarchicalVector(data::Union(@compat(Tuple{HierarchicalVector{S,T},HierarchicalVector{S,T}}),@compat(Tuple{S,S})),n::Int) = new(data,n)
end


function HierarchicalVector{S}(data::Vector{S},n::Int)
    T = mapreduce(eltype,promote_type,data)
    HierarchicalVector{S,T}(data,n)
end
HierarchicalVector{S}(data::Vector{S})=HierarchicalVector(data,round(Int,log2(length(data))))

HierarchicalVector{S,T}(data::Union(@compat(Tuple{HierarchicalVector{S,T},HierarchicalVector{S,T}}),@compat(Tuple{S,S})),n::Int) = HierarchicalVector{S,T}(data,n)
function HierarchicalVector{S}(data::@compat(Tuple{S,S}),n::Int)
    T = promote_type(eltype(data[1]),eltype(data[2]))
    HierarchicalVector{S,T}(data,n)
end


function collectdata{S,T}(H::HierarchicalVector{S,T})
    data = S[]
    if H.n == 1
        push!(data,H.data...)
    elseif H.n ≥ 2
        append!(data,mapreduce(collectdata,vcat,H.data))
    end
    data
end


Base.eltype{S,T}(::HierarchicalVector{S,T})=T
Base.convert{U,V}(::Type{HierarchicalVector{U,V}},M::HierarchicalVector) = HierarchicalVector{U,V}(convert(Vector{U},collectdata(M)),M.n)
Base.promote_rule{S,T,U,V}(::Type{HierarchicalVector{S,T}},::Type{HierarchicalVector{U,V}})=HierarchicalVector{promote_type(S,U),promote_type(T,V)}
Base.size{S<:AbstractVector}(H::HierarchicalVector{S}) = (size(H.data[1])[1]+size(H.data[2])[1],)

function Base.getindex{S<:AbstractVector,T}(H::HierarchicalVector{S,T},i::Int)
    m1,m2 = length(H.data[1]),length(H.data[2])
    if 1 ≤ i ≤ m1
        return getindex(H.data[1],i)
    elseif m1 < i ≤ m1+m2
        return getindex(H.data[2],i-m1)
    else
        throw(BoundsError())
    end
end
Base.getindex{S<:AbstractVector,T}(H::HierarchicalVector{S,T},ir::Range) = eltype(H)[H[i] for i=ir].'#'
Base.full{S<:AbstractVector,T}(H::HierarchicalVector{S,T})=H[1:size(H,1)]

partitionvector(H::HierarchicalVector) = H.data

for op in (:+,:-)
    @eval begin
        function $op(H::HierarchicalVector,J::HierarchicalVector)
            @assert (n = H.n) == J.n
            Hd,Jd = collectdata(H),collectdata(J)
            HierarchicalVector($op(Hd,Jd),n)
        end
        $op(H::HierarchicalVector) = HierarchicalVector($op(collectdata(H)),H.n)
    end
end
