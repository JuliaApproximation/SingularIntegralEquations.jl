export HierarchicalVector, partitionvector, ishierarchical, degree

##
# Represent a binary hierarchical vector
##

abstract AbstractHierarchicalArray{SV,T,B,N} <: AbstractArray{T,N}

ishierarchical(A::AbstractArray) = false
ishierarchical{SV,T,B}(H::AbstractHierarchicalArray{SV,T,B}) = B
degree{SV,T}(H::AbstractHierarchicalArray{SV,T,false}) = 1

typealias AbstractHierarchicalVector{S,T,B} AbstractHierarchicalArray{S,T,B,1}

type HierarchicalVector{S,T,B} <: AbstractHierarchicalVector{S,T,B}
    data::NTuple{2,S} # Tuple of two S
    hierarchicaldata::NTuple{2,HierarchicalVector{S,T}} # Tuple of two HierarchicalVector{S,T}

    function HierarchicalVector(data::NTuple{2,S})
        H = new()
        H.data = data

        H
    end

    function HierarchicalVector(hierarchicaldata::NTuple{2,HierarchicalVector{S,T}})
        H = new()
        H.hierarchicaldata = hierarchicaldata

        H
    end
end

HierarchicalVector{S}(data::NTuple{2,S}) = HierarchicalVector{S,eltype(S),false}(data)
HierarchicalVector{S,T}(hierarchicaldata::NTuple{2,HierarchicalVector{S,T}}) = HierarchicalVector{S,T,true}(hierarchicaldata)

HierarchicalVector(data::Vector) = HierarchicalVector(data,round(Int,log2(length(data))))

function HierarchicalVector(data::Vector,n::Int)
    @assert length(data) == 2^n
    if n == 1
        return HierarchicalVector(tuple(data...))
    elseif n ≥ 2
        return HierarchicalVector((HierarchicalVector(data[1:2^(n-1)],n-1),HierarchicalVector(data[1+2^(n-1):end],n-1)))
    end
end


data{S,T}(H::HierarchicalVector{S,T,false}) = H.data
data{S,T}(H::HierarchicalVector{S,T,true}) = H.hierarchicaldata

degree{S,T}(H::HierarchicalVector{S,T,true}) = 1+degree(first(data(H)))

partitionvector(H::HierarchicalVector) = data(H)

function partitionvector{S,T,B}(H::Vector{HierarchicalVector{S,T,B}})
    n = length(H)
    H11,H12 = partitionvector(H[1])
    H1,H2 = fill(H11,n),fill(H12,n)
    for i=1:n
        H1[i],H2[i] = partitionvector(H[i])
    end
    H1,H2
end

collectdata{S,T}(H::HierarchicalVector{S,T,false}) = collect(data(H))
function collectdata{S,T}(H::HierarchicalVector{S,T,true})
    ret = S[]
    append!(ret,mapreduce(collectdata,vcat,data(H)))
    ret
end


Base.convert{S,T,B}(::Type{HierarchicalVector{S,T,B}},M::HierarchicalVector) = HierarchicalVector(convert(Vector{S},collectdata(M)))
Base.promote_rule{S,T,SS,TT}(::Type{HierarchicalVector{S,T,true}},::Type{HierarchicalVector{SS,TT,true}})=HierarchicalVector{promote_type(S,SS),promote_type(T,TT),true}
Base.promote_rule{S,T,SS,TT}(::Type{HierarchicalVector{S,T,false}},::Type{HierarchicalVector{SS,TT,false}})=HierarchicalVector{promote_type(S,SS),promote_type(T,TT),false}

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

        function $op{S,T,B,SS,TT,BB}(H::Array{HierarchicalVector{S,T,B}}, J::Array{HierarchicalVector{SS,TT,BB}})
            ret = similar(H)
            for i in eachindex(H,J)
                @inbounds ret[i] = $op(H[i], J[i])
            end
            ret
        end
        function $op{S,T,B,V}(H::Array{HierarchicalVector{S,T,B}}, J::Array{V})
            ret = similar(H)
            for i in eachindex(H,J)
                @inbounds ret[i] = $op(H[i], J[i])
            end
            ret
        end
        $op{S,T,B,V}(J::Array{V}, H::Array{HierarchicalVector{S,T,B}}) = $op(H,J)

        function $op(H::HierarchicalVector,J::HierarchicalVector)
            Hd,Jd = collectdata(H),collectdata(J)
            HierarchicalVector($op(Hd,Jd))
        end
        $op(H::HierarchicalVector) = HierarchicalVector($op(collectdata(H)))
    end
end
