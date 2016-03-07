export HierarchicalVector, partition, nlevels, nchildren

partition(x)=x
nlevels(x)=0
nchildren(x)=0

##
# Represent an n-ary hierarchical vector
##

abstract AbstractHierarchicalArray{T,N,d} <: AbstractArray{T,d}

typealias AbstractHierarchicalVector{T,N} AbstractHierarchicalArray{T,N,1}

type HierarchicalVector{T,N} <: AbstractHierarchicalVector{T,N}
    data::NTuple{N,AbstractVector{T}}
end

data(H::HierarchicalVector) = H.data
partition(H::HierarchicalVector) = data(H)
nlevels(H::HierarchicalVector) = 1+mapreduce(nlevels,max,data(H))
nchildren{T,N}(H::HierarchicalVector{T,N}) = N

Base.similar(H::HierarchicalVector) = HierarchicalVector(map(similar,data(H)))
Base.similar{T,N}(H::HierarchicalVector{T,N}, S) = HierarchicalVector(map(x->similar(x,S),data(H)))
Base.convert{T,V,N}(::Type{AbstractVector{T}},M::HierarchicalVector{V,N}) = HierarchicalVector(map(x->convert(AbstractVector{T},x),data(M)))
Base.convert{T,V,N}(::Type{HierarchicalVector{T}},M::HierarchicalVector{V,N}) = HierarchicalVector(map(x->convert(AbstractVector{T},x),data(M)))
Base.convert{T,V,N}(::Type{HierarchicalVector{T,N}},M::HierarchicalVector{V,N}) = HierarchicalVector(map(x->convert(AbstractVector{T},x),data(M)))
Base.promote_rule{T,V,N}(::Type{HierarchicalVector{T,N}},::Type{HierarchicalVector{V,N}})=HierarchicalVector{promote_type(T,V),N}
Base.length(H::HierarchicalVector) = mapreduce(length,+,data(H))
Base.size(H::HierarchicalVector) = (length(H),)


function Base.getindex(H::HierarchicalVector, i::Int)
    j = 0
    for k in 1:nchildren(H)
        jₖ = length(H.data[k])
        if j < i ≤ j+jₖ return getindex(H.data[k], i-j) end
        j += jₖ
    end
    throw(BoundsError())
end
function Base.setindex!(H::HierarchicalVector, x, i::Int)
    j = 0
    for k in 1:nchildren(H)
        jₖ = length(H.data[k])
        if j < i ≤ j+jₖ return setindex!(H.data[k], x, i-j) end
        j += jₖ
    end
    throw(BoundsError())
end
Base.getindex(H::HierarchicalVector,ir::Range) = eltype(H)[H[i] for i in ir]
Base.full(H::HierarchicalVector)=H[1:size(H,1)]

# algebra

for op in (:+,:-,:.+,:.-,:.*)
    @eval begin
        $op{N}(a::Bool,H::HierarchicalVector{Bool,N}) = error("Not callable")
        $op{N}(H::HierarchicalVector{Bool,N},a::Bool) = error("Not callable")

        $op(H::HierarchicalVector) = HierarchicalVector(map($op,data(H)))
        $op(H::HierarchicalVector,a::Number) = HierarchicalVector(map(x->$op(x,a),data(H)))
        $op(a::Number,H::HierarchicalVector) = HierarchicalVector(map(x->$op(a,x),data(H)))

        $op(H::HierarchicalVector,J::HierarchicalVector) = HierarchicalVector(map($op,data(H),data(J)))
        $op(H::HierarchicalVector,J::Vector) = $op(full(H),J)
        $op(J::Vector,H::HierarchicalVector) = $op(J,full(H))
    end
end

*(H::HierarchicalVector,a::Number) = HierarchicalVector(map(x->x*a,data(H)))
*(a::Number,H::HierarchicalVector) = H*a

for op in (:(Base.dot),:dotu)
    @eval begin
        $op(H::HierarchicalVector,J::HierarchicalVector) = sum(map($op,data(H),data(J)))
        $op(H::HierarchicalVector,J::Vector) = $op(full(H),J)
        $op(J::Vector,H::HierarchicalVector) = $op(J,full(H))
    end
end

function Base.cumsum(H::HierarchicalVector)
    CS = map(cumsum,data(H))
    cs = cumsum(collect(map(sum,data(H))))
    for i in 2:nchildren(H), j in 1:length(CS[i])
        CS[i][j] += cs[i-1]
    end
    HierarchicalVector(CS)
end
Base.sum(H::HierarchicalVector) = mapreduce(sum,+,data(H))
Base.conj!(H::HierarchicalVector) = (map(conj!,data(H));H)
Base.copy!(H::HierarchicalVector,J::HierarchicalVector) = (map(copy!,data(H),data(J));H)

for op in (:(Base.transpose),:(Base.ctranspose))
    @eval begin
        $op(H::HierarchicalVector) = $op(full(H))
    end
end

for op in (:(Base.zero),:(Base.ones),:(Base.abs),:(Base.abs2),:(Base.conj),:(Base.copy),:.^)
    @eval begin
        $op(H::HierarchicalVector) = HierarchicalVector(map($op,data(H)))
    end
end

Base.A_mul_B!(b::Vector,A::Matrix,H::HierarchicalVector) = A_mul_B!(b,A,full(H))
Base.A_mul_B!(b::Vector,A::LowRankMatrix,H::HierarchicalVector) = A_mul_B!(b,A,full(H))
