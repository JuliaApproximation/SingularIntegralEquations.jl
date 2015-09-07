export LowRankOperator

immutable LowRankOperator{S<:FunctionSpace,M<:FunctionSpace,T} <: InfiniteOperator{T}
    U::Vector{Fun{S,T}}
    V::Vector{Fun{M,T}}

    function LowRankOperator(U::Vector{Fun{S,T}},V::Vector{Fun{M,T}})
        @assert length(U) == length(V)
        @assert length(U) > 0
        new(U,V)
    end
end

LowRankOperator{S,M,T}(U::Vector{Fun{S,T}},V::Vector{Fun{M,T}})=LowRankOperator{S,M,T}(U,V)
LowRankOperator{S,M,T1,T2}(U::Vector{Fun{S,T1}},V::Vector{Fun{M,T2}})=LowRankOperator(convert(Vector{Fun{S,promote_type(T1,T2)}},U),convert(Vector{Fun{M,promote_type(T1,T2)}},V))

Base.rank(L::LowRankOperator)=length(L.U)
