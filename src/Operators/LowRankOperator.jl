export LowRankOperator

immutable LowRankOperator{S<:Space,T} <: InfiniteOperator{T}
    U::Vector{Fun{S,T}}
    V::Vector{Functional{T}}

    function LowRankOperator(U::Vector{Fun{S,T}},V::Vector{Functional{T}})
        @assert length(U) == length(V)
        @assert length(U) > 0
        ds=domainspace(first(V))
        for k=2:length(V)
            @assert domainspace(V[k])==ds
        end
        rs=space(first(U))
        for k=2:length(U)
            @assert space(U[k])==rs
        end
        new(U,V)
    end
end

domainspace(L::LowRankOperator)=domainspace(first(L.V))
rangespace(L::LowRankOperator)=space(first(L.U))

LowRankOperator{S,T}(U::Vector{Fun{S,T}},V::Vector{Functional{T}})=LowRankOperator{S,T}(U,V)
LowRankOperator{S,T1,T2}(U::Vector{Fun{S,T1}},V::Vector{Functional{T2}})=LowRankOperator(convert(Vector{Fun{S,promote_type(T1,T2)}},U),convert(Vector{Fun{M,promote_type(T1,T2)}},V))

Base.rank(L::LowRankOperator)=length(L.U)
