export LowRankIntegralOperator

immutable LowRankIntegralOperator{S<:Space,M<:Space,T} <: AbstractLowRankOperator{T}
    U::Vector{Fun{S,T}}
    V::Vector{Fun{M,T}}

    function (::Type{LowRankIntegralOperator{S,M,T}}){S,M,T}(U::Vector{Fun{S,T}},V::Vector{Fun{M,T}})
        @assert length(U) == length(V)
        @assert length(U) > 0
        ds=space(first(V))
        for k=2:length(V)
            @assert space(V[k])==ds
        end
        rs=space(first(U))
        for k=2:length(U)
            @assert space(U[k])==rs
        end
        new{S,M,T}(U,V)
    end
end

LowRankIntegralOperator{S,M,T}(U::Vector{Fun{S,T}},V::Vector{Fun{M,T}})=LowRankIntegralOperator{S,M,T}(U,V)


LowRankIntegralOperator{S,M,T1,T2}(U::Vector{Fun{S,T1}},V::Vector{Fun{M,T2}})=LowRankIntegralOperator(convert(Vector{Fun{S,promote_type(T1,T2)}},U),convert(Vector{Fun{M,promote_type(T1,T2)}},V))

LowRankIntegralOperator(A::Fun,B::Fun)=LowRankIntegralOperator([A],[B])

Base.promote_rule{S,M,T,S1,M1,T1}(::Type{LowRankIntegralOperator{S,M,T}},::Type{LowRankIntegralOperator{S1,M1,T1}})=LowRankIntegralOperator{promote_type(S,S1),promote_type(M,M1),promote_type(T,T1)}
Base.rank(L::AbstractLowRankOperator)=length(L.U)

domainspace(L::LowRankIntegralOperator)=space(first(L.V))
rangespace(L::LowRankIntegralOperator)=space(first(L.U))
promoterangespace(L::LowRankIntegralOperator,sp::Space)=LowRankIntegralOperator(map(u->Fun(u,sp),L.U),L.V)

+(L::LowRankIntegralOperator)=LowRankIntegralOperator(+L.U,L.V)
-(L::LowRankIntegralOperator)=LowRankIntegralOperator(-L.U,L.V)

*(L::LowRankIntegralOperator,f::Fun)=sum(map((u,v)->u*dot(v,f),L.U,L.V))

+(A::LowRankIntegralOperator,B::LowRankIntegralOperator)=LowRankIntegralOperator([A.U;B.U],[A.V;B.V])
-(A::LowRankIntegralOperator,B::LowRankIntegralOperator)=LowRankIntegralOperator([A.U;-B.U],[A.V;B.V])
