export LowRankOperator

immutable LowRankOperator{S<:FunctionSpace,M<:FunctionSpace,T1,T2} <: InfiniteOperator{promote_type(T1,T2)}
    U::Vector{Fun{S,T1}}
    V::Vector{Fun{M,T2}}

    function LowRankOperator(U::Vector{Fun{S,T1}},V::Vector{Fun{M,T2}})
        @assert length(U) == length(V)
        @assert length(U) > 0
        new(U,V)
    end
end

LowRankOperator{S,M,T1,T2}(U::Vector{Fun{S,T1}},V::Vector{Fun{M,T2}})=LowRankOperator{S,M,T1,T2}(U,V)


Base.rank(L::LowRankOperator)=length(L.U)
#Base.size(L::LowRankFun,k::Integer)=k==1?mapreduce(length,max,f.A):mapreduce(length,max,f.B)
#Base.size(f::LowRankFun)=size(f,1),size(f,2)
#Base.eltype{S,M,SS,T,V}(::LowRankFun{S,M,SS,T,V})=promote_type(T,V)

LowRankOperator{L<:LowRankFun}(G::GreensFun{L}) = LowRankOperator(slices(G)...)
