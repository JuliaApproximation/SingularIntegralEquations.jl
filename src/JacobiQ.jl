import ApproxFun: recA, recB, recC, recα, recβ, recγ
import ApproxFun: jacobirecA, jacobirecB, jacobirecC, jacobirecα, jacobirecβ, jacobirecγ


export JacobiQ, LegendreQ, WeightedJacobiQ


struct JacobiQ{D<:Domain,T} <: Space{D,T}
    a::T
    b::T
    domain::D
end
LegendreQ(domain) = JacobiQ(0.,0.,domain)
LegendreQ() = LegendreQ(Segment())
JacobiQ(a,b,d::Domain) = JacobiQ(promote(a,b)...,d)
JacobiQ(a,b,d) = JacobiQ(a,b,Domain(d))
JacobiQ(a,b) = JacobiQ(a,b,Segment())


Base.promote_rule(::Type{JacobiQ{D,T}},::Type{JacobiQ{D,V}}) where {T,V,D} =
    JacobiQ{D,promote_type(T,V)}
convert(::Type{JacobiQ{D,T}},J::JacobiQ{D,V}) where {T,V,D} =
    JacobiQ{D,T}(J.a,J.b,J.domain)

const WeightedJacobiQ{D,T} = JacobiQWeight{JacobiQ{D,T},D}

WeightedJacobiQ(α,β,d::Domain) = JacobiQWeight(α,β,JacobiQ(β,α,d))
WeightedJacobiQ(α,β) = JacobiQWeight(α,β,JacobiQ(β,α))

spacescompatible(a::JacobiQ,b::JacobiQ)=a.a==b.a && a.b==b.b

function canonicalspace(S::JacobiQ)
    #if isapproxinteger(S.a+0.5) && isapproxinteger(S.b+0.5)
    #    Chebyshev(domain(S))
    #else
        # return space with parameters in (-1,0.]
        JacobiQ(mod(S.a,-1),mod(S.b,-1),domain(S))
    #end
end

setdomain(S::JacobiQ,d::Domain) = JacobiQ(S.a,S.b,d)

for (REC,JREC) in ((:recα,:jacobirecα),(:recβ,:jacobirecβ),(:recγ,:jacobirecγ),
                   (:recA,:jacobirecA),(:recB,:jacobirecB),(:recC,:jacobirecC))
    @eval $REC(::Type{T},sp::JacobiQ,k) where {T} = $JREC(T,sp.a,sp.b,k)
end

function stieltjes(f::Fun{<:Jacobi})
    g = Fun(f,Legendre(domain(f)))
    Fun(LegendreQ(domain(f)),2coefficients(g))
end
function stieltjes(f::Fun{<:Chebyshev})
    g = Fun(f,Legendre(domain(f)))
    Fun(LegendreQ(domain(f)),2coefficients(g))
end

function stieltjes(f::Fun{<:JacobiWeight})
    # Jacobi parameters need to transform to:
    α,β = f.space.β,f.space.α
    g = Fun(f,WeightedJacobi(α,β,domain(f)))
    Fun(WeightedJacobiQ(α,β,domain(f)),2coefficients(g))
end

evaluate(f::AbstractVector,S::JacobiQ{<:Segment},x) =
    stieltjesintervalrecurrence(S,f,mobius(S,x))./2jacobiQweight(S.b,S.a,mobius(S,x))
evaluate(f::AbstractVector,S::JacobiQ{<:Curve},z::Number) =
    sum(evaluate(f,setcanonicaldomain(S),complexroots(domain(S).curve-z)))
