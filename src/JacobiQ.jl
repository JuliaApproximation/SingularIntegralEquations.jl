import ApproxFun: recA, recB, recC, recα, recβ, recγ
import ApproxFun: jacobirecA, jacobirecB, jacobirecC, jacobirecα, jacobirecβ, jacobirecγ


export JacobiQ, LegendreQ, WeightedJacobiQ


immutable JacobiQ{T,D<:Domain} <: RealUnivariateSpace{D}
    a::T
    b::T
    domain::D
end
LegendreQ(domain)=JacobiQ(0.,0.,domain)
LegendreQ()=LegendreQ(Segment())
JacobiQ(a,b,d::Domain)=JacobiQ(promote(a,b)...,d)
JacobiQ(a,b,d)=JacobiQ(a,b,Domain(d))
JacobiQ(a,b)=JacobiQ(a,b,Segment())


Base.promote_rule{T,V,D}(::Type{JacobiQ{T,D}},::Type{JacobiQ{V,D}})=JacobiQ{promote_type(T,V),D}
Base.convert{T,V,D}(::Type{JacobiQ{T,D}},J::JacobiQ{V,D})=JacobiQ{T,D}(J.a,J.b,J.domain)

WeightedJacobiQ{T,D} = JacobiQWeight{JacobiQ{T,D},D}

(::Type{WeightedJacobiQ})(α,β,d::Domain)=JacobiQWeight(α,β,JacobiQ(β,α,d))
(::Type{WeightedJacobiQ})(α,β)=JacobiQWeight(α,β,JacobiQ(β,α))

spacescompatible(a::JacobiQ,b::JacobiQ)=a.a==b.a && a.b==b.b

function canonicalspace(S::JacobiQ)
    #if isapproxinteger(S.a+0.5) && isapproxinteger(S.b+0.5)
    #    Chebyshev(domain(S))
    #else
        # return space with parameters in (-1,0.]
        JacobiQ(mod(S.a,-1),mod(S.b,-1),domain(S))
    #end
end

setdomain(S::JacobiQ,d::Domain)=JacobiQ(S.a,S.b,d)

for (REC,JREC) in ((:recα,:jacobirecα),(:recβ,:jacobirecβ),(:recγ,:jacobirecγ),
                   (:recA,:jacobirecA),(:recB,:jacobirecB),(:recC,:jacobirecC))
    @eval $REC{T}(::Type{T},sp::JacobiQ,k)=$JREC(T,sp.a,sp.b,k)
end

function stieltjes{T,D}(f::Fun{Jacobi{T,D}})
    g = Fun(f,Legendre(domain(f)))
    Fun(LegendreQ(domain(f)),2coefficients(g))
end
function stieltjes{D}(f::Fun{Chebyshev{D}})
    g = Fun(f,Legendre(domain(f)))
    Fun(LegendreQ(domain(f)),2coefficients(g))
end

function stieltjes{S,D}(f::Fun{JacobiWeight{S,D}})
    # Jacobi parameters need to transform to:
    α,β = f.space.β,f.space.α
    g = Fun(f,WeightedJacobi(α,β,domain(f)))
    Fun(WeightedJacobiQ(α,β,domain(f)),2coefficients(g))
end

evaluate{T,D<:Segment}(f::AbstractVector,S::JacobiQ{T,D},x) =
    stieltjesintervalrecurrence(S,f,tocanonical(S,x))./2jacobiQweight(S.b,S.a,tocanonical(S,x))
function evaluate{T,D<:Curve}(f::AbstractVector,S::JacobiQ{T,D},z::Number)
    sum(evaluate(f,setcanonicaldomain(S),complexroots(domain(S).curve-z)))
end
evaluate{T,D<:Curve}(f::AbstractVector,S::JacobiQ{T,D},z::AbstractArray) =
    reshape(promote_type(eltype(f),T,eltype(z))[ evaluate(f,S,z[i]) for i in eachindex(z) ], size(z))
