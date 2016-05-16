import ApproxFun: recA, recB, recC, recα, recβ, recγ
import ApproxFun: jacobirecA, jacobirecB, jacobirecC, jacobirecα, jacobirecβ, jacobirecγ


export JacobiQ, LegendreQ, WeightedJacobiQ


immutable JacobiQ{T,D<:Domain} <: RealUnivariateSpace{D}
    a::T
    b::T
    domain::D
end
LegendreQ(domain)=JacobiQ(0.,0.,domain)
LegendreQ()=LegendreQ(Interval())
JacobiQ(a,b,d::Domain)=JacobiQ{promote_type(typeof(a),typeof(b)),typeof(d)}(a,b,d)
JacobiQ(a,b,d)=JacobiQ(a,b,Domain(d))
JacobiQ(a,b)=JacobiQ(a,b,Interval())
#JacobiQ{m}(A::Ultraspherical{m})=JacobiQ(m-0.5,m-0.5,domain(A))


Base.promote_rule{T,V,D}(::Type{JacobiQ{T,D}},::Type{JacobiQ{V,D}})=JacobiQ{promote_type(T,V),D}
Base.convert{T,V,D}(::Type{JacobiQ{T,D}},J::JacobiQ{V,D})=JacobiQ{T,D}(J.a,J.b,J.domain)

typealias WeightedJacobiQ{D} JacobiQWeight{JacobiQ{Float64,D},D}

Base.call(::Type{WeightedJacobiQ},α,β,d::Domain)=JacobiQWeight(α,β,JacobiQ(β,α,d))
Base.call(::Type{WeightedJacobiQ},α,β)=JacobiQWeight(α,β,JacobiQ(β,α))

spacescompatible(a::JacobiQ,b::JacobiQ)=a.a==b.a && a.b==b.b

function canonicalspace(S::JacobiQ)
    #if isapproxinteger(S.a+0.5) && isapproxinteger(S.b+0.5)
    #    Chebyshev(domain(S))
    #else
        # return space with parameters in (-1,0.]
        JacobiQ(mod(S.a,-1),mod(S.b,-1),domain(S))
    #end
end

for (REC,JREC) in ((:recα,:jacobirecα),(:recβ,:jacobirecβ),(:recγ,:jacobirecγ),
                   (:recA,:jacobirecA),(:recB,:jacobirecB),(:recC,:jacobirecC))
    @eval $REC{T}(::Type{T},sp::JacobiQ,k)=$JREC(T,sp.a,sp.b,k)
end

function stieltjes{T,D}(f::Fun{Jacobi{T,D}})
    @assert f.space.a == f.space.b == 0
    Fun(coefficients(f),WeightedJacobiQ(0,0,domain(f)))
end
stieltjes{D}(f::Fun{WeightedJacobi{D}}) = Fun(coefficients(f),WeightedJacobiQ(f.space.α,f.space.β,domain(f)))

evaluate(f::AbstractVector,S::JacobiQ,x) = stieltjesintervalrecurrence(S,f,tocanonical(S,x))
