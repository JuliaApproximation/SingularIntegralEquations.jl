struct ComplexPlane <: Domain{ComplexF64} end

Base.in(x, d::ComplexPlane) = true

const ℂ = ComplexPlane()

canonicaldomain(d::ComplexPlane) = d
Base.reverse(C::ComplexPlane) = C

Base.intersect(a::ComplexPlane,b::ComplexPlane) = a
Base.union(a::ComplexPlane,b::ComplexPlane) = a



export JacobiQ, LegendreQ, WeightedJacobiQ


struct JacobiQ{D<:Domain,T} <: Space{D,T}
    a::T
    b::T
    domain::D
end
LegendreQ(domain) = JacobiQ(0.,0.,domain)
LegendreQ() = LegendreQ(ChebyshevInterval())
JacobiQ(a,b,d::Domain) = JacobiQ(promote(a,b)...,d)
JacobiQ(a,b,d) = JacobiQ(a,b,Domain(d))
JacobiQ(a,b) = JacobiQ(a,b,ChebyshevInterval())

domain(::JacobiQ) = ComplexPlane()

Base.promote_rule(::Type{JacobiQ{D,T}},::Type{JacobiQ{D,V}}) where {T,V,D} =
    JacobiQ{D,promote_type(T,V)}
convert(::Type{JacobiQ{D,T}},J::JacobiQ{D,V}) where {T,V,D} =
    JacobiQ{D,T}(J.a,J.b,J.domain)

const WeightedJacobiQ{D,T} = JacobiQWeight{JacobiQ{D,T},D}

WeightedJacobiQ(α,β,d::Domain) = JacobiQWeight(α,β,JacobiQ(β,α,d))
WeightedJacobiQ(α,β) = JacobiQWeight(α,β,JacobiQ(β,α))

spacescompatible(a::JacobiQ,b::JacobiQ) = a.a==b.a && a.b==b.b && a.domain == b.domain

function canonicalspace(S::JacobiQ)
    #if isapproxinteger(S.a+0.5) && isapproxinteger(S.b+0.5)
    #    Chebyshev(domain(S))
    #else
        # return space with parameters in (-1,0.]
        JacobiQ(mod(S.a,-1),mod(S.b,-1),S.domain)
    #end
end

for (REC,JREC) in ((:recα,:jacobirecα),(:recβ,:jacobirecβ),(:recγ,:jacobirecγ),
                   (:recA,:jacobirecA),(:recB,:jacobirecB),(:recC,:jacobirecC))
    @eval $REC(::Type{T},sp::JacobiQ,k) where {T} = $JREC(T,sp.a,sp.b,k)
end

stieltjes(f::Fun{<:Jacobi}) = Fun(LegendreQ(domain(f)),2coefficients(f,Legendre(domain(f))))
stieltjes(f::Fun{<:Chebyshev}) = Fun(LegendreQ(domain(f)),2coefficients(f,Legendre(domain(f))))


function stieltjes(f::Fun{<:JacobiWeight})
    # Jacobi parameters need to transform to:
    α,β = f.space.α,f.space.β
    Fun(WeightedJacobiQ(β,α,domain(f)),2coefficients(f,WeightedJacobi(β,α,domain(f))))
end

function evaluate(f::AbstractVector,S::JacobiQ{<:IntervalOrSegment},x)
    isinf(x) && return zero(promote_type(typeof(x), eltype(f)))
    stieltjesintervalrecurrence(S,f,mobius(S.domain,x))./2jacobiQweight(S.b,S.a,mobius(S.domain,x))
end
function evaluate(f::AbstractVector,S::JacobiQ{<:Curve},z::Number)
    isinf(z) && return zero(promote_type(typeof(z), eltype(f)))
    sum(evaluate(f,setcanonicaldomain(S),complexroots(S.domain.curve-z)))
end


# restrict to
function istieltjes(f::Fun{<:JacobiQ})
    # Jacobi parameters need to transform to:
    a,b = f.space.b,f.space.a
    d = f.space.domain
    a == b == 0 && return Fun(Legendre(d), coefficients(f)/2)
    Fun(WeightedJacobiQ(b,a,d),coefficients(g)/2)
end



stieltjes(f::Fun{<:ZeroSpace},z) = zero(promote_type(cfstype(f),typeof(z)))

istieltjes(f::Fun{<:ConstantSpace}) = Fun(ZeroSpace(),cfstype(f)[])
function istieltjes(f::Fun{<:SumSpace})
    is = istieltjes.(components(f))
    filter!(s -> !(space(s) isa ZeroSpace), is)
    Fun(is,PiecewiseSpace)
end

istieltjes(f::Fun{<:SumSpace{<:Tuple{<:Any,<:ConstantSpace}}}) =
    istieltjes(component(f,1))




istieltjes(f::Fun{<:ArraySpace}) = Fun(istieltjes.(Array(f)))


function union_rule(A::JacobiQ, B::JacobiQ)
    if A.domain == B.domain
        Jacobi(min(A.b,B.b),min(A.a,B.a),A.domain)
    else
        NoSpace()
    end
end
function maxspace_rule(A::JacobiQ,B::JacobiQ)
    A.domain == B.domain && return JacobiQ(max(A.b,B.b),max(A.a,B.a),A.domain)
    NoSpace()
end
