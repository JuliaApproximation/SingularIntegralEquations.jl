import ApproxFunBase: weight, dimension

export JacobiQWeight

"""
`JacobiQWeight`
weights a basis on `ℂ\\𝕀` weighted by `(z+1)^α*(z-1)^β`.
Note the inconsistency of the parameters with `JacobiQ`.
When the domain is `[a,b]` the weight is inferred by mapping to `[-1,1]`
"""
struct JacobiQWeight{S,DD} <: WeightSpace{S,DD,Float64}
    α::Float64
    β::Float64
    space::S
    function JacobiQWeight{S,DD}(α::Float64,β::Float64,space::S) where {S,DD}
        if isa(space,JacobiQWeight)
            JacobiQWeight(α+space.α,β+space.β,space.space)
        else
            new{S,DD}(α,β,space)
        end
    end
end

JacobiQWeight(a::Number,b::Number,d::RealUnivariateSpace) =
    JacobiQWeight{typeof(d),typeof(domain(d))}(Float64(a),Float64(b),d)
JacobiQWeight(a::Number,b::Number,d::IntervalOrSegmentDomain) =
    JacobiQWeight(Float64(a),Float64(b),Space(d))
JacobiQWeight(a::Number,b::Number,d::Vector) =
    JacobiQWeight(Float64(a),Float64(b),Space(d))
JacobiQWeight(a::Number,b::Number) = JacobiQWeight(a,b,Chebyshev())

JacobiQWeight(a::Number,b::Number,s::PiecewiseSpace) = PiecewiseSpace(JacobiQWeight(a,b,vec(s)))


spacescompatible(A::JacobiQWeight,B::JacobiQWeight) =
    A.α==B.α && A.β == B.β && spacescompatible(A.space,B.space)

transformtimes(f::Fun{JW1},g::Fun{JW2}) where {JW1<:JacobiQWeight,JW2<:JacobiQWeight}=
            Fun(JacobiQWeight(f.space.α+g.space.α,f.space.β+g.space.β,f.space.space),
                coefficients(transformtimes(Fun(f.space.space,f.coefficients),
                                            Fun(g.space.space,g.coefficients))))
transformtimes(f::Fun{JW},g::Fun) where {JW<:JacobiQWeight} = Fun(f.space,coefficients(transformtimes(Fun(f.space.space,f.coefficients),g)))
transformtimes(f::Fun,g::Fun{JW}) where {JW<:JacobiQWeight} = Fun(g.space,coefficients(transformtimes(Fun(g.space.space,g.coefficients),f)))

##  α and β are opposite the convention for JacobiQ polynomials
# Here, α is the left algebraic singularity and β is the right algebraic singularity.


jacobiQweight(α,β,x) = (x+1)^α*(x-1)^β
jacobiQweight(α,β,d::Domain) = Fun(JacobiQWeight(α,β,ConstantSpace(d)),[1.])
jacobiQweight(α,β) = jacobiQweight(α,β,ChebyshevInterval())

weight(sp::JacobiQWeight,x) = jacobiQweight(sp.α,sp.β,tocanonical(sp.space.domain,x))
dimension(sp::JacobiQWeight) = dimension(sp.space)


Base.first(f::Fun{JW}) where {JW<:JacobiQWeight} =
    space(f).α > 0 ? zero(cfstype(f)) : f(first(domain(f)))
Base.last(f::Fun{JW}) where {JW<:JacobiQWeight} =
    space(f).β > 0 ? zero(cfstype(f)) : f(last(domain(f)))

setdomain(sp::JacobiQWeight,d::Domain) = JacobiQWeight(sp.α,sp.β,setdomain(sp.space,d))
