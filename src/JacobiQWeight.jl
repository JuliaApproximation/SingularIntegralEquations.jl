import ApproxFun: weight, dimension

export JacobiQWeight

"""
`JacobiQWeight`
weights a basis on `ℂ\𝕀` weighted by `(z+1)^α*(z-1)^β`.
Note the inconsistency of the parameters with `JacobiQ`.
When the domain is `[a,b]` the weight is inferred by mapping to `[-1,1]`
"""
immutable JacobiQWeight{S,DD} <: WeightSpace{S,RealBasis,DD,1}
    α::Float64
    β::Float64
    space::S
    function JacobiQWeight(α::Float64,β::Float64,space::S)
        if isa(space,JacobiQWeight)
            JacobiQWeight(α+space.α,β+space.β,space.space)
        else
            new(α,β,space)
        end
    end
end

JacobiQWeight(a::Number,b::Number,d::RealUnivariateSpace)=JacobiQWeight{typeof(d),typeof(domain(d))}(Float64(a),Float64(b),d)
JacobiQWeight(a::Number,b::Number,d::IntervalDomain)=JacobiQWeight(Float64(a),Float64(b),Space(d))
JacobiQWeight(a::Number,b::Number,d::Vector)=JacobiQWeight(Float64(a),Float64(b),Space(d))
JacobiQWeight(a::Number,b::Number)=JacobiQWeight(a,b,Chebyshev())

JacobiQWeight(a::Number,b::Number,s::PiecewiseSpace) = PiecewiseSpace(JacobiQWeight(a,b,vec(s)))


spacescompatible(A::JacobiQWeight,B::JacobiQWeight)=A.α==B.α && A.β == B.β && spacescompatible(A.space,B.space)
#spacescompatible{D<:IntervalDomain}(A::JacobiQWeight,B::RealUnivariateSpace{D})=spacescompatible(A,JacobiQWeight(0,0,B))
#spacescompatible{D<:IntervalDomain}(B::RealUnivariateSpace{D},A::JacobiQWeight)=spacescompatible(A,JacobiQWeight(0,0,B))

transformtimes{JW1<:JacobiQWeight,JW2<:JacobiQWeight}(f::Fun{JW1},g::Fun{JW2})=
            Fun(JacobiQWeight(f.space.α+g.space.α,f.space.β+g.space.β,f.space.space),
                coefficients(transformtimes(Fun(f.space.space,f.coefficients),
                                            Fun(g.space.space,g.coefficients))))
transformtimes{JW<:JacobiQWeight}(f::Fun{JW},g::Fun) = Fun(f.space,coefficients(transformtimes(Fun(f.space.space,f.coefficients),g)))
transformtimes{JW<:JacobiQWeight}(f::Fun,g::Fun{JW}) = Fun(g.space,coefficients(transformtimes(Fun(g.space.space,g.coefficients),f)))

##  α and β are opposite the convention for JacobiQ polynomials
# Here, α is the left algebraic singularity and β is the right algebraic singularity.


jacobiQweight(α,β,x)=(x+1).^α.*(x-1).^β
jacobiQweight(α,β,d::Domain)=Fun(JacobiQWeight(α,β,ConstantSpace(d)),[1.])
jacobiQweight(α,β)=jacobiQweight(α,β,Interval())

weight(sp::JacobiQWeight,x)=jacobiQweight(sp.α,sp.β,tocanonical(sp,x))
dimension(sp::JacobiQWeight)=dimension(sp.space)


Base.first{JW<:JacobiQWeight}(f::Fun{JW})=space(f).α>0?zero(eltype(f)):f(first(domain(f)))
Base.last{JW<:JacobiQWeight}(f::Fun{JW})=space(f).β>0?zero(eltype(f)):f(last(domain(f)))

setdomain(sp::JacobiQWeight,d::Domain)=JacobiQWeight(sp.α,sp.β,setdomain(sp.space,d))
