# Principal Value Integral

export PrincipalValue

immutable PrincipalValue{D<:FunctionSpace,T<:Number} <: Functional{T}
    domainspace::D
end

PrincipalValue()=PrincipalValue(UnsetSpace())
PrincipalValue(dsp::FunctionSpace) = PrincipalValue{typeof(dsp),eltype(dsp)}(dsp)
#promotedomainspace(::PrincipalValue,sp::FunctionSpace)=PrincipalValue(sp)

domain(⨍::PrincipalValue)=domain(⨍.domainspace)
domainspace(⨍::PrincipalValue)=⨍.domainspace

#Base.getindex(::PrincipalValue{UnsetSpace},kr::Range)=error("Spaces cannot be inferred for operator")

PrincipalValue(d::IntervalDomain)=PrincipalValue(JacobiWeight(-.5,-.5,Chebyshev(d)))

function PrincipalValue(α::Number,β::Number,d::IntervalDomain)
    @assert α == β == -.5 || α == β == .5
    PrincipalValue(JacobiWeight(α,β,Ultraspherical{int(α+.5)}(d)))
end
PrincipalValue(α::Number,β::Number) = PrincipalValue(α,β,Interval())


Base.getindex{S,V,O,T1,T2}(⨍::PrincipalValue{V,T1},f::ProductFun{S,V,CauchyWeight{O},T2}) = Hilbert(⨍.domainspace,O)[f]
Base.getindex{S,V,SS,T1,T2}(⨍::PrincipalValue{V,T1},f::ProductFun{S,V,SS,T2}) = DefiniteIntegral(⨍.domainspace)[f]
