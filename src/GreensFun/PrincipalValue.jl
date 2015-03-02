# Principal Value Integral (Could be called PrincipalValueIntegral)

export PrincipalValue

immutable PrincipalValue{D<:FunctionSpace,T<:Number} <: Functional{T}
    domainspace::D
end

PrincipalValue()=PrincipalValue(UnsetSpace())
PrincipalValue(dsp::FunctionSpace) = PrincipalValue{typeof(dsp),eltype(dsp)}(dsp)
promotedomainspace(::PrincipalValue,sp::FunctionSpace)=PrincipalValue(sp)

Base.convert{T}(::Type{Functional{T}},⨍::PrincipalValue)=PrincipalValue{typeof(⨍.domainspace),T}(⨍.domainspace)

domain(⨍::PrincipalValue)=domain(⨍.domainspace)
domainspace(⨍::PrincipalValue)=⨍.domainspace

getindex(::PrincipalValue{UnsetSpace},kr::Range)=error("Spaces cannot be inferred for operator")

PrincipalValue(d::IntervalDomain)=PrincipalValue(JacobiWeight(-.5,-.5,Chebyshev(d)))

function PrincipalValue(α::Number,β::Number,d::IntervalDomain)
    @assert α == β == -.5 || α == β == .5
    PrincipalValue(JacobiWeight(α,β,Ultraspherical{int(α+.5)}(d)))
end
PrincipalValue(α::Number,β::Number) = PrincipalValue(α,β,Interval())


Base.getindex{S,V,O,T}(⨍::PrincipalValue{V,T},f::ProductFun{S,V,CauchyWeight{O},T}) = Hilbert(⨍.domainspace,O)[f]
Base.getindex{S,V,SS,T}(⨍::PrincipalValue{V,T},f::ProductFun{S,V,SS,T}) = DefiniteIntegral(⨍.domainspace)[f]
