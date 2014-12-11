export Σ
# The Σ operator is intended to act as the orthogonal summation
# operator, expressing the operation of integrating a Fun with
# a bivariate kernel.
# TODO: This should be domain independent. Since a LowRankFun
# can be constructed with two distinct spaces and Σ only has
# one space, it may require rangespace & domainspace assertions.
# Use at own risk. 

immutable Σ{S<:FunctionSpace,T<:Number} <: BandedOperator{T}
    space::S        # the domain space
end

Σ{S<:PeriodicDomainSpace}(sp::S) = Σ{S,Complex{Float64}}(sp)
Σ{S<:FunctionSpace}(sp::S)=Σ{S,Float64}(sp)
Σ()=Σ(AnySpace())

Σ(d::PeriodicDomain)=Σ(LaurentSpace(d))
Σ(d::IntervalDomain)=Σ(ChebyshevSpace(d))

domain(S::Σ)=domain(S.space)
domainspace(S::Σ)=S.space
rangespace(S::Σ)=S.space

bandinds(S::Σ) = 0,0

## addentries

addentries!{T}(::Σ{AnySpace,T},A::ShiftArray,kr::Range)=error("Spaces cannot be inferred for operator")

function addentries!(S::Σ{JacobiWeightSpace{ChebyshevSpace}},A::ShiftArray,kr::Range1)
    d=domain(S)
    @assert isa(d,Interval)
    @assert domainspace(S).α==domainspace(S).β==-0.5   
    
    for k=kr
        k == 1? A[k,0] += 1.0 : A[k,0] += 0.0
    end
    
    A
end