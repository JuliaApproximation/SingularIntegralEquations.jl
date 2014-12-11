

# this macro creates the operators in the ApproxFun namespace, for some reason
ApproxFun.@calculus_operator(Hilbert,AbstractHilbert,HilbertWrapper)

import ApproxFun: Hilbert, AbstractHilbert, HilbertWrapper
export Hilbert

Hilbert(d::IntervalDomain,n::Integer)=Hilbert(JacobiWeightSpace(-.5,-.5,ChebyshevSpace(d)),n)

## JacobiWeightSpace

function Hilbert(S::JacobiWeightSpace{ChebyshevSpace},k::Integer)
    if S.α==S.β==-0.5
        Hilbert{JacobiWeightSpace{ChebyshevSpace},Float64}(S,k)
    elseif S.α==S.β==0.5
        @assert k==1
        HilbertWrapper(
            Hilbert(JacobiWeightSpace(0.5,0.5,UltrasphericalSpace{1}(domain(S))),k)*Conversion(S,JacobiWeightSpace(0.5,0.5,UltrasphericalSpace{1}(domain(S)))),
            k)
    else
        error("Hilbert not implemented")
    end
end

function rangespace(H::Hilbert{JacobiWeightSpace{ChebyshevSpace}})
    @assert domainspace(H).α==domainspace(H).β==-0.5
    UltrasphericalSpace{H.order}(domain(H))
end
function rangespace(H::Hilbert{JacobiWeightSpace{UltrasphericalSpace{1}}})
    @assert domainspace(H).α==domainspace(H).β==0.5
    @assert H.order==1
    ChebyshevSpace(domain(H))
end
bandinds(H::Hilbert{JacobiWeightSpace{ChebyshevSpace}})=0,H.order
bandinds(H::Hilbert{JacobiWeightSpace{UltrasphericalSpace{1}}})=-1,0

function addentries!(H::Hilbert{JacobiWeightSpace{ChebyshevSpace}},A::ShiftArray,kr::Range1)
    m=H.order
    d=domain(H)
    sp=domainspace(H)

    @assert isa(d,Interval)
    @assert sp.α==sp.β==-0.5    

    if m == 0
        C=(d.b-d.a)/2.
        for k=kr
            k == 1? A[k,0] += C*log(.5abs(C)) : A[k,0] += -C/(k-1)
        end
    else
        C=(4./(d.b-d.a))^(m-1)
        for k=kr
            A[k,m] += C
        end
    end
    
    A
end

function addentries!(H::Hilbert{JacobiWeightSpace{UltrasphericalSpace{1}}},A::ShiftArray,kr::Range1)
    m=H.order
    d=domain(H)

    @assert isa(d,Interval)
    @assert domainspace(H).α==domainspace(H).β==0.5    
    @assert m==1 
    for k=max(kr[1],2):kr[end]
        A[k,-1] -= 1.
    end
    
    A
end


## CurveSpace

function Hilbert(S::JacobiWeightSpace{CurveSpace{ChebyshevSpace}},k::Integer)
    @assert k==1
    #TODO: choose dimensions
    m,n=40,40
    c=domain(S)
    Sproj=JacobiWeightSpace(S.α,S.β)
    
    rts=[filter(y->!in(y,Interval()),complexroots(c.curve-c.curve[x])) for x in points(Interval(),n)]
    Hc=Hilbert(Sproj)
    
     M=2im*hcat(Vector{Complex{Float64}}[transform(rangespace(Hc),Complex{Float64}[sum(cauchy(Fun([zeros(k-1),1.0],Sproj),rt)) 
        for rt in rts]) for k=1:m]...)   
    
    rs=MappedSpace(c,rangespace(Hc))
    
    SpaceOperator(Hc,S,rs)+SpaceOperator(CompactOperator(M),S,rs) 
end