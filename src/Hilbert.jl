export Hilbert

abstract AbstractHilbert{S,T} <: CalculusOperator{S,T}
ApproxFun.@calculus_operator(Hilbert,AbstractHilbert,HilbertWrapper)

#TODO: do in @calculus_operator?
Hilbert(S::SumSpace,k::Integer)=HilbertWrapper(sumblkdiagm([Hilbert(S.spaces[1],k),Hilbert(S.spaces[2],k)]),k)


Hilbert(d::IntervalDomain,n::Integer)=Hilbert(JacobiWeightSpace(-.5,-.5,ChebyshevSpace(d)),n)
Hilbert(d::IntervalDomain)=Hilbert(JacobiWeightSpace(-.5,-.5,ChebyshevSpace(d)))
Hilbert(d::PeriodicDomain,n::Integer)=Hilbert(LaurentSpace(d),n)
Hilbert(d::PeriodicDomain)=Hilbert(LaurentSpace(d))

Hilbert(d::Domain)=Hilbert(Space(d))


## Circle

bandinds{s}(::Hilbert{HardySpace{s}})=0,0
domainspace{s}(H::Hilbert{HardySpace{s}})=H.space
rangespace{s}(H::Hilbert{HardySpace{s}})=H.space

function addentries!{s}(H::Hilbert{HardySpace{s}},A::ShiftArray,kr::Range1)
    @assert isa(domain(H),Circle) && H.order == 1
    for k=kr
        A[k,0]+=s?1.im:-1.im
    end
    
    A
end


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

#function getindex{S<:UltrasphericalSpace}#(H::AbstractHilbert{JacobiWeightSpace{S}},w::Fun{JacobiWeightSpace{ChebyshevSpace}})
#    @assert domainspace(H)==space(w)
#
#   H*Multiplication(w,space(w).space)
#end

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

function Hilbert(S::JacobiWeightSpace{OpenCurveSpace{ChebyshevSpace}},k::Integer)
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