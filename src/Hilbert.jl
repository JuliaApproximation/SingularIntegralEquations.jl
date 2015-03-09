export Hilbert

################################################
# Hilbert implements the Hilbert operator
# Note that the Hilbert operator can be defined using
#
#    H = im*C^+  +  im*C^-
#
# where C^± are the left/right limits of the Cauchy transform
###############################################

ApproxFun.@calculus_operator(Hilbert,AbstractHilbert,HilbertWrapper)

## Convenience routines

Hilbert(d::IntervalDomain,n::Integer)=Hilbert(JacobiWeight(-.5,-.5,Chebyshev(d)),n)
Hilbert(d::IntervalDomain)=Hilbert(JacobiWeight(-.5,-.5,Chebyshev(d)))
Hilbert(d::PeriodicDomain,n::Integer)=Hilbert(Laurent(d),n)
Hilbert(d::PeriodicDomain)=Hilbert(Laurent(d))
Hilbert(d::Domain)=Hilbert(Space(d))


## Modifiers including SumSpace and ArraySpace

#TODO: do in @calculus_operator?
Hilbert(S::SumSpace,k::Integer)=HilbertWrapper(sumblkdiagm([Hilbert(S.spaces[1],k),Hilbert(S.spaces[2],k)]),k)
Hilbert(AS::ArraySpace,k::Integer)=HilbertWrapper(DiagonalArrayOperator(Hilbert(AS.space,k),size(AS)),k)
Hilbert(AS::ReImSpace,k::Integer)=HilbertWrapper(ReImOperator(Hilbert(AS.space,k)),k)

## PiecewiseSpace


function Hilbert(S::PiecewiseSpace,n::Integer)
    sp=vec(S)
  #This isn't correct, but how to anticipate the unifying type without creating every block to begin with?
    C=BandedOperator{eltype(OffHilbert(sp[1],rangespace(Hilbert(sp[2],n)),n))}[k==j?Hilbert(sp[k],n):OffHilbert(sp[k],rangespace(Hilbert(sp[j],n)),n) for j=1:length(sp),k=1:length(sp)]
    HilbertWrapper(interlace(C),n)
end

## Circle

bandinds{s}(::Hilbert{Hardy{s}})=0,0
domainspace{s}(H::Hilbert{Hardy{s}})=H.space
rangespace{s}(H::Hilbert{Hardy{s}})=H.space

function addentries!(H::Hilbert{Hardy{true}},A,kr::Range)
##TODO: Add scale for different radii.
    m=H.order
    d=domain(H)
    sp=domainspace(H)
    @assert isa(d,Circle)
    if m == 0
        for k=kr
            A[k,k] += k==1?-2log(2):1./(k-1)
        end
    elseif m == 1
        for k=kr
            A[k,k] += im
        end
    else
        for k=kr
            A[k,k] += k==1?0.0:1.im*(1.im*(k-1))^(m-1)
        end
    end
    A
end

function addentries!(H::Hilbert{Hardy{false}},A,kr::Range)
##TODO: Add scale for different radii.
    m=H.order
    d=domain(H)
    sp=domainspace(H)
    @assert isa(d,Circle)
    if m== 1
        for k=kr
            A[k,k]-= im
        end
    else
        for k=kr
            A[k,k]-=1.im*(1.im*k)^(m-1)
        end
    end
    A
end

# Override sumspace
Hilbert(F::Fourier,k::Integer)=Hilbert{typeof(F),k==0?Float64:Complex{Float64}}(F,k)

bandinds{F<:Fourier}(H::Hilbert{F})=-H.order,H.order
domainspace{F<:Fourier}(H::Hilbert{F})=H.space
rangespace{F<:Fourier}(H::Hilbert{F})=H.space

function addentries!{F<:Fourier}(H::Hilbert{F},A,kr::Range)
    @assert isa(domain(H),Circle)

    r = domain(H).radius
    if H.order == 0
        for k=kr
            if k==1
                A[1,1]+=2r*log(r)
            else
                j=div(k,2)
                A[k,k]+=-r/j
            end
        end
    elseif H.order == 1
        for k=kr
            if k==1
                A[1,1]+=1.0im
            elseif iseven(k)
                A[k,k+1]-=1
            else   #isodd(k)
                A[k,k-1]+=1
            end
        end
    else
            error("Hilbert order $(H.order) not implemented for Fourier")
    end

    A
end




## JacobiWeight

function Hilbert(S::JacobiWeight{Chebyshev},k::Integer)
    if S.α==S.β==-0.5
        Hilbert{JacobiWeight{Chebyshev},Float64}(S,k)
    elseif S.α==S.β==0.5
        d=domain(S)
        if k==1
            J=JacobiWeight(0.5,0.5,Ultraspherical{1}(d))
            HilbertWrapper(Hilbert(J,k)*Conversion(S,J),k)
        else
            J=JacobiWeight(-0.5,-0.5,Chebyshev(d))
            HilbertWrapper(Hilbert(J,k)*Conversion(S,J),k)
        end
    else
        error("Hilbert not implemented for parameters $(S.α),$(S.β)")
    end
end

function rangespace(H::Hilbert{JacobiWeight{Chebyshev}})
    @assert domainspace(H).α==domainspace(H).β==-0.5
    Ultraspherical{H.order}(domain(H))
end
function rangespace(H::Hilbert{JacobiWeight{Ultraspherical{1}}})
    @assert domainspace(H).α==domainspace(H).β==0.5
    Ultraspherical{max(H.order-1,0)}(domain(H))
end
bandinds{λ}(H::Hilbert{JacobiWeight{Ultraspherical{λ}}})=-λ,H.order-λ


function addentries!(H::Hilbert{JacobiWeight{Chebyshev}},A,kr::Range)
    m=H.order
    d=domain(H)
    sp=domainspace(H)

    @assert isa(d,Interval)
    @assert sp.α==sp.β==-0.5

    if m == 0
        C=length(d)/2.
        for k=kr
            A[k,k] += k==1?C*log(C/2):-C/(k-1)
        end
    else
        C=(4./(d.b-d.a))^(m-1)
        for k=kr
            A[k,k+m] += C
        end
    end

    A
end

function addentries!(H::Hilbert{JacobiWeight{Ultraspherical{1}}},A,kr::UnitRange)
    m=H.order
    d=domain(H)
    sp=domainspace(H)

    @assert isa(d,Interval)
    @assert sp.α==sp.β==0.5

    if m == 1
        for k=max(kr[1],2):kr[end]
            A[k,k-1] -= 1.
        end
    else
        C=(4./(d.b-d.a))^(m-1)
        for k=kr
            A[k,k+m-2] -= .5C*k/(m-1)
        end
    end

    A
end


## CurveSpace

function Hilbert(S::JacobiWeight{OpenCurveSpace{Chebyshev}},k::Integer)
    @assert k==1
    #TODO: choose dimensions
    m,n=40,40
    c=domain(S)
    Sproj=JacobiWeight(S.α,S.β)

    rts=[filter(y->!in(y,Interval()),complexroots(c.curve-c.curve[x])) for x in points(Interval(),n)]
    Hc=Hilbert(Sproj)

     M=2im*hcat(Vector{Complex{Float64}}[transform(rangespace(Hc),Complex{Float64}[sum(cauchy(Fun([zeros(k-1),1.0],Sproj),rt))
        for rt in rts]) for k=1:m]...)

    rs=MappedSpace(c,rangespace(Hc))

    SpaceOperator(Hc,S,rs)+SpaceOperator(CompactOperator(M),S,rs)
end
