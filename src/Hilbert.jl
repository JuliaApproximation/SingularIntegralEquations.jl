export Hilbert,SingularIntegral

#############
# Hilbert implements the Hilbert operator as a contour integral:
#
#       OH f(z) := 1/π\int_Γ f(t)/(t-z) dt,  z ∈ Γ
#
# SingularIntegral implements the Hilbert operator as a line integral:
#
#       SI f(z) := 1/π\int_Γ f(t)/(t-z) ds(t),  z ∈ Γ
#
#############

ApproxFun.@calculus_operator(Hilbert,AbstractHilbert,HilbertWrapper)
ApproxFun.@calculus_operator(SingularIntegral,AbstractSingularIntegral,SingularIntegralWrapper)

for (Op,OpWrap,OffOp) in ((:Hilbert,:HilbertWrapper,:OffHilbert),(:SingularIntegral,:SingularIntegralWrapper,:OffSingularIntegral))
    @eval begin
        ## Convenience routines
        $Op(d::IntervalDomain,n::Int)=$Op(JacobiWeight(-.5,-.5,Chebyshev(d)),n)
        $Op(d::IntervalDomain)=$Op(JacobiWeight(-.5,-.5,Chebyshev(d)))
        $Op(d::PeriodicDomain,n::Int)=$Op(Laurent(d),n)
        $Op(d::PeriodicDomain)=$Op(Laurent(d))
        $Op(d::Domain)=$Op(Space(d))

        ## Modifiers for SumSpace, ArraySpace, ReImSpace, and PiecewiseSpace

        #TODO: do in @calculus_operator?
        $Op(S::SumSpace,n::Int)=$OpWrap(sumblkdiagm([$Op(S.spaces[1],n),$Op(S.spaces[2],n)]),n)
        $Op(AS::ArraySpace,n::Int)=$OpWrap(DiagonalArrayOperator($Op(AS.space,n),size(AS)),n)
        $Op(AS::ReImSpace,n::Int)=$OpWrap(ReImOperator($Op(AS.space,n)),n)
        function $Op(S::PiecewiseSpace,n::Int)
            sp=vec(S)
            #This isn't correct, but how to anticipate the unifying type without creating every block to begin with?
            C=BandedOperator{eltype($OffOp(sp[1],rangespace($Op(sp[2],n)),n))}[k==j?$Op(sp[k],n):$OffOp(sp[k],rangespace($Op(sp[j],n)),n) for j=1:length(sp),k=1:length(sp)]
            $OpWrap(interlace(C),n)
        end

        bandinds{s}(::$Op{Hardy{s}})=0,0
        domainspace{s}(H::$Op{Hardy{s}})=H.space
        rangespace{s}(H::$Op{Hardy{s}})=H.space

        bandinds{F<:Fourier}(H::$Op{F})=-H.order,H.order
        domainspace{F<:Fourier}(H::$Op{F})=H.space
        rangespace{F<:Fourier}(H::$Op{F})=H.space

        function rangespace(H::$Op{JacobiWeight{Chebyshev}})
            @assert domainspace(H).α==domainspace(H).β==-0.5
            Ultraspherical{H.order}(domain(H))
        end
        function rangespace(H::$Op{JacobiWeight{Ultraspherical{1}}})
            @assert domainspace(H).α==domainspace(H).β==0.5
            Ultraspherical{max(H.order-1,0)}(domain(H))
        end
        bandinds{λ}(H::$Op{JacobiWeight{Ultraspherical{λ}}})=-λ,H.order-λ

        function $Op(S::JacobiWeight{Chebyshev},n::Int)
            if S.α==S.β==-0.5
                $Op{JacobiWeight{Chebyshev},Float64}(S,n)
            elseif S.α==S.β==0.5
                d=domain(S)
                if n==1
                    J=JacobiWeight(0.5,0.5,Ultraspherical{1}(d))
                    $OpWrap($Op(J,n)*Conversion(S,J),n)
                else
                    J=JacobiWeight(-0.5,-0.5,Chebyshev(d))
                    $OpWrap($Op(J,n)*Conversion(S,J),n)
                end
            else
                error("$Op not implemented for parameters $(S.α),$(S.β)")
            end
        end
    end
end

# Override sumspace
Hilbert(F::Fourier,n::Int)=Hilbert{typeof(F),Complex{Float64}}(F,n)
SingularIntegral(F::Fourier,n::Int)=SingularIntegral{typeof(F),Float64}(F,n)

### Operator Entries

## Circle

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

for (Op,Len) in ((:Hilbert,:complexlength),(:SingularIntegral,:length))
    @eval begin
        function addentries!(H::$Op{JacobiWeight{Chebyshev}},A,kr::Range)
            m=H.order
            d=domain(H)
            sp=domainspace(H)

            @assert isa(d,Interval)
            @assert sp.α==sp.β==-0.5

            if m == 0
                C=$Len(d)/2.
                for k=kr
                    A[k,k] += k==1?C*log(C/2):-C/(k-1)
                end
            else
                C=(4./$Len(d))^(m-1)
                for k=kr
                    A[k,k+m] += C
                end
            end

            A
        end

        function addentries!(H::$Op{JacobiWeight{Ultraspherical{1}}},A,kr::UnitRange)
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
                C=(4./$Len(d))^(m-1)
                for k=kr
                    A[k,k+m-2] -= .5C*k/(m-1)
                end
            end

            A
        end
    end
end

## CurveSpace

function Hilbert(S::JacobiWeight{OpenCurveSpace{Chebyshev}},k::Int)
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
