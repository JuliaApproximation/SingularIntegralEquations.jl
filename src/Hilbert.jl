export Hilbert,SingularIntegral

#############
# Hilbert implements the Hilbert operator as a contour integral:
#
#        H f(z) := 1/π\int_Γ f(t)/(t-z) dt,  z ∈ Γ
#
# SingularIntegral implements the Hilbert operator as a line integral:
#
#       SI f(z) := 1/π\int_Γ f(t)/(t-z) ds(t),  z ∉ Γ
#
#############

ApproxFun.@calculus_operator(Hilbert)
ApproxFun.@calculus_operator(SingularIntegral)

for (Op,OpWrap,OffOp) in ((:Hilbert,:HilbertWrapper,:OffHilbert),
                          (:SingularIntegral,:SingularIntegralWrapper,:OffSingularIntegral))
    @eval begin
        ## Convenience routines
        $Op(d::IntervalDomain,n::Int)=$Op(JacobiWeight(-.5,-.5,Chebyshev(d)),n)
        $Op(d::IntervalDomain)=$Op(JacobiWeight(-.5,-.5,Chebyshev(d)))
        $Op(d::PeriodicDomain,n::Int)=$Op(Laurent(d),n)
        $Op(d::PeriodicDomain)=$Op(Laurent(d))

        ## Modifiers for SumSpace, ArraySpace, ReImSpace, and PiecewiseSpace

        #TODO: do in @calculus_operator?
        $Op(S::SumSpace,n)=$OpWrap(sumblkdiagm([$Op(S.spaces[1],n),$Op(S.spaces[2],n)]),n)
        $Op(AS::ArraySpace,n::Int)=$OpWrap(DiagonalArrayOperator($Op(AS.space,n),size(AS)),n)
        $Op(AS::ReImSpace,n::Int)=$OpWrap(ReImOperator($Op(AS.space,n)),n)
        function $Op(S::PiecewiseSpace,n::Int)
            sp=vec(S)
            m=length(sp)
            C=[k==j?$Op(sp[k],n):$OffOp(sp[k],rangespace($Op(sp[j],n)),n) for j=1:m,k=1:m]
            C=BandedOperator{mapreduce(i->eltype(C[i]),promote_type,1:m^2)}[C[j,k] for j=1:m,k=1:m]
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
        bandinds(H::$Op{JacobiWeight{Chebyshev}})=0,H.order
        bandinds(H::$Op{JacobiWeight{Ultraspherical{1}}})=H.order > 0 ? (-1,H.order-1) : (-2,0)
    end
end

# Length catch

Hilbert(sp::FunctionSpace{ComplexBasis},n)=Hilbert{typeof(sp),typeof(n),Complex{real(eltype(domain(sp)))}}(sp,n)
Hilbert(sp::FunctionSpace,n)=Hilbert{typeof(sp),typeof(n),typeof(complexlength(domain(sp)))}(sp,n)

SingularIntegral(sp::FunctionSpace{ComplexBasis},n)=SingularIntegral{typeof(sp),typeof(n),Complex{real(eltype(domain(sp)))}}(sp,n)
SingularIntegral(sp::FunctionSpace,n)=SingularIntegral{typeof(sp),typeof(n),typeof(length(domain(sp)))}(sp,n)

# Override sumspace

Hilbert(F::Fourier,n)=Hilbert{typeof(F),typeof(n),Complex{Float64}}(F,n)
SingularIntegral(F::Fourier,n)=SingularIntegral{typeof(F),typeof(n),Float64}(F,n)

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

function addentries!(H::SingularIntegral{Hardy{true}},A,kr::Range)
##TODO: Add scale for different radii.
    m=H.order
    d=domain(H)
    sp=domainspace(H)
    @assert isa(d,Circle)
    r = domain(H).radius
    if m == 0
        for k=kr
            A[k,k] += k==1?2r*log(r):-r./(k-1)
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

function addentries!(H::SingularIntegral{Hardy{false}},A,kr::Range)
##TODO: Add scale for different radii.
    m=H.order
    d=domain(H)
    sp=domainspace(H)
    @assert isa(d,Circle)
    r = domain(H).radius
    if m== 1
        for k=kr
            A[k,k]-= im
        end
    else
        for k=kr
            A[k,k]-=1.im*(1.im*k/r)^(m-1)
        end
    end
    A
end

function addentries!{F<:Fourier}(H::SingularIntegral{F},A,kr::Range)
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
                A[1,1]+=0
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

for (Op,OpWrap,Len) in ((:Hilbert,:HilbertWrapper,:complexlength),
                        (:SingularIntegral,:SingularIntegralWrapper,:length))
    @eval begin

        function $Op(S::JacobiWeight{Chebyshev},n::Int)
            if S.α==S.β==-0.5
                $Op{JacobiWeight{Chebyshev},typeof(n),typeof($Len(domain(S)))}(S,n)
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

        function addentries!(H::$Op{JacobiWeight{Chebyshev}},A,kr::Range)
            m=H.order
            d=domain(H)
            sp=domainspace(H)

            @assert isa(d,Interval)
            @assert sp.α==sp.β==-0.5

            C=(4./$Len(d))^(m-1)
            if m == 0
                for k=kr
                    A[k,k] -= k==1?-2C*log(C):2C/(k-1)
                end
            else
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

            C=(4./$Len(d))^(m-1)
            if m == 0
                for k=kr
                    A[k,k] -= k==1? -C*log(C) : C/(k-1)
                end
                for k=max(kr[1],3):kr[end]
                    A[k,k-2] += C/(k-1)
                end
            elseif m == 1
                for k=max(kr[1],2):kr[end]
                    A[k,k-1] -= 1.
                end
            else
                for k=kr
                    A[k,k+m-2] -= .5C*k/(m-1)
                end
            end

            A
        end
    end
end

