export PseudoHilbert,Hilbert,SingularIntegral

#############
# Hilbert implements the Hilbert operator as a contour integral:
#
#        H f(z) := 1/π\int_Γ f(t)/(t-z) dt,  z ∈ Γ
#
# SingularIntegral implements the Hilbert operator as a line integral:
#
#       SI f(z) := 1/π\int_Γ f(t)/(t-z) ds(t),  z ∉ Γ
#
# PseudoHilbert corresponds to pseudocauchy, which may not be normalized at
# infinity.
#############


ApproxFun.@calculus_operator(PseudoHilbert)
ApproxFun.@calculus_operator(Hilbert)
ApproxFun.@calculus_operator(SingularIntegral)

for Op in (:PseudoHilbert,:Hilbert,:SingularIntegral)
    ConcOp=parse("Concrete"*string(Op))
    OpWrap=parse(string(Op)*"Wrapper")
    OffOp=parse("Off"*string(Op))
    @eval begin
        ## Convenience routines
        $Op(d::IntervalDomain,n::Int)=$Op(JacobiWeight(-.5,-.5,Chebyshev(d)),n)
        $Op(d::IntervalDomain)=$Op(JacobiWeight(-.5,-.5,Chebyshev(d)))
        $Op(d::PeriodicDomain,n::Int)=$Op(Laurent(d),n)
        $Op(d::PeriodicDomain)=$Op(Laurent(d))

        ## Modifiers for SumSpace, ArraySpace, ReImSpace, and PiecewiseSpace

        $Op(AS::ArraySpace,n::Int)=$OpWrap(DiagonalArrayOperator($Op(AS.space,n),size(AS)),n)
        $Op(AS::ReImSpace,n::Int)=$OpWrap(ReImOperator($Op(AS.space,n)),n)
        function $Op(S::PiecewiseSpace,n::Int)
            sp=vec(S)
            m=length(sp)
            diag=Any[$Op(sp[k],n) for k=1:m]
            C=Any[k==j?diag[k]:$OffOp(sp[k],rangespace(diag[j]),n) for j=1:m,k=1:m]
            D=BandedOperator{mapreduce(i->eltype(C[i]),promote_type,1:m^2)}[C[j,k] for j=1:m,k=1:m]
            $OpWrap(interlace(D),n)
        end

        bandinds{s,DD}(::$ConcOp{Hardy{s,DD}})=0,0
        domainspace{s,DD}(H::$ConcOp{Hardy{s,DD}})=H.space
        rangespace{s,DD}(H::$ConcOp{Hardy{s,DD}})=H.space

        bandinds{DD}(H::$ConcOp{Fourier{DD}})=-H.order,H.order
        domainspace{DD}(H::$ConcOp{Fourier{DD}})=H.space
        rangespace{DD}(H::$ConcOp{Fourier{DD}})=H.space

        function rangespace{DD}(H::$ConcOp{JacobiWeight{Chebyshev{DD},DD}})
            @assert domainspace(H).α==domainspace(H).β==-0.5
            Ultraspherical{H.order}(domain(H))
        end
        function rangespace{DD}(H::$ConcOp{JacobiWeight{Ultraspherical{1,DD},DD}})
            @assert domainspace(H).α==domainspace(H).β==0.5
            Ultraspherical{max(H.order-1,0)}(domain(H))
        end
        bandinds{λ,DD}(H::$ConcOp{JacobiWeight{Ultraspherical{λ,DD},DD}})=-λ,H.order-λ
        bandinds{DD}(H::$ConcOp{JacobiWeight{Chebyshev{DD},DD}})=0,H.order
        bandinds{DD}(H::$ConcOp{JacobiWeight{Ultraspherical{1,DD},DD}})=H.order > 0 ? (-1,H.order-1) : (-2,0)

        choosedomainspace(H::$Op{UnsetSpace},sp::Ultraspherical)=ChebyshevWeight(ChebyshevDirichlet{1,1}(domain(sp)))
        choosedomainspace(H::$Op{UnsetSpace},sp::PiecewiseSpace)=PiecewiseSpace(map(s->choosedomainspace(H,s),sp.spaces))



        # BlockOperator [1 Hilbert()] which allows for bounded solutions
        #TODO: Array values?
        choosedomainspace{T,V}(P::BlockOperator{$ConcOp{UnsetSpace,T,V}},
                               sp::Ultraspherical)=TupleSpace(ConstantSpace(),
                                                            JacobiWeight(0.5,0.5,
                                                                         Ultraspherical{1}(domain(sp))))

        choosedomainspace{T,V,W}(P::BlockOperator{ReOperator{$ConcOp{UnsetSpace,T,V},W}},
                               sp::Ultraspherical)=TupleSpace(ConstantSpace(),
                                                            JacobiWeight(0.5,0.5,
                                                                         Ultraspherical{1}(domain(sp))))
    end
end

for TYP in (:SumSpace,:PiecewiseSpace,:TupleSpace),(Op,OpWrap) in ((:PseudoHilbert,:PseudoHilbertWrapper),
                          (:Hilbert,:HilbertWrapper),
                          (:SingularIntegral,:SingularIntegralWrapper))
    @eval $Op(S::$TYP,k)=$OpWrap(DiagonalInterlaceOperator(map(s->$Op(s,k),S.spaces),$TYP),k)
end

# Length catch

Hilbert(sp::Space{ComplexBasis},n)=ConcreteHilbert{typeof(sp),typeof(n),Complex{real(eltype(domain(sp)))}}(sp,n)
Hilbert(sp::Space,n)=ConcreteHilbert{typeof(sp),typeof(n),typeof(complexlength(domain(sp)))}(sp,n)

SingularIntegral(sp::Space{ComplexBasis},n)=ConcreteSingularIntegral(sp,n)
SingularIntegral(sp::Space,n)=ConcreteSingularIntegral(sp,n)

# Override sumspace

for TYP in (:Hilbert,:SingularIntegral)
    ConcOp=parse("Concrete"*string(TYP))
    @eval function $TYP{DD}(F::Fourier{DD},n)
        if !domain(F).orientation
            R=reverseorientation(F)
            Conversion(R,F)*(-$TYP(R,n))*Conversion(F,R)
        else
            $ConcOp(F,n)
        end
    end
end

### Operator Entries

## Circle

function addentries!{DD<:Circle}(H::ConcreteHilbert{Hardy{true,DD}},A,kr::Range,::Colon)
##TODO: Add scale for different radii.
    m=H.order
    d=domain(H)
    sp=domainspace(H)

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

function addentries!{DD<:Circle}(H::ConcreteHilbert{Hardy{false,DD}},A,kr::Range,::Colon)
##TODO: Add scale for different radii.
    m=H.order
    d=domain(H)
    sp=domainspace(H)

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

function addentries!{DD<:Circle}(H::ConcreteHilbert{Fourier{DD}},A,kr::Range,::Colon)
    r = domain(H).radius
    if H.order == 0
        for k=kr
            A[k,k]+=k==1?2r*log(r):(-r/(k÷2))
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

function addentries!{DD<:Circle}(H::ConcreteSingularIntegral{Hardy{true,DD}},A,kr::Range,::Colon)
##TODO: Add scale for different radii.
    m=H.order
    d=domain(H)
    sp=domainspace(H)

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

function addentries!{DD<:Circle}(H::ConcreteSingularIntegral{Hardy{false,DD}},A,kr::Range,::Colon)
##TODO: Add scale for different radii.
    m=H.order
    d=domain(H)
    sp=domainspace(H)

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

function addentries!{DD<:Circle}(H::ConcreteSingularIntegral{Fourier{DD}},A,kr::Range,::Colon)
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

for (Op,Len) in ((:Hilbert,:complexlength),
                        (:SingularIntegral,:length))
    ConcOp=parse("Concrete"*string(Op))
    OpWrap=parse(string(Op)*"Wrapper")

    @eval begin
        function $Op{DD<:Interval}(S::JacobiWeight{Chebyshev{DD},DD},n::Int)
            if S.α==S.β==-0.5
                $ConcOp(S,n)
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
                error(string($Op)*" not implemented for parameters $(S.α),$(S.β)")
            end
        end

        function addentries!{DD<:Interval}(H::$ConcOp{JacobiWeight{Chebyshev{DD},DD}},A,kr::Range,::Colon)
            m=H.order
            d=domain(H)
            sp=domainspace(H)

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

        # we always have real for n==1
        $Op{DD<:Interval}(sp::JacobiWeight{Ultraspherical{1,DD},DD},n)=$ConcOp{typeof(sp),typeof(n),
                                                           n==1?real(eltype(domain(sp))):typeof($Len(domain(sp)))}(sp,n)
        function addentries!{DD<:Interval}(H::$ConcOp{JacobiWeight{Ultraspherical{1,DD},DD}},A,kr::UnitRange,::Colon)
            m=H.order
            d=domain(H)
            sp=domainspace(H)
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





## PseudoHilbert
# The default is Hilbert


addentries!(H::ConcretePseudoHilbert,A,kr::Range,::Colon)=addentries!(Hilbert(H.space,H.order),A,kr,:)
bandinds(H::ConcretePseudoHilbert)=bandinds(Hilbert(H.space,H.order))
