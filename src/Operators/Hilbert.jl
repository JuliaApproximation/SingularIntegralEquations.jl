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


@calculus_operator(PseudoHilbert)
@calculus_operator(Hilbert)
@calculus_operator(SingularIntegral)

for Op in (:PseudoHilbert,:Hilbert,:SingularIntegral)
    ConcOp=Meta.parse("Concrete"*string(Op))
    OpWrap=Meta.parse(string(Op)*"Wrapper")
    OffOp=Meta.parse("Off"*string(Op))
    @eval begin
        ## Convenience routines
        $Op(d::IntervalOrSegmentDomain, n::Int) =
            $Op(JacobiWeight(-.5,-.5,Chebyshev(d)),n)
        $Op(d::IntervalOrSegmentDomain) =
            $Op(JacobiWeight(-.5,-.5,Chebyshev(d)))
        $Op(d::PeriodicDomain,n::Int) = $Op(Laurent(d),n)
        $Op(d::PeriodicDomain) = $Op(Laurent(d))

        ## Modifiers for SumSpace, ArraySpace, and PiecewiseSpace
        function $Op(S::PiecewiseSpace,n::Int)
            sp = components(S)
            m = length(sp)
            diag = Any[$Op(sp[k],n) for k=1:m]
            C = Any[k==j ? diag[k] : $OffOp(sp[k],rangespace(diag[j]),n) for j=1:m,k=1:m]
            D = Operator{mapreduce(i->eltype(C[i]),promote_type,1:m^2)}[C[j,k] for j=1:m,k=1:m]
            D = promotespaces(D)
            $OpWrap(InterlaceOperator(D,S,PiecewiseSpace(map(rangespace,D[:,1]))),n)
        end

        bandwidths(::$ConcOp{Hardy{s,DD,RR}}) where {s,DD,RR} = 0,0
        domainspace(H::$ConcOp{Hardy{s,DD,RR}}) where {s,DD,RR} = H.space
        rangespace(H::$ConcOp{Hardy{s,DD,RR}}) where {s,DD,RR} = H.space

        bandwidths(::$ConcOp{Laurent{DD,RR}}) where {DD,RR} = 0,0
        domainspace(H::$ConcOp{Laurent{DD,RR}}) where {DD,RR} = H.space
        rangespace(H::$ConcOp{Laurent{DD,RR}}) where {DD,RR} = H.space

        bandwidths(H::$ConcOp{Fourier{DD,RR}}) where {DD,RR} = H.order,H.order
        domainspace(H::$ConcOp{Fourier{DD,RR}}) where {DD,RR} = H.space
        rangespace(H::$ConcOp{Fourier{DD,RR}}) where {DD,RR} = H.space

        function rangespace(H::$ConcOp{<:JacobiWeight{<:Chebyshev}})
            @assert domainspace(H).α == domainspace(H).β == -0.5
            H.order==0 ? Chebyshev(domain(H)) : Ultraspherical(H.order,domain(H))
        end
        function rangespace(H::$ConcOp{<:JacobiWeight{<:Ultraspherical{Int}}})
            @assert order(domainspace(H).space) == 1
            @assert domainspace(H).α==domainspace(H).β==0.5
            (H.order==1||H.order==0) ? Chebyshev(domain(H)) : Ultraspherical(H.order-1,domain(H))
        end
        # bandwidths{λ,DD,RR}(H::$ConcOp{JacobiWeight{Ultraspherical{λ,DD,RR},DD,RR}})=-λ,H.order-λ
        bandwidths(H::$ConcOp{<:JacobiWeight{<:Chebyshev}}) = 0,H.order
        bandwidths(H::$ConcOp{<:JacobiWeight{<:Ultraspherical{Int}}}) =
            H.order > 0 ? (1,H.order-1) : (2,0)

        choosedomainspace(H::$Op{UnsetSpace}, sp::Ultraspherical) =
            ChebyshevWeight(ChebyshevDirichlet{1,1}(domain(sp)))
        choosedomainspace(H::$Op{UnsetSpace}, sp::Chebyshev) =
            ChebyshevWeight(ChebyshevDirichlet{1,1}(domain(sp)))
        choosedomainspace(H::$Op{UnsetSpace}, sp::PiecewiseSpace) =
            PiecewiseSpace(map(s->choosedomainspace(H,s),sp.spaces))
    end
end

for TYP in (:SumSpace,:PiecewiseSpace),(Op,OpWrap) in ((:PseudoHilbert,:PseudoHilbertWrapper),
                          (:Hilbert,:HilbertWrapper),
                          (:SingularIntegral,:SingularIntegralWrapper))
    @eval $Op(S::$TYP,k) = $OpWrap(InterlaceOperator(Diagonal([map(s->$Op(s,k),S.spaces)...]),$TYP),k)
end


for (Op,OpWrap) in ((:PseudoHilbert,:PseudoHilbertWrapper),
                          (:Hilbert,:HilbertWrapper),
                          (:SingularIntegral,:SingularIntegralWrapper))
    @eval function $Op(S::ArraySpace,k)
        ops = map(s->$Op(s,k),vec(S.spaces))
        $OpWrap(InterlaceOperator(Diagonal(ops),S,ArraySpace(reshape(map(rangespace,ops),size(S)))),k)
    end
end

# Length catch

ConcreteHilbert(sp::Space{D,R},n) where {D,R<:Complex} =
    ConcreteHilbert{typeof(sp),typeof(n),prectype(sp)}(sp,n)
ConcreteSingularIntegral(sp::Space{D,R},n) where {D,R<:Complex} =
    ConcreteSingularIntegral{typeof(sp),typeof(n),prectype(sp)}(sp,n)


# Override sumspace

for TYP in (:Hilbert,:SingularIntegral)
    ConcOp=Meta.parse("Concrete"*string(TYP))
    @eval function $TYP(F::Fourier{DD,RR},n) where {DD<:Circle,RR}
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


for Typ in (:Hilbert,:SingularIntegral)
    ConcTyp = Meta.parse("Concrete"*string(Typ))
    WrapTyp = Meta.parse(string(Typ)*"Wrapper")
    @eval begin
        $Typ(S::Hardy{s,DD,RR},m::Integer) where {s,DD<:Circle,RR} =
            m ≤ 1 ? $ConcTyp(S,m) : $WrapTyp(Derivative(m-1)*Hilbert(S),m)
        $Typ(S::Laurent{DD,RR},m::Integer) where {DD<:Circle,RR} =
            m ≤ 1 ? $ConcTyp(S,m) : $WrapTyp(Derivative(m-1)*Hilbert(S),m)
        $Typ(S::Fourier{DD,RR},m::Integer) where {DD<:Circle,RR} =
            m ≤ 1 ? $ConcTyp(S,m) : $WrapTyp(Derivative(m-1)*Hilbert(S),m)
    end
end



function getindex(H::ConcreteHilbert{Hardy{true,DD,RR},OT,T},
                  k::Integer,j::Integer) where {DD<:Circle,RR,OT,T}
    ##TODO: Add scale for different radii.
    m = H.order
    d = domain(H)
    r = d.radius

    if k == j
        if m==0
            k == 1 ? -T(2r*log(r)) : T(r/(k-1))
        else
            im*(im*(k-1)/r)^(m-1)
        end
    else
        zero(T)
    end
end

function getindex(H::ConcreteHilbert{Hardy{false,DD,RR},OT,T},
                  k::Integer,j::Integer) where {DD<:Circle,RR,OT,T}
    ##TODO: Add scale for different radii.
    m = H.order
    d = domain(H)
    r = d.radius

    if k == j
        -im*(im*k/r)^(m-1)
    else
        zero(T)
    end
end

function getindex(H::ConcreteHilbert{Laurent{DD,RR},OT,T},
                  k::Integer,j::Integer) where {DD<:Circle,RR,OT,T}
    ##TODO: Add scale for different radii.
    m = H.order
    d = domain(H)
    r = d.radius

    if k == j
        if m == 0
            if k == 1
                -T(2r*log(r))
            elseif isodd(k)
                T(r/((k-1)÷2))
            else
                T(-r/(k÷2))
            end
        else
            if isodd(k)
                T(im*(im*((k-1)÷2)/r)^(m-1))
            else
                -T(im*(im*(k÷2)/r)^(m-1))
            end
        end
    else
        zero(T)
    end
end

function getindex(H::ConcreteHilbert{Fourier{DD,RR},OT,T},k::Integer,j::Integer) where {DD<:Circle,RR,OT,T}
    m = H.order
    d = domain(H)
    r = d.radius
    o = d.orientation

    if m == 0 && k==j
        T(k==1 ? 2r*log(r) : (-r/(k÷2)))
    elseif m == 1
        if k==j==1
            T(o ? im : -im)
        elseif iseven(k) && j==k+1
            -one(T)
        elseif isodd(k) && j==k-1
            one(T)
        else
            zero(T)
        end
    else
        zero(T)
    end
end

function getindex(H::ConcreteSingularIntegral{Hardy{true,DD,RR},OT,T},k::Integer,j::Integer) where {DD<:Circle,RR,OT,T}
##TODO: Add scale for different radii.
    m = H.order
    d = domain(H)
    r = d.radius

    if k == j
        if m == 0
            k == 1 ? T(2r*log(r)) : T(-r/(k-1))
        else
            k == 1 ? zero(T) : im*(im*(k-1)/r)^(m-1)
        end
    else
        zero(T)
    end
end

function getindex(H::ConcreteSingularIntegral{Hardy{false,DD,RR},OT,T},k::Integer,j::Integer) where {DD<:Circle,RR,OT,T}
##TODO: Add scale for different radii.
    m = H.order
    d = domain(H)
    r = d.radius

    if k == j
        -im*(im*k/r)^(m-1)
    else
        zero(T)
    end
end

function getindex(H::ConcreteSingularIntegral{Laurent{DD,RR},OT,T},k::Integer,j::Integer) where {DD<:Circle,RR,OT,T}
    ##TODO: Add scale for different radii.
    m = H.order
    d = domain(H)
    r = d.radius

    if k == j
        if m == 0
            if k == 1
                T(2r*log(r))
            else
                T(-r/(k÷2))
            end
        else
            if isodd(k)
                k == 1 ? zero(T) : im*(im*((k-1)÷2)/r)^(m-1)
            else
                -im*(im*(k÷2)/r)^(m-1)
            end
        end
    else
        zero(T)
    end
end

function getindex(H::ConcreteSingularIntegral{Fourier{DD,RR},OT,T},k::Integer,j::Integer) where {DD<:Circle,RR,OT,T}
    m = H.order
    d = domain(H)
    r = d.radius

    if m == 0
        if k == j
            k == 1 ? T(2r*log(r)) : T(-r/(k÷2))
        end
    elseif m == 1
        if k+1 == j
            -one(T)
        elseif k-1 == j
            one(T)
        else
            zero(T)
        end
    else
        error("SingularIntegral order $(H.order) not implemented for Fourier")
    end
end


## JacobiWeight

for (Op,Len) in ((:Hilbert,:complexlength),
                        (:SingularIntegral,:arclength))
    ConcOp=Meta.parse("Concrete"*string(Op))
    OpWrap=Meta.parse(string(Op)*"Wrapper")

    @eval begin
        function $Op(S::JacobiWeight{Chebyshev{DD,RR},DD},m::Int) where {DD<:IntervalOrSegment,RR}
            if S.α==S.β==-0.5
                if m==0
                    $ConcOp(S,m)
                else
                    d=domain(S)
                    C=(4/$Len(d))^(m-1)
                    $OpWrap(SpaceOperator(
                        ToeplitzOperator([0.],[zeros(m);C]),S,Ultraspherical(m,d)),m)
                end
            elseif S.α==S.β==0.5
                d=domain(S)
                if m==1
                    J=JacobiWeight(0.5,0.5,Ultraspherical(1,d))
                    $OpWrap($Op(J,m)*Conversion(S,J),m)
                else
                    J=JacobiWeight(-0.5,-0.5,Chebyshev(d))
                    $OpWrap($Op(J,m)*Conversion(S,J),m)
                end
            else
                error(string($Op)*" not implemented for parameters $(S.α),$(S.β)")
            end
        end

        function getindex(H::$ConcOp{<:JacobiWeight{<:Chebyshev{<:IntervalOrSegment}},OT,T},k::Integer,j::Integer) where {OT,T}
            sp=domainspace(H)
            @assert H.order == 0
            @assert sp.α==sp.β==-0.5

            if k==j
                d=domain(H)
                C=$Len(d)/4
                -T(k==1 ? -2C*log(C) : 2C/(k-1))
            else
                zero(T)
            end
        end

        # we always have real for n==1
        function $Op(S::JacobiWeight{<:Ultraspherical{Int,<:IntervalOrSegment}},m)
            @assert order(S.space) == 1
            if S.α==S.β==0.5
                if m==1
                    d=domain(S)
                    $OpWrap(SpaceOperator(
                        ToeplitzOperator([-1.0],[0.]),S,m==1 ? Chebyshev(d) : Ultraspherical(m-1,d)),m)
                else
                    $ConcOp(S,m)
                end
            else
                error(string($Op)*" not implemented for parameters $(S.α),$(S.β)")
            end
        end
        function getindex(H::$ConcOp{<:JacobiWeight{<:Ultraspherical{Int,<:IntervalOrSegment}},OT,T},k::Integer,j::Integer) where {OT,T}
            # order(domainspace(H))==1
            m=H.order
            d=domain(H)
            sp=domainspace(H)
            @assert sp.α==sp.β==0.5
            @assert m ≠ 1



            C=(4/$Len(d))^(m-1)
            if m == 0
                if k==j==1
                    T(C*log(C))
                elseif k==j
                    -T(C/(k-1))
                elseif j==k-2
                    T(C/(k-1))
                else
                    zero(T)
                end
            elseif j==k+m-2
                -T(.5C*k/(m-1))
            else
                zero(T)
            end
        end
    end
end





## PseudoHilbert
# The default is Hilbert


# getindex(H::ConcretePseudoHilbert,k::Integer,j::Integer)=Hilbert(H.space,H.order)[k,j]
# bandwidths(H::ConcretePseudoHilbert)=bandwidths(Hilbert(H.space,H.order))
