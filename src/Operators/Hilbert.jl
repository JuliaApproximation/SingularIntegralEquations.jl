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

        ## Modifiers for SumSpace, ArraySpace, and PiecewiseSpace
        function $Op(S::PiecewiseSpace,n::Int)
            sp=vec(S)
            m=length(sp)
            diag=Any[$Op(sp[k],n) for k=1:m]
            C=Any[k==j?diag[k]:$OffOp(sp[k],rangespace(diag[j]),n) for j=1:m,k=1:m]
            D=Operator{mapreduce(i->eltype(C[i]),promote_type,1:m^2)}[C[j,k] for j=1:m,k=1:m]
            D=promotespaces(D)
            $OpWrap(InterlaceOperator(D,S,PiecewiseSpace(map(rangespace,D[:,1]))),n)
        end

        bandinds{s,DD}(::$ConcOp{Hardy{s,DD}})=0,0
        rangespace{s,DD}(H::$ConcOp{Hardy{s,DD}})=H.space

        bandinds{DD}(::$ConcOp{Laurent{DD}})=0,0
        domainspace{DD}(H::$ConcOp{Laurent{DD}})=H.space
        rangespace{DD}(H::$ConcOp{Laurent{DD}})=H.space

        bandinds{DD}(H::$ConcOp{Fourier{DD}})=-H.order,H.order
        rangespace{DD}(H::$ConcOp{Fourier{DD}})=H.space

        bandinds{DD<:PeriodicInterval}(H::$ConcOp{CosSpace{DD}})=(0,1)
        bandinds{DD<:PeriodicInterval}(H::$ConcOp{SinSpace{DD}})=(-1,0)
        rangespace{DD<:PeriodicInterval}(H::$ConcOp{CosSpace{DD}})=SinSpace(domain(H))
        rangespace{DD<:PeriodicInterval}(H::$ConcOp{SinSpace{DD}})=CosSpace(domain(H))


        function rangespace{DD}(H::$ConcOp{JacobiWeight{Chebyshev{DD},DD}})
            @assert domainspace(H).α==domainspace(H).β==-0.5
            H.order==0 ? Chebyshev(domain(H)) : Ultraspherical(H.order,domain(H))
        end
        function rangespace{DD}(H::$ConcOp{JacobiWeight{Ultraspherical{Int,DD},DD}})
            @assert order(domainspace(H).space) == 1
            @assert domainspace(H).α==domainspace(H).β==0.5
            (H.order==1||H.order==0) ? Chebyshev(domain(H)) : Ultraspherical(H.order-1,domain(H))
        end
        # bandinds{λ,DD}(H::$ConcOp{JacobiWeight{Ultraspherical{λ,DD},DD}})=-λ,H.order-λ
        bandinds{DD}(H::$ConcOp{JacobiWeight{Chebyshev{DD},DD}}) = 0,H.order
        bandinds{DD}(H::$ConcOp{JacobiWeight{Ultraspherical{Int,DD},DD}}) =
            H.order > 0 ? (-1,H.order-1) : (-2,0)

        choosedomainspace(H::$Op{UnsetSpace},sp::Ultraspherical) =
            ChebyshevWeight(ChebyshevDirichlet{1,1}(domain(sp)))
        choosedomainspace(H::$Op{UnsetSpace},sp::Chebyshev) =
            ChebyshevWeight(ChebyshevDirichlet{1,1}(domain(sp)))
        choosedomainspace(H::$Op{UnsetSpace},sp::PiecewiseSpace) =
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

ConcreteHilbert(sp::Space{ComplexBasis},n) =
    ConcreteHilbert{typeof(sp),typeof(n),Complex{real(eltype(domain(sp)))}}(sp,n)
ConcreteSingularIntegral(sp::Space{ComplexBasis},n) =
    ConcreteSingularIntegral{typeof(sp),typeof(n),Complex{real(eltype(domain(sp)))}}(sp,n)


# Override sumspace

for TYP in (:Hilbert,:SingularIntegral)
    ConcOp=parse("Concrete"*string(TYP))
    @eval function $TYP{DD<:Circle}(F::Fourier{DD},n)
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
    ConcTyp = parse("Concrete"*string(Typ))
    WrapTyp = parse(string(Typ)*"Wrapper")
    @eval begin
        $Typ{s,DD<:Circle}(S::Hardy{s,DD},m::Integer) =
            m ≤ 1 ? $ConcTyp(S,m) : $WrapTyp(Derivative(m-1)*Hilbert(S),m)
        $Typ{DD<:Circle}(S::Laurent{DD},m::Integer) =
            m ≤ 1 ? $ConcTyp(S,m) : $WrapTyp(Derivative(m-1)*Hilbert(S),m)
        $Typ{DD<:Circle}(S::Fourier{DD},m::Integer) =
            m ≤ 1 ? $ConcTyp(S,m) : $WrapTyp(Derivative(m-1)*Hilbert(S),m)
    end
end



function getindex{DD<:Circle,OT,T}(H::ConcreteHilbert{Hardy{true,DD},OT,T},k::Integer,j::Integer)
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

function getindex{DD<:Circle,OT,T}(H::ConcreteHilbert{Hardy{false,DD},OT,T},k::Integer,j::Integer)
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

function getindex{DD<:Circle,OT,T}(H::ConcreteHilbert{Laurent{DD},OT,T},k::Integer,j::Integer)
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

function getindex{DD<:Circle,OT,T}(H::ConcreteHilbert{Fourier{DD},OT,T},k::Integer,j::Integer)
    m = H.order
    d = domain(H)
    r = d.radius
    o = d.orientation

    if m == 0 && k==j
        T(k==1?2r*log(r):(-r/(k÷2)))
    elseif m == 1
        if k==j==1
            T(o?im:-im)
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

function getindex{DD<:Circle,OT,T}(H::ConcreteSingularIntegral{Hardy{true,DD},OT,T},k::Integer,j::Integer)
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

function getindex{DD<:Circle,OT,T}(H::ConcreteSingularIntegral{Hardy{false,DD},OT,T},k::Integer,j::Integer)
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

function getindex{DD<:Circle,OT,T}(H::ConcreteSingularIntegral{Laurent{DD},OT,T},k::Integer,j::Integer)
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

function getindex{DD<:Circle,OT,T}(H::ConcreteSingularIntegral{Fourier{DD},OT,T},k::Integer,j::Integer)
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

## PeriodicInterval

function getindex{DD<:PeriodicInterval,OT,T}(H::ConcreteHilbert{Taylor{DD},OT,T},k::Integer,j::Integer)
    d = domain(H)
    a,b = d.a,d.b
    if H.order == 1
        if k>1
            if k == j
                T(im)
            else
                zero(T)
            end
        else
            zero(T)
        end
    else
        error("Hilbert order $(H.order) not implemented for Taylor")
    end
end

function getindex{DD<:PeriodicInterval,OT,T}(H::ConcreteHilbert{Hardy{false,DD},OT,T},k::Integer,j::Integer)
    d = domain(H)
    a,b = d.a,d.b
    if H.order == 1
        if k>1
            if k == j
                T(-im)
            else
                zero(T)
            end
        else
            zero(T)
        end
    else
        error("Hilbert order $(H.order) not implemented for Hardy{false}")
    end
end

function getindex{DD<:PeriodicInterval,OT,T}(H::ConcreteHilbert{CosSpace{DD},OT,T},k::Integer,j::Integer)
    d = domain(H)
    a,b = d.a,d.b
    if H.order == 1
        if k == j-1
            -one(T)
        else
            zero(T)
        end
    else
        error("Hilbert order $(H.order) not implemented for CosSpace")
    end
end

function getindex{DD<:PeriodicInterval,OT,T}(H::ConcreteHilbert{SinSpace{DD},OT,T},k::Integer,j::Integer)
    d = domain(H)
    a,b = d.a,d.b
    if H.order == 1
        if k == j+1
            one(T)
        else
            zero(T)
        end
    else
        error("Hilbert order $(H.order) not implemented for CosSpace")
    end
end

## JacobiWeight

for (Op,Len) in ((:Hilbert,:complexlength),
                        (:SingularIntegral,:arclength))
    ConcOp=parse("Concrete"*string(Op))
    OpWrap=parse(string(Op)*"Wrapper")

    @eval begin
        function $Op{DD<:Segment}(S::JacobiWeight{Chebyshev{DD},DD},m::Int)
            if S.α==S.β==-0.5
                if m==0
                    $ConcOp(S,m)
                else
                    d=domain(S)
                    C=(4./$Len(d))^(m-1)
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

        function getindex{DD<:Segment,OT,T}(H::$ConcOp{JacobiWeight{Chebyshev{DD},DD},OT,T},k::Integer,j::Integer)
            sp=domainspace(H)
            @assert H.order == 0
            @assert sp.α==sp.β==-0.5

            if k==j
                d=domain(H)
                C=$Len(d)/4
                -T(k==1?-2C*log(C):2C/(k-1))
            else
                zero(T)
            end
        end

        # we always have real for n==1
        function $Op{DD<:Segment}(S::JacobiWeight{Ultraspherical{Int,DD},DD},m)
            @assert order(S.space) == 1
            if S.α==S.β==0.5
                if m==1
                    d=domain(S)
                    $OpWrap(SpaceOperator(
                        ToeplitzOperator([-1.0],[0.]),S,m==1?Chebyshev(d):Ultraspherical(m-1,d)),m)
                else
                    $ConcOp(S,m)
                end
            else
                error(string($Op)*" not implemented for parameters $(S.α),$(S.β)")
            end
        end
        function getindex{DD<:Segment,OT,T}(H::$ConcOp{JacobiWeight{Ultraspherical{Int,DD},DD},OT,T},k::Integer,j::Integer)
            # order(domainspace(H))==1
            m=H.order
            d=domain(H)
            sp=domainspace(H)
            @assert sp.α==sp.β==0.5
            @assert m ≠ 1



            C=(4./$Len(d))^(m-1)
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
# bandinds(H::ConcretePseudoHilbert)=bandinds(Hilbert(H.space,H.order))
