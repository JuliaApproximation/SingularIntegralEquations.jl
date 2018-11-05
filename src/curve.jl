## Segment map

stieltjes(S::Space{<:Curve},f,z::Number) =
    sum(stieltjes.(Fun(setcanonicaldomain(S),f),complexroots(domain(S).curve-z)))

function stieltjes(S::Space{<:Curve},f,z::Directed{s}) where {s}
    #project
    rts=complexroots(domain(S).curve-z.x)
    di=domain(S.space)
    tol = 10E-15
    mapreduce(rt->abs(imag(rt)) < tol && in(real(rt),di)  ? stieltjes(S.space,f,Directed{s}(real(rt))) :
                              stieltjes(S.space,f,rt),+,rts)
end

## hilbert on JacobiWeight space mapped by open curves

function hilbert(S::JacobiWeight{SS,C},f,x::Number) where {C<:Curve,SS}
    #project
    rts=complexroots(domain(S).curve-x)
    csp=setcanonicaldomain(S)
    di=domain(csp)
    tol = 10E-15
    mapreduce(rt->abs(imag(rt)) < tol && in(real(rt),di) ?  hilbert(csp,f,real(rt)) :
                              -stieltjes(csp,f,rt)/π,+,rts)
end



#TODO: the branch cuts of this are screwy, and the large z asymptotics may
# be off by an integer constant
function stieltjesintegral(sp::Space{IntervalCurve{CC,T,VT}},w,z) where {CC<:Chebyshev,T,VT}
    d=domain(sp)
    # leading order coefficient
    b=d.curve.coefficients[end]*2^(max(ncoefficients(d.curve)-2,0))
    g=Fun(setcanonicaldomain(sp),w)
    g=g*fromcanonicalD(d)
    sum(stieltjesintegral.(g,complexroots(d.curve-z)))+sum(Fun(sp,w))*log(b)
end


function logkernel(sp::Space{IntervalCurve{CC,T,VT}},w,z) where {CC<:Chebyshev,T,VT}
    d=domain(sp)
    # leading order coefficient
    b=d.curve.coefficients[end]*2^(max(ncoefficients(d.curve)-2,0))
    g=Fun(setcanonicaldomain(sp),w)
    g=g*abs(fromcanonicalD(d))
    sum(logkernel.(g,complexroots(d.curve-z)))+linesum(Fun(sp,w))*logabs(b)/π
end


## Circle map

# pseudo stieltjes is not normalized at infinity
function pseudostieltjes(S::Laurent{C,R},f,z::Number) where {C<:PeriodicCurve,R}
    c=domain(S)  # the curve that f lives on
    @assert domain(c.curve)==Circle()

    sum(stieltjes.(Fun(setdomain(S,Circle()),f),complexroots(c.curve-z)))
end

function stieltjes(S::Laurent{C,R},f,z::Number) where {C<:PeriodicCurve,R}
    c=domain(S)  # the curve that f lives on
    @assert domain(c.curve)==Circle()
    # subtract out value at infinity, determined by the fact that leading term is poly
    # we find the
    sum(stieltjes.(Fun(setdomain(S,Circle()),f),complexroots(c.curve-z)))-
            div(ncoefficients(domain(S).curve),2)*stieltjes(setdomain(S,Circle()),f,0.)
end

function stieltjes(S::Laurent{C,R},f,z::Directed{s}) where {s,C<:PeriodicCurve,R}
    c=domain(S)  # the curve that f lives on
    @assert domain(c.curve)==Circle()
    rts=complexroots(c.curve-z.x)

    csp=setdomain(S,Circle())
    ret=-div(ncoefficients(domain(S).curve),2)*stieltjes(csp,f,0.)

    for k=2:length(rts)
        ret+=in(rts[k],Circle()) ? stieltjes(csp,f,Directed{s}(rts[k])) :
                                   stieltjes(csp,f,rts[k])
    end
    ret
end


function hilbert(S::Laurent{C,R},f,z::Number) where {C<:PeriodicCurve,R}
    c=domain(S)  # the curve that f lives on
    @assert domain(c.curve)==Circle()
    rts=complexroots(c.curve-z)

    csp=setdomain(S,Circle())
    ret=div(ncoefficients(domain(S).curve),2)*stieltjes(csp,f,0.)/π

    for k=2:length(rts)
        ret+=in(rts[k],Circle()) ? hilbert(csp,f,rts[k]) :
                                   stieltjes(csp,f,rts[k])/(-π)
    end
    ret
end



## CurveSpace

function Hilbert(S::JacobiWeight{SS,C},k::Int) where {C<:IntervalCurve,SS}
    @assert k==1
    tol=1E-15

    csp=setcanonicaldomain(S)
    # the mapped logkernel
    Σ=Hilbert(csp)
    rs=rangespace(Σ)
    d=domain(S)

    # find the number of coefficients needed to resolve the first column
    m=ncoefficients(Fun(x->sum(stieltjes.(Fun(csp,[1.0]),filter(y->abs(imag(y)) > tol || (real(y) ∉ ChebyshevInterval()),
                                                                complexroots(d.curve-fromcanonical(d,x))))/π),rs))
    #precompute the roots
    rts=Vector{ComplexF64}[filter(y->abs(imag(y)) > tol || (real(y) ∉ ChebyshevInterval()),
                                 complexroots(d.curve-x)) for x in fromcanonical.(Ref(d),points(rs,m))]

    # generate cols until smaller than tol
    cols=Vector{Vector{ComplexF64}}()
    for k=1:10000
        push!(cols,transform(rs,ComplexF64[-sum(stieltjes.(Fun(csp,[zeros(k-1);1.0]),rt)/π) for rt in rts]))
        if norm(last(cols))<tol
            break
        end
    end

    K=hcat(cols...)
    A=Σ+FiniteOperator(K,csp,rs)

    # Multiply by |r'(t)| to get arclength

    SpaceOperator(A,S,setdomain(rs,d))
end


function SingularIntegral(S::JacobiWeight{TTT,IntervalCurve{CC,TT,VT}},k::Integer) where {CC<:Chebyshev,TTT,TT,VT}
    @assert k==0
    tol=1E-15
    # the mapped logkernel
    csp=setcanonicaldomain(S)
    Σ=SingularIntegral(csp,0)
    rs=rangespace(Σ)
    d=domain(S)

    # hiighest order coeff
    b=d.curve.coefficients[end]*2^(max(ncoefficients(d.curve)-2,0))


    # find the number of coefficients needed to resolve the first column
    m=ncoefficients(Fun(x->sum(logkernel.(Fun(csp,[1.0]),filter(y->abs(imag(y)) > tol || (real(y) ∉ ChebyshevInterval()),
                                                                complexroots(d.curve-fromcanonical(d,x))))),rs))
    #precompute the roots
    rts=Vector{ComplexF64}[filter(y->abs(imag(y)) > tol || (real(y) ∉ ChebyshevInterval()),
                                  complexroots(d.curve-x)) for x in fromcanonical.(Ref(d),points(rs,m))]

    # generate cols until smaller than tol
    cols=Vector{Vector{Float64}}()
    for k=1:10000
        push!(cols,transform(rs,Float64[sum(logkernel.(Fun(csp,[zeros(k-1);1.0]),rt)) for rt in rts]))
        if norm(last(cols))<tol
            break
        end
    end

    K=hcat(cols...)
    A=Σ+FiniteOperator(K,csp,rs)+(logabs(b)/π)*DefiniteLineIntegral(csp)

    # Multiply by |r'(t)| to get arclength
    M=Multiplication(abs(fromcanonicalD(d,Fun(identity,csp))),csp)

    SpaceOperator(A*M,S,setdomain(rs,d))
end
