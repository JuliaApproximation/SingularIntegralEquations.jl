## Interval map

stieltjes{C<:Curve,SS}(S::Space{SS,C},f,z::Number)=sum(stieltjes(setcanonicaldomain(S),f,complexroots(domain(S).curve-z)))

function stieltjes{C<:Curve,SS}(S::Space{SS,C},f,z::Number,s::Bool)
    #project
    rts=complexroots(domain(S).curve-z)
    di=domain(S.space)
    mapreduce(rt->in(rt,di)?stieltjes(S.space,f,rt,s):stieltjes(S.space,f,rt),+,rts)
end

stieltjes{C<:Curve,SS}(S::Space{SS,C},f,z::Vector)=Complex128[stieltjes(S,f,z[k]) for k=1:size(z,1)]
stieltjes{C<:Curve,SS}(S::Space{SS,C},f,z::Matrix)=Complex128[stieltjes(S,f,z[k,j]) for k=1:size(z,1),j=1:size(z,2)]

## hilbert on JacobiWeight space mapped by open curves

function hilbert{C<:Curve,SS}(S::JacobiWeight{SS,C},f,x::Number)
    #project
    rts=complexroots(domain(S).curve-x)
    csp=setcanonicaldomain(S)
    di=domain(csp)
    mapreduce(rt->in(rt,di)?hilbert(csp,f,rt):-stieltjes(csp,f,rt)/π,+,rts)
end






#TODO: the branch cuts of this are screwy, and the large z asymptotics may
# be off by an integer constant
function stieltjesintegral{CC<:Chebyshev,S,T}(sp::Space{S,IntervalCurve{CC,T}},w,z)
    d=domain(sp)
    # leading order coefficient
    b=d.curve.coefficients[end]*2^(max(length(d.curve)-2,0))
    g=Fun(w,setcanonicaldomain(sp))
    g=g*fromcanonicalD(d)
    sum(stieltjesintegral(g,complexroots(d.curve-z)))+sum(Fun(w,sp))*log(b)
end


function logkernel{CC<:Chebyshev,S,T}(sp::Space{S,IntervalCurve{CC,T}},w,z)
    d=domain(sp)
    # leading order coefficient
    b=d.curve.coefficients[end]*2^(max(length(d.curve)-2,0))
    g=Fun(w,setcanonicaldomain(sp))
    g=g*abs(fromcanonicalD(d))
    sum(logkernel(g,complexroots(d.curve-z)))+linesum(Fun(w,sp))*log(abs(b))/π
end


## Circle map

# pseudo stieltjes is not normalized at infinity
function pseudostieltjes{C<:PeriodicCurve}(S::Laurent{C},f,z::Number)
    c=domain(S)  # the curve that f lives on
    @assert domain(c.curve)==Circle()

    sum(stieltjes(setdomain(S,Circle()),f,complexroots(c.curve-z)))
end

function stieltjes{C<:PeriodicCurve}(S::Laurent{C},f,z::Number)
    c=domain(S)  # the curve that f lives on
    @assert domain(c.curve)==Circle()
    # subtract out value at infinity, determined by the fact that leading term is poly
    # we find the
    sum(stieltjes(setdomain(S,Circle()),f,complexroots(c.curve-z)))-
            div(length(domain(S).curve),2)*stieltjes(setdomain(S,Circle()),f,0.)
end

function stieltjes{C<:PeriodicCurve}(S::Laurent{C},f,z::Number,s::Bool)
    c=domain(S)  # the curve that f lives on
    @assert domain(c.curve)==Circle()
    rts=complexroots(c.curve-z)

    csp=setdomain(S,Circle())
    ret=-div(length(domain(S).curve),2)*stieltjes(csp,f,0.)

    for k=2:length(rts)
        ret+=in(rts[k],Circle())?stieltjes(csp,f,rts[k]):stieltjes(csp,f,rts[k],s)
    end
    ret
end


function hilbert{C<:PeriodicCurve}(S::Laurent{C},f,z::Number)
    c=domain(S)  # the curve that f lives on
    @assert domain(c.curve)==Circle()
    rts=complexroots(c.curve-z)

    csp=setdomain(S,Circle())
    ret=div(length(domain(S).curve),2)*stieltjes(csp,f,0.)/π

    for k=2:length(rts)
        ret+=in(rts[k],Circle())?hilbert(csp,f,rts[k]):stieltjes(csp,f,rts[k])/(-π)
    end
    ret
end



## CurveSpace

function Hilbert{C<:IntervalCurve,SS}(S::JacobiWeight{SS,C},k::Int)
    @assert k==1
    tol=1E-15

    csp=setcanonicaldomain(S)
    # the mapped logkernel
    Σ=Hilbert(csp)
    rs=rangespace(Σ)
    d=domain(S)

    # find the number of coefficients needed to resolve the first column
    m=length(Fun(x->sum(stieltjes(Fun([1.0],csp),filter(y->!in(y,Interval()),complexroots(d.curve-fromcanonical(d,x))))/π),rs))
    #precompute the roots
    rts=Vector{Complex128}[filter(y->!in(y,Interval()),complexroots(d.curve-x)) for x in fromcanonical(d,points(rs,m))]

    # generate cols until smaller than tol
    cols=Array(Vector{Complex128},0)
    for k=1:10000
        push!(cols,transform(rs,Complex128[-sum(stieltjes(Fun([zeros(k-1);1.0],csp),rt)/π) for rt in rts]))
        if norm(last(cols))<tol
            break
        end
    end

    K=hcat(cols...)
    A=Σ+SpaceOperator(FiniteOperator(K),csp,rs)

    # Multiply by |r'(t)| to get arclength

    SpaceOperator(A,S,setdomain(rs,d))
end


function SingularIntegral{CC<:Chebyshev,TTT,TT}(S::JacobiWeight{TTT,IntervalCurve{CC,TT}},k::Integer)
    @assert k==0
    tol=1E-15
    # the mapped logkernel
    csp=setcanonicaldomain(S)
    Σ=SingularIntegral(csp,0)
    rs=rangespace(Σ)
    d=domain(S)

    # hiighest order coeff
    b=d.curve.coefficients[end]*2^(max(length(d.curve)-2,0))


    # find the number of coefficients needed to resolve the first column
    m=length(Fun(x->sum(logkernel(Fun([1.0],csp),filter(y->!in(y,Interval()),complexroots(d.curve-fromcanonical(d,x))))),rs))
    #precompute the roots
    rts=Vector{Complex128}[filter(y->!in(y,Interval()),complexroots(d.curve-x)) for x in fromcanonical(d,points(rs,m))]

    # generate cols until smaller than tol
    cols=Array(Vector{Float64},0)
    for k=1:10000
        push!(cols,transform(rs,Float64[sum(logkernel(Fun([zeros(k-1);1.0],csp),rt)) for rt in rts]))
        if norm(last(cols))<tol
            break
        end
    end

    K=hcat(cols...)
    A=Σ+SpaceOperator(FiniteOperator(K),csp,rs)+(log(abs(b))/π)*DefiniteLineIntegral(csp)

    # Multiply by |r'(t)| to get arclength
    M=Multiplication(abs(fromcanonicalD(d,Fun(identity,csp))),csp)

    SpaceOperator(A*M,S,setdomain(rs,d))
end
