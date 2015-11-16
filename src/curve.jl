## Interval map

stieltjes{C<:Curve,SS}(S::MappedSpace{SS,C},f,z::Number)=sum(stieltjes(S.space,f,complexroots(domain(S).curve-z)))

function stieltjes{C<:Curve,SS}(S::MappedSpace{SS,C},f,z::Number,s::Bool)
    #project
    rts=complexroots(domain(S).curve-z)
    di=domain(S.space)
    mapreduce(rt->in(rt,di)?stieltjes(S.space,f,rt,s):stieltjes(S.space,f,rt),+,rts)
end

stieltjes{C<:Curve,SS}(S::MappedSpace{SS,C},f,z::Vector)=Complex128[stieltjes(S,f,z[k]) for k=1:size(z,1)]
stieltjes{C<:Curve,SS}(S::MappedSpace{SS,C},f,z::Matrix)=Complex128[stieltjes(S,f,z[k,j]) for k=1:size(z,1),j=1:size(z,2)]

## hilbert on JacobiWeight space mapped by open curves

function hilbert{C<:Curve,JW<:JacobiWeight}(S::MappedSpace{JW,C},f,x::Number)
    #project
    rts=complexroots(domain(S).curve-x)
    di=domain(S.space)
    mapreduce(rt->in(rt,di)?hilbert(S.space,f,rt):-stieltjes(S.space,f,rt)/π,+,rts)
end






#TODO: the branch cuts of this are screwy, and the large z asymptotics may
# be off by an integer constant
function stieltjesintegral{CC<:Chebyshev,S,T,TT}(w::Fun{MappedSpace{S,Curve{CC,T},TT}},z)
    d=domain(w)
    # leading order coefficient
    b=d.curve.coefficients[end]*2^(max(length(d.curve)-2,0))
    g=Fun(w.coefficients,w.space.space)
    g=g*ApproxFun.fromcanonicalD(d,Fun())
    sum(stieltjesintegral(g,complexroots(d.curve-z)))+sum(w)*log(b)
end


function logkernel{CC<:Chebyshev,S,T,TT}(w::Fun{MappedSpace{S,Curve{CC,T},TT}},z)
    d=domain(w)
    # leading order coefficient
    b=d.curve.coefficients[end]*2^(max(length(d.curve)-2,0))
    g=Fun(w.coefficients,w.space.space)
    g=g*abs(ApproxFun.fromcanonicalD(d,Fun()))
    sum(logkernel(g,complexroots(d.curve-z)))+linesum(w)*log(abs(b))/π
end


## Circle map

# pseudo stieltjes is not normalized at infinity
function pseudostieltjes{DD<:Circle,C<:Curve}(S::MappedSpace{Laurent{DD},C},f,z::Number)
    c=domain(S)  # the curve that f lives on
    @assert domain(S.space)==Circle()

    sum(stieltjes(S.space,f,complexroots(c.curve-z)))
end

function stieltjes{DD<:Circle,C<:Curve}(S::MappedSpace{Laurent{DD},C},f,z::Number)
    c=domain(S)  # the curve that f lives on
    @assert domain(S.space)==Circle()
    # subtract out value at infinity, determined by the fact that leading term is poly
    # we find the
    sum(stieltjes(S.space,f,complexroots(c.curve-z)))-div(length(domain(S).curve),2)*stieltjes(S.space,f,0.)
end

function stieltjes{DD<:Circle,C<:Curve}(S::MappedSpace{Laurent{DD},C},f,z::Number,s::Bool)
    c=domain(S)  # the curve that f lives on
    @assert domain(S.space)==Circle()
    rts=complexroots(c.curve-z)


    ret=-div(length(domain(S).curve),2)*stieltjes(S.space,f,0.)

    for k=2:length(rts)
        ret+=in(rts[k],Circle())?stieltjes(S.space,f,rts[k]):stieltjes(S.space,f,rts[k],s)
    end
    ret
end


function hilbert{DD<:Circle,C<:Curve}(S::MappedSpace{Laurent{DD},C},f,z::Number)
    c=domain(S)  # the curve that f lives on
    @assert domain(S.space)==Circle()
    rts=complexroots(c.curve-z)


    ret=div(length(domain(S).curve),2)*stieltjes(S.space,f,0.)/π

    for k=2:length(rts)
        ret+=in(rts[k],Circle())?hilbert(S.space,f,rts[k]):stieltjes(S.space,f,rts[k])/(-π)
    end
    ret
end



## CurveSpace

function Hilbert{C<:Curve,JW<:JacobiWeight,T}(S::MappedSpace{JW,C,T},k::Int)
    @assert k==1
    tol=1E-15
    # the mapped logkernel
    Σ=Hilbert(S.space)
    rs=rangespace(Σ)
    d=domain(S)

    # find the number of coefficients needed to resolve the first column
    m=length(Fun(x->sum(stieltjes(Fun([1.0],S.space),filter(y->!in(y,Interval()),complexroots(d.curve-fromcanonical(d,x))))/π),rs))
    #precompute the roots
    rts=Vector{Complex128}[filter(y->!in(y,Interval()),complexroots(d.curve-x)) for x in fromcanonical(d,points(rs,m))]

    # generate cols until smaller than tol
    cols=Array(Vector{Complex128},0)
    for k=1:10000
        push!(cols,transform(rs,Complex128[-sum(stieltjes(Fun([zeros(k-1);1.0],S.space),rt)/π) for rt in rts]))
        if norm(last(cols))<tol
            break
        end
    end

    K=hcat(cols...)
    A=Σ+SpaceOperator(FiniteOperator(K),S.space,rs)

    # Multiply by |r'(t)| to get arclength

    SpaceOperator(A,S,MappedSpace(d,rs))
end


function SingularIntegral{CC<:Chebyshev,JW,TT}(S::MappedSpace{JW,Curve{CC,TT}},k::Integer)
    @assert k==0
    tol=1E-15
    # the mapped logkernel
    Σ=SingularIntegral(S.space,0)
    rs=rangespace(Σ)
    d=domain(S)

    # hiighest order coeff
    b=d.curve.coefficients[end]*2^(max(length(d.curve)-2,0))


    # find the number of coefficients needed to resolve the first column
    m=length(Fun(x->sum(logkernel(Fun([1.0],S.space),filter(y->!in(y,Interval()),complexroots(d.curve-fromcanonical(d,x))))),rs))
    #precompute the roots
    rts=Vector{Complex128}[filter(y->!in(y,Interval()),complexroots(d.curve-x)) for x in fromcanonical(d,points(rs,m))]

    # generate cols until smaller than tol
    cols=Array(Vector{Float64},0)
    for k=1:10000
        push!(cols,transform(rs,Float64[sum(logkernel(Fun([zeros(k-1);1.0],S.space),rt)) for rt in rts]))
        if norm(last(cols))<tol
            break
        end
    end

    K=hcat(cols...)
    A=Σ+SpaceOperator(FiniteOperator(K),S.space,rs)+(log(abs(b))/π)*DefiniteLineIntegral(S.space)

    # Multiply by |r'(t)| to get arclength
    M=Multiplication(abs(fromcanonicalD(d,Fun(identity,S.space))),S.space)

    SpaceOperator(A*M,S,MappedSpace(d,rs))
end
