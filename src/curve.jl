## Interval map

function stieltjes{C<:Curve,S,T,BT}(f::Fun{MappedSpace{S,C,BT},T},z::Number)
    #project
    fm=Fun(f.coefficients,space(f).space)
    sum(stieltjes(fm,complexroots(domain(f).curve-z)))
end

function stieltjes{C<:Curve,S,BT,T}(f::Fun{MappedSpace{S,C,BT},T},z::Number,s::Bool)
    #project
    fm=Fun(f.coefficients,space(f).space)
    rts=complexroots(domain(f).curve-z)
    di=domain(fm)
    mapreduce(rt->in(rt,di)?stieltjes(fm,rt,s):stieltjes(fm,rt),+,rts)
end

stieltjes{C<:Curve,S,T,BT}(f::Fun{MappedSpace{S,C,BT},T},z::Vector)=Complex128[stieltjes(f,z[k]) for k=1:size(z,1)]
stieltjes{C<:Curve,S,T,BT}(f::Fun{MappedSpace{S,C,BT},T},z::Matrix)=Complex128[stieltjes(f,z[k,j]) for k=1:size(z,1),j=1:size(z,2)]

## hilbert on JacobiWeight space mapped by open curves

function hilbert{C<:Curve,JW<:JacobiWeight,T,BT}(f::Fun{MappedSpace{JW,C,BT},T},x::Number)
    #project
    fm=Fun(f.coefficients,space(f).space)
    rts=complexroots(domain(f).curve-x)
    di=domain(fm)
    mapreduce(rt->in(rt,di)?hilbert(fm,rt):-stieltjes(fm,rt)/π,+,rts)
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
function pseudostieltjes{DD<:Circle,C<:Curve}(f::Fun{MappedSpace{Laurent{DD},C,Complex{Float64}}},z::Number)
    fcirc=Fun(f.coefficients,f.space.space)  # project to circle
    c=domain(f)  # the curve that f lives on
    @assert domain(fcirc)==Circle()

    sum(stieltjes(fcirc,complexroots(c.curve-z)))
end

function stieltjes{DD<:Circle,C<:Curve}(f::Fun{MappedSpace{Laurent{DD},C,Complex{Float64}}},z::Number)
    fcirc=Fun(f.coefficients,f.space.space)  # project to circle
    c=domain(f)  # the curve that f lives on
    @assert domain(fcirc)==Circle()
    # subtract out value at infinity, determined by the fact that leading term is poly
    # we find the
    sum(stieltjes(fcirc,complexroots(c.curve-z)))-div(length(domain(f).curve),2)*stieltjes(fcirc,0.)
end

function stieltjes{DD<:Circle,C<:Curve}(f::Fun{MappedSpace{Laurent{DD},C,Complex{Float64}}},z::Number,s::Bool)
    fcirc=Fun(f.coefficients,f.space.space)  # project to circle
    c=domain(f)  # the curve that f lives on
    @assert domain(fcirc)==Circle()
    rts=complexroots(c.curve-z)


    ret=-div(length(domain(f).curve),2)*stieltjes(fcirc,0.)

    for k=2:length(rts)
        ret+=in(rts[k],Circle())?stieltjes(fcirc,rts[k]):stieltjes(fcirc,rts[k],s)
    end
    ret
end


function hilbert{DD<:Circle,C<:Curve}(f::Fun{MappedSpace{Laurent{DD},C,Complex{Float64}}},z::Number)
    fcirc=Fun(f.coefficients,f.space.space)  # project to circle
    c=domain(f)  # the curve that f lives on
    @assert domain(fcirc)==Circle()
    rts=complexroots(c.curve-z)


    ret=div(length(domain(f).curve),2)*stieltjes(fcirc,0.)/π

    for k=2:length(rts)
        ret+=in(rts[k],Circle())?hilbert(fcirc,rts[k]):stieltjes(fcirc,rts[k])/(-π)
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
