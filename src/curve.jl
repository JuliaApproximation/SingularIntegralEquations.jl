## Interval map

function cauchy{C<:Curve,S,T,BT}(f::Fun{MappedSpace{S,C,BT},T},z::Number)
    #project
    fm=Fun(f.coefficients,space(f).space)
    sum(cauchy(fm,complexroots(domain(f).curve-z)))
end

function cauchy{C<:Curve,S,BT,T}(s::Bool,f::Fun{MappedSpace{S,C,BT},T},z::Number)
    #project
    fm=Fun(f.coefficients,space(f).space)
    rts=complexroots(domain(f).curve-z)
    di=Interval()
    mapreduce(rt->in(rt,di)?cauchy(s,fm,rt):cauchy(fm,rt),+,rts)
end





#TODO: the branch cuts of this are screwy, and the large z asymptotics may
# be off by an integer constant
function cauchyintegral{S,T,TT}(w::Fun{MappedSpace{S,Curve{Chebyshev,T},TT}},z)
    d=domain(w)
    # leading order coefficient
    b=d.curve.coefficients[end]*2^(max(length(d.curve)-2,0))
    g=Fun(w.coefficients,w.space.space)
    g=g*ApproxFun.fromcanonicalD(d,Fun())
    sum(cauchyintegral(g,complexroots(d.curve-z)))+sum(w)*log(b)/(-2π*im)
end


function logkernel{S,T,TT}(w::Fun{MappedSpace{S,Curve{Chebyshev,T},TT}},z)
    d=domain(w)
    # leading order coefficient
    b=d.curve.coefficients[end]*2^(max(length(d.curve)-2,0))
    g=Fun(w.coefficients,w.space.space)
    g=g*abs(ApproxFun.fromcanonicalD(d,Fun()))
    sum(logkernel(g,complexroots(d.curve-z)))+linesum(w)*log(abs(b))/π
end


## Circle map

# pseudo cauchy is not normalized at infinity
function pseudocauchy{C<:Curve}(f::Fun{MappedSpace{Laurent,C,Complex{Float64}}},z::Number)
    fcirc=Fun(f.coefficients,f.space.space)  # project to circle
    c=domain(f)  # the curve that f lives on
    @assert domain(fcirc)==Circle()

    sum(cauchy(fcirc,complexroots(c.curve-z)))
end

function cauchy{C<:Curve}(f::Fun{MappedSpace{Laurent,C,Complex{Float64}}},z::Number)
    fcirc=Fun(f.coefficients,f.space.space)  # project to circle
    c=domain(f)  # the curve that f lives on
    @assert domain(fcirc)==Circle()
    # subtract out value at infinity, determined by the fact that leading term is poly
    # we find the
    sum(cauchy(fcirc,complexroots(c.curve-z)))-div(length(domain(f).curve),2)*cauchy(fcirc,0.)
end

function cauchy{C<:Curve}(s::Bool,f::Fun{MappedSpace{Laurent,C,Complex{Float64}}},z::Number)
    fcirc=Fun(f.coefficients,f.space.space)  # project to circle
    c=domain(f)  # the curve that f lives on
    @assert domain(fcirc)==Circle()
    rts=complexroots(c.curve-z)


    ret=-div(length(domain(f).curve),2)*cauchy(fcirc,0.)

    for k=2:length(rts)
        ret+=in(rts[k],Circle())?cauchy(s,fcirc,rts[k]):cauchy(fcirc,rts[k])
    end
    ret
end


function hilbert{C<:Curve}(f::Fun{MappedSpace{Laurent,C,Complex{Float64}}},z::Number)
    fcirc=Fun(f.coefficients,f.space.space)  # project to circle
    c=domain(f)  # the curve that f lives on
    @assert domain(fcirc)==Circle()
    rts=complexroots(c.curve-z)


    ret=-2im*div(length(domain(f).curve),2)*cauchy(fcirc,0.)

    for k=2:length(rts)
        ret+=in(rts[k],Circle())?hilbert(fcirc,rts[k]):2im*cauchy(fcirc,rts[k])
    end
    ret
end