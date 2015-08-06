

## Cauchy


function cauchy{LS,RR<:Arc,TT,T}(f::Fun{MappedSpace{LS,RR,TT},T},z)
    g=Fun(f.coefficients,space(f).space)
    cauchy(g,tocanonical(f,z))-cauchy(g,tocanonical(f,Inf))
end

function cauchy{LS,RR<:Arc,TT,T}(s,f::Fun{MappedSpace{LS,RR,TT},T},z)
    g=Fun(f.coefficients,space(f).space)
    cauchy(s,g,tocanonical(f,z))-cauchy(g,tocanonical(f,Inf))
end

function hilbert{LS,RR<:Arc,TT,T}(f::Fun{MappedSpace{LS,RR,TT},T},z)
    g=Fun(f.coefficients,space(f).space)
    hilbert(g,tocanonical(f,z))+(1/π)*stieltjes(g,tocanonical(f,Inf))
end



function Hilbert{LS,RR<:Arc,TT}(sp::MappedSpace{LS,RR,TT},k::Integer)
    @assert k==1
    csp=sp.space
    H=Hilbert(csp)+(1/π)*Stieltjes(csp,tocanonical(sp,Inf))
    HilbertWrapper(SpaceOperator(H,MappedSpace(domain(sp),domainspace(H)),MappedSpace(domain(sp),rangespace(H))),k)
end



## cauchyintegral

function cauchyintegral{LS,RR<:Arc,TT}(w::Fun{MappedSpace{LS,RR,TT}},z)
    g=Fun(w.coefficients,w.space.space)*ApproxFun.fromcanonicalD(w,Fun())
    cauchyintegral(g,tocanonical(w,z))-
        cauchyintegral(g,tocanonical(w,Inf))+
        sum(w)*log(z-fromcanonical(w,Inf))/(-2π*im)
end


function linesumcauchyintegral{LS,RR<:Arc,TT}(w::Fun{MappedSpace{LS,RR,TT}},z)
    g=Fun(w.coefficients,w.space.space)*abs(ApproxFun.fromcanonicalD(w,Fun()))
    cauchyintegral(g,tocanonical(w,z))-
        cauchyintegral(g,tocanonical(w,Inf))+
        linesum(w)*log(z-fromcanonical(w,Inf))/(-2π*im)
end


function logkernel{LS,RR<:Arc,TT}(w::Fun{MappedSpace{LS,RR,TT}},z)
    g=Fun(w.coefficients,w.space.space)*abs(ApproxFun.fromcanonicalD(w,Fun()))
    logkernel(g,tocanonical(w,z))-
        logkernel(g,tocanonical(w,Inf))+
        linesum(w)*log(abs(z-fromcanonical(w,Inf)))/π
end


function SingularIntegral{JW,RR<:Arc}(S::MappedSpace{JW,RR},k::Integer)
    @assert k==0
    tol=1E-15
    # the mapped logkernel
    d=domain(S)
    Σ=SingularIntegral(S.space,0)
    M=Multiplication(abs(fromcanonicalD(d,Fun(identity,S.space))),S.space)

    z∞=tocanonical(d,Inf)
    cnst=Array(Float64,0)
    for k=1:10000
        push!(cnst,logkernel(Fun([zeros(k-1);1.],S.space),z∞))
        if k≥3&&norm(cnst[end-2:end])<tol
            break
        end
    end
    L∞=CompactFunctional(cnst,S.space)

    x=Fun(identity,S)
    SpaceOperator((Σ-L∞)*M,S,MappedSpace(d,rangespace(Σ)))+(log(abs(x-fromcanonical(d,Inf)))/π)*DefiniteLineIntegral(S)
end




