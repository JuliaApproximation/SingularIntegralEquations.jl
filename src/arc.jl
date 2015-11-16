

## Cauchy

# pseudocauchy does not normalize at ∞
pseudostieltjes{LS,RR<:Arc}(S::MappedSpace{LS,RR},f,z,s...)=stieltjes(S.space,f,tocanonical(f,z),s...)
pseudohilbert{LS,RR<:Arc}(S::MappedSpace{LS,RR},f,z)=hilbert(S.space,f,tocanonical(f,z))


stieltjes{LS,RR<:Arc}(S::MappedSpace{LS,RR},f,z,s...)=stieltjes(S.space,f,tocanonical(S,z),s...)-stieltjes(S.space,f,tocanonical(S,Inf))
hilbert{LS,RR<:Arc}(S::MappedSpace{LS,RR},f,z)=hilbert(S.space,f,tocanonical(S,z))+(1/π)*stieltjes(S.space,f,tocanonical(S,Inf))





function PseudoHilbert{LS,RR<:Arc,TT}(sp::MappedSpace{LS,RR,TT},k::Integer)
    @assert k==1
    csp=sp.space
    H=Hilbert(csp)
    PseudoHilbertWrapper(SpaceOperator(H,MappedSpace(domain(sp),domainspace(H)),MappedSpace(domain(sp),rangespace(H))),k)
end


function Hilbert{LS,RR<:Arc,TT}(sp::MappedSpace{LS,RR,TT},k::Integer)
    @assert k==1
    csp=sp.space
    H=Hilbert(csp)+(1/π)*Stieltjes(csp,tocanonical(sp,Inf))
    HilbertWrapper(SpaceOperator(H,MappedSpace(domain(sp),domainspace(H)),MappedSpace(domain(sp),rangespace(H))),k)
end



## stieltjesintegral

function stieltjesintegral{LS,RR<:Arc,TT}(w::Fun{MappedSpace{LS,RR,TT}},z)
    g=Fun(w.coefficients,w.space.space)*fromcanonicalD(w,Fun())
    stieltjesintegral(g,tocanonical(w,z))-
        stieltjesintegral(g,tocanonical(w,Inf))+
        sum(w)*log(z-fromcanonical(w,Inf))
end


function linesumstieltjesintegral{LS,RR<:Arc,TT}(w::Fun{MappedSpace{LS,RR,TT}},z)
    g=Fun(w.coefficients,w.space.space)*abs(ApproxFun.fromcanonicalD(w,Fun()))
    stieltjesintegral(g,tocanonical(w,z))-
        stieltjesintegral(g,tocanonical(w,Inf))+
        linesum(w)*log(z-fromcanonical(w,Inf))
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
    L∞=FiniteFunctional(cnst,S.space)

    x=Fun(identity,S)
    SpaceOperator((Σ-L∞)*M,S,MappedSpace(d,rangespace(Σ)))+(log(abs(x-fromcanonical(d,Inf)))/π)*DefiniteLineIntegral(S)
end
