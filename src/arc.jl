

## Cauchy

# pseudocauchy does not normalize at ∞
pseudostieltjes{LS,RR<:Arc}(S::Space{LS,RR},f,z,s...)=stieltjes(setcanonicaldomain(S),f,tocanonical(S,z),s...)
pseudohilbert{LS,RR<:Arc}(S::Space{LS,RR},f,z)=hilbert(setdomain(S,Interval()),f,tocanonical(S,z))


stieltjes{LS,RR<:Arc}(S::Space{LS,RR},f,z,s...)=stieltjes(setcanonicaldomain(S),f,tocanonical(S,z),s...)-stieltjes(setcanonicaldomain(S),f,tocanonical(S,Inf))
hilbert{LS,RR<:Arc}(S::Space{LS,RR},f,z)=hilbert(setcanonicaldomain(S),f,tocanonical(S,z))+(1/π)*stieltjes(setcanonicaldomain(S),f,tocanonical(S,Inf))





function PseudoHilbert{LS,RR<:Arc}(sp::JacobiWeight{LS,RR},k::Integer)
    @assert k==1
    csp=setcanonicaldomain(sp)
    H=Hilbert(csp)
    PseudoHilbertWrapper(SpaceOperator(H,setdomain(domainspace(H),domain(sp)),setdomain(rangespace(H),domain(sp))),k)
end


function Hilbert{LS,RR<:Arc}(sp::JacobiWeight{LS,RR},k::Integer)
    @assert k==1
    csp=setcanonicaldomain(sp)
    H=Hilbert(csp)+(1/π)*Stieltjes(csp,tocanonical(sp,Inf))
    HilbertWrapper(SpaceOperator(H,setdomain(domainspace(H),domain(sp)),setdomain(rangespace(H),domain(sp))),k)
end



## stieltjesintegral

function stieltjesintegral{LS,RR<:Arc}(sp::Space{LS,RR},w,z)
    g=Fun(w,setcanonicaldomain(sp))*fromcanonicalD(sp)
    stieltjesintegral(g,tocanonical(sp,z))-
        stieltjesintegral(g,tocanonical(sp,Inf))+
        sum(Fun(w,sp))*log(z-fromcanonical(sp,Inf))
end


function linesumstieltjesintegral{LS,RR<:Arc}(sp::Space{LS,RR},w,z)
    g=Fun(w,setcanonicaldomain(sp))*abs(fromcanonicalD(sp))
    stieltjesintegral(g,tocanonical(sp,z))-
        stieltjesintegral(g,tocanonical(sp,Inf))+
        linesum(Fun(w,sp))*log(z-fromcanonical(sp,Inf))
end


function logkernel{LS,RR<:Arc}(sp::Space{LS,RR},w,z)
    g=Fun(w,setcanonicaldomain(sp))*abs(fromcanonicalD(sp))
    logkernel(g,tocanonical(sp,z))-
        logkernel(g,tocanonical(sp,Inf))+
        linesum(Fun(w,sp))*log(abs(z-fromcanonical(sp,Inf)))/π
end


function SingularIntegral{JW,RR<:Arc}(S::JacobiWeight{JW,RR},k::Integer)
    @assert k==0
    tol=1E-15
    # the mapped logkernel
    d=domain(S)
    csp=setcanonicaldomain(S)
    Σ=SingularIntegral(csp,0)
    M=Multiplication(abs(fromcanonicalD(d,Fun(identity,csp))),csp)

    z∞=tocanonical(d,Inf)
    cnst=Array(Float64,0)
    for k=1:10000
        push!(cnst,logkernel(Fun([zeros(k-1);1.],csp),z∞))
        if k≥3&&norm(cnst[end-2:end])<tol
            break
        end
    end
    L∞=FiniteFunctional(cnst,csp)

    x=Fun(identity,S)
    SpaceOperator((Σ-L∞)*M,S,setdomain(rangespace(Σ),d))+(log(abs(x-fromcanonical(d,Inf)))/π)*DefiniteLineIntegral(S)
end
