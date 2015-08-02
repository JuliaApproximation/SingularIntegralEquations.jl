


function cauchy{LS,RR<:Arc,TT,T}(f::Fun{MappedSpace{LS,RR,TT},T},z::Number)
    g=Fun(f.coefficients,space(f).space)
    cauchy(g,tocanonical(f,z))-cauchy(g,tocanonical(f,Inf))
end

function cauchy{LS,RR<:Arc,TT,T}(s,f::Fun{MappedSpace{LS,RR,TT},T},z::Number)
    g=Fun(f.coefficients,space(f).space)
    cauchy(s,g,tocanonical(f,z))-cauchy(g,tocanonical(f,Inf))
end

function hilbert{LS,RR<:Arc,TT,T}(f::Fun{MappedSpace{LS,RR,TT},T},z::Number)
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
        cauchyintegral(g,tocanonical(w,Inf))-
        sum(w)*(-log(z-fromcanonical(w,Inf)))/(-2π*im)
end


function linesumcauchyintegral{LS,RR<:Arc,TT}(w::Fun{MappedSpace{LS,RR,TT}},z)
    g=Fun(w.coefficients,w.space.space)*abs(ApproxFun.fromcanonicalD(w,Fun()))
    cauchyintegral(g,tocanonical(w,z))-
        cauchyintegral(g,tocanonical(w,Inf))-
        linesum(w)*(-log(z-fromcanonical(w,Inf)))/(-2π*im)
end


function logkernel{LS,RR<:Arc,TT}(w::Fun{MappedSpace{LS,RR,TT}},z)
    g=Fun(w.coefficients,w.space.space)*abs(ApproxFun.fromcanonicalD(w,Fun()))
    logkernel(g,tocanonical(w,z))-
        logkernel(g,tocanonical(w,Inf))-
        linesum(w)*(-log(abs(z-fromcanonical(w,Inf))))/(π)
end




