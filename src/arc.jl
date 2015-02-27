


function cauchy{LS,RR<:Arc,TT,DS,T}(f::Fun{MappedSpace{LS,RR,TT,DS},T},z::Number)
    g=Fun(f.coefficients,space(f).space)
    cauchy(g,tocanonical(f,z))-cauchy(g,tocanonical(f,Inf))
end

function cauchy{LS,RR<:Arc,TT,DS,T}(s,f::Fun{MappedSpace{LS,RR,TT,DS},T},z::Number)
    g=Fun(f.coefficients,space(f).space)
    cauchy(s,g,tocanonical(f,z))-cauchy(g,tocanonical(f,Inf))
end

function hilbert{LS,RR<:Arc,TT,DS,T}(f::Fun{MappedSpace{LS,RR,TT,DS},T},z::Number)
    g=Fun(f.coefficients,space(f).space)
    hilbert(g,tocanonical(f,z))+(1/π)*stieltjes(g,tocanonical(f,Inf))
end



function Hilbert{LS,RR<:Arc,TT,DS}(sp::MappedSpace{LS,RR,TT,DS},k::Integer)
    @assert k==1
    csp=sp.space
    H=Hilbert(csp)+(1/π)*Stieltjes(csp,tocanonical(sp,Inf))
    HilbertWrapper(SpaceOperator(H,MappedSpace(domain(sp),domainspace(H)),MappedSpace(domain(sp),rangespace(H))),k)
end