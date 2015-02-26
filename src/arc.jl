


function cauchy{LS,RR<:Arc,TT,DS,T}(f::Fun{MappedSpace{LS,RR,TT,DS},T},z::Number)
    g=Fun(f.coefficients,space(f).space)
    cauchy(g,tocanonical(f,z))-cauchy(g,tocanonical(f,Inf))
end
