

function LogKernelAsymptotics(S::Fourier,k)
    d=domain(S)
    r=d.radius
    if k==-1  # coefficient in front of logabs term
        CompactFunctional([2π*r],S)
    else
        error("Not implemented")
    end
end





function LogKernelAsymptotics(S::JacobiWeight{ChebyshevDirichlet{1,1}},k)
    @assert S.α==S.β==-0.5
    d=domain(f)
    if k==-1  # coefficient in front of logabs term
        CompactFunctional([π*length(d)/2,0,-π*length(d)/2],S)
    else
        error("Not implemented")
    end
end
