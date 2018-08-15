

function LogKernelAsymptotics(S::Fourier,k)
    d=domain(S)
    r=d.radius
    if k==-1  # coefficient in front of logabs term
        FiniteOperator([2r]',S,ConstantSpace(typeof(r)))
    else
        error("Not implemented")
    end
end

LogKernelAsymptotics(S::Laurent,k)=LogKernelAsymptotics(Fourier(domain(S)),k)*Conversion(S,Fourier(domain(S)))




function LogKernelAsymptotics(S::JacobiWeight{CD},k) where CD<:ChebyshevDirichlet{1,1}
    @assert S.α==S.β==-0.5
    d=domain(S)
    if k==-1  # coefficient in front of logabs term
        FiniteOperator([arclength(d)/2,0,-arclength(d)/2]',S)
    else
        error("Not implemented")
    end
end


function LogKernelAsymptotics(S::JacobiWeight{C},k) where C<:Chebyshev
    @assert S.α==S.β==-0.5
    d=domain(S)
    if k==-1  # coefficient in front of logabs term
        r = arclength(d)
        FiniteOperator([r/2]',S,ConstantSpace(typeof(r)))
    else
        error("Not implemented")
    end
end
