

function LogKernelAsymptotics(S::Fourier,k)
    d=domain(S)
    r=d.radius
    if k==-1  # coefficient in front of logabs term
        FiniteOperator([2r].',S,ConstantSpace())
    else
        error("Not implemented")
    end
end

LogKernelAsymptotics(S::Laurent,k)=LogKernelAsymptotics(Fourier(domain(S)),k)*Conversion(S,Fourier(domain(S)))




function LogKernelAsymptotics{CD<:ChebyshevDirichlet{1,1}}(S::JacobiWeight{CD},k)
    @assert S.α==S.β==-0.5
    d=domain(S)
    if k==-1  # coefficient in front of logabs term
        FiniteOperator([arclength(d)/2,0,-arclength(d)/2].',S)
    else
        error("Not implemented")
    end
end


function LogKernelAsymptotics{C<:Chebyshev}(S::JacobiWeight{C},k)
    @assert S.α==S.β==-0.5
    d=domain(S)
    if k==-1  # coefficient in front of logabs term
        FiniteOperator([arclength(d)/2].',S,ConstantSpace())
    else
        error("Not implemented")
    end
end
