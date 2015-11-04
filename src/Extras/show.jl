## CauchyWeight

function Base.show(io::IO,s::CauchyWeight)
    #TODO: Get shift and weights right
    if order(s) == 0
        print(io,"π⁻¹log|y-x|[")
    else
        print(io,"π⁻¹|y-x|^⁻$(O)[")
    end
    show(io,s.space)
    print(io,"]")
end

## GreensFun

function Base.show(io::IO,G::GreensFun)
    print(io,"GreensFun with kernels:\n{")
    Kernels = kernels(G)
    for i in 1:length(G)
        print(io,"\n  ")
        show(io,Kernels[i])
    end
    print(io,"\n}")
end
