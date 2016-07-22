const superscripts = Dict(0=>"⁰",1=>"¹",2=>"²",3=>"³",4=>"⁴",5=>"⁵",6=>"⁶",7=>"⁷",8=>"⁸",9=>"⁹")

## JacobiQ and JacobiQWeight

function Base.show(io::IO,S::JacobiQ)
    S.a == S.b == 0 ? print(io,"LegendreQ(") : print(io,"JacobiQ($(S.a),$(S.b),")
    show(io,domain(S))
    print(io,")")
end

function Base.show(io::IO,s::JacobiQWeight)
    d=domain(s)
    #TODO: Get shift and weights right
    if s.α==s.β
        print(io,"(x^2-1)^$(s.α)[")
    elseif s.α==0
        print(io,"(x-1)^$(s.β)[")
    elseif s.β==0
        print(io,"(x+1)^$(s.α)[")
    else
        print(io,"(x+1)^$(s.α)*(x-1)^$(s.β)[")
    end

    show(io,s.space)
    print(io,"]")
end

## CauchyWeight

function Base.show(io::IO,s::CauchyWeight)
    #TODO: Get shift and weights right
    if order(s) == 0
        print(io,"π⁻¹log|y-x|[")
    elseif order(s) ≥ 1
        print(io,"π⁻¹|y-x|⁻"*mapreduce(x->superscripts[x],*,reverse!(digits(order(s))))*"[")
    else
        print(io,"π⁻¹|y-x|"*mapreduce(x->superscripts[x],*,reverse!(digits(-order(s))))*"[")
    end
    show(io,s.space)
    print(io,"]")
end

## GreensFun

function Base.show(io::IO,G::GreensFun)
    print(io,"GreensFun with kernels:\n{")
    Kernels = kernels(G)
    for i in 1:length(G)
        print(io,"\n ")
        show(io,Kernels[i])
    end
    print(io,"\n}")
end

## HierarchicalDomain

function Base.show{S,T}(io::IO, H::HierarchicalDomain{S,T})
    print(io,"$(nlevels(H))-level HierarchicalDomain{$S,$T}:\n")
    show(io,UnionDomain(collectdata(H)))
end

## HierarchicalSpace

function Base.show{S,T}(io::IO, H::HierarchicalSpace{S,T})
    print(io,"$(nlevels(H))-level HierarchicalSpace{$S,$T}:\n")
    show(io,PiecewiseSpace(collectdata(H)))
end

## HierarchicalMatrix{F<:GreensFun,G<:GreensFun}
# Base.writemime because HierarchicalMatrix <: AbstractArray

@compat function Base.show{F<:GreensFun,L<:LowRankFun,T}(io::IO, ::MIME"text/plain", H::HierarchicalMatrix{F,GreensFun{L,T}})
    print(io,"$(nlevels(H))-level HierarchicalMatrix of GreensFun's with blockwise ranks:\n")
    show(io,blockrank(H))
end

## HierarchicalOperator{U<:Operator,V<:AbstractLowRankOperator}

@compat function Base.show{U<:Operator,V<:AbstractLowRankOperator}(io::IO, ::MIME"text/plain", H::HierarchicalOperator{U,V})
    print(io,"$(nlevels(H))-level HierarchicalOperator with blockwise ranks:\n")
    A = blockrank(H)
    m,n = size(A)
    print(io,"[")
        for j=1:n-1
            print(io,"$(A[1,j]) ")
        end
        print(io,"$(A[1,n])\n")
    for i=2:m-1
        for j=1:n-1
            print(io," $(A[i,j])")
        end
        print(io," $(A[i,n])\n")
    end
    for j=1:n-1
        print(io," $(A[m,j])")
    end
    print(io," $(A[m,n])]")
end
