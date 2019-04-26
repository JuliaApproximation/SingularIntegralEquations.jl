# I think it makes more sense to let the array into the function.
# That way the coefficient conversions happen once.
#=
function cauchy{S,T}(f::Fun{S,T},z::Array)
    ret=Array(Complex{Float64},size(z)...)
    for k=1:size(z,1),j=1:size(z,2)
        @inbounds ret[k,j]=cauchy(f,z[k,j])
    end
    ret
end
=#

for op in (:(stieltjes),:(cauchy),:(logkernel),:(stieltjesintegral),:(cauchyintegral))
    @eval begin
        $op(S::PiecewiseSpace,v,z) = mapreduce(f->$op(f,z),+,components(Fun(S,v)))
        function $op(f::Fun{S,T}) where {S<:PiecewiseSpace,T}
            v = map($op,components(f))
            Fun(SumSpace(map(space,v)),vec(permutedims(coefficientmatrix(v))))
        end
        $op(S::PiecewiseSpace,v) = depiece($op(components(Fun(S,v))))

        # directed is usually analytic continuation, so we need to unwrap
        # directd
        $op(S::PiecewiseSpace,v,z::Directed) =
            mapreduce(f-> (z ∈ domain(f)) ? $op(f,z) : $op(f,z.x),+,components(Fun(S,v)))
    end
end

hilbert(v::Vector{F},x) where {F<:Union{Fun,Any}} =

hilbert(S::PiecewiseSpace,v,z) = mapreduce(f->(x in domain(f)) ? hilbert(f,x) : -stieltjes(f,x)/π,
                                           +, components(Fun(S,v)))




stieltjes(S::ArraySpace,v,z::Number) = stieltjes.(Array(Fun(S,v)),z)
stieltjes(S::ArraySpace,v) = Fun(stieltjes.(Array(Fun(S,v))))
