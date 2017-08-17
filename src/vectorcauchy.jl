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
        $op(v::Vector{F},z) where {F<:Fun} = mapreduce(f->$op(f,z),+,v)
        $op(v::Vector{F}) where {F<:Fun} = map($op,v)
        $op(v::Vector{Any},z) = mapreduce(f->$op(f,z),+,v)
        $op(S::PiecewiseSpace,v,z) = $op(components(Fun(S,v)),z)
        $op(f::Fun{S,T}) where {S<:PiecewiseSpace,T} =
            (v = $op(components(f)); Fun(ApproxFun.SumSpace(map(space,v)),vec(coefficientmatrix(v).')))
        $op(S::PiecewiseSpace,v) = depiece($op(components(Fun(S,v))))

        # directed is usually analytic continuation, so we need to unwrap
        # directd
        $op(v::Vector{F},z::Directed) where {F<:Fun} = mapreduce(f->(z in domain(f))?$op(f,z):$op(f,z.x),+,v)
        $op(v::Vector{Any},z::Directed) = mapreduce(f->(z in domain(f))?$op(f,z):$op(f,z.x),+,v)
    end
end

hilbert(v::Vector{F},x) where {F<:Union{Fun,Any}} =
    mapreduce(f->(x in domain(f))?hilbert(f,x):-stieltjes(f,x)/π,+,v)
hilbert(S::PiecewiseSpace,v,z) = hilbert(components(Fun(S,v)),z)



function cauchy(v::Fun{S,T},z::Number) where {S<:ArraySpace,T}
    m = Array(v)
    Complex{Float64}[cauchy(m[k,j],z) for k=1:size(m,1),j=1:size(m,2)]
end
