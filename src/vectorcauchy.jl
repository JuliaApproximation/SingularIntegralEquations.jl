# I think it makes more sense to let the array into the function.
# That way the coefficient conversions happen once.
#=
function cauchy{S,T}(f::Fun{S,T},z::Array,s...)
    ret=Array(Complex{Float64},size(z)...)
    for k=1:size(z,1),j=1:size(z,2)
        @inbounds ret[k,j]=cauchy(f,z[k,j],s...)
    end
    ret
end
=#

for op in (:(stieltjes),:(cauchy),:(logkernel),:(stieltjesintegral),:(cauchyintegral))
    @eval begin
        $op{F<:Fun}(v::Vector{F},z)=mapreduce(f->$op(f,z),+,v)
        $op(v::Vector{Any},z)=mapreduce(f->$op(f,z),+,v)
        $op{P<:PiecewiseSpace,T}(v::Fun{P,T},z)=$op(vec(v),z)

        $op{F<:Fun}(v::Vector{F},z,s::Bool)=mapreduce(f->(z in domain(f))?$op(f,z,s):$op(f,z),+,v)
        $op(v::Vector{Any},z,s::Bool)=mapreduce(f->(z in domain(f))?$op(f,z,s):$op(f,z),+,v)
        $op{P<:PiecewiseSpace,T}(v::Fun{P,T},z,s::Bool)=$op(pieces(v),z,s)
    end
end

hilbert{F<:Union{Fun,Any}}(v::Vector{F},x)=mapreduce(f->(x in domain(f))?hilbert(f,x):-stieltjes(f,x)/π,+,v)
hilbert{P<:PiecewiseSpace,T}(v::Fun{P,T},z)=hilbert(pieces(v),z)



function cauchy{S<:ArraySpace,T}(v::Fun{S,T},z::Number,s...)
    m=mat(v)
    Complex{Float64}[cauchy(m[k,j],z,s...) for k=1:size(m,1),j=1:size(m,2)]
end
