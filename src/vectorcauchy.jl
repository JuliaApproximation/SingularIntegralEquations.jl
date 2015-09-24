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
function cauchy{S,T}(s::Int,f::Fun{S,T},z::Array)
    ret=Array(Complex{Float64},size(z)...)
    for k=1:size(z,1),j=1:size(z,2)
        @inbounds ret[k,j]=cauchy(s,f,z[k,j])
    end
    ret
end
function cauchy{S,T}(s,f::Fun{S,T},z::Array)
    ret=Array(Complex{Float64},size(z)...)
    for k=1:size(z,1),j=1:size(z,2)
        @inbounds ret[k,j]=cauchy(s,f,z[k,j])
    end
    ret
end
=#

for op in (:(stieltjes),:(cauchy),:(logkernel),:(stieltjesintegral),:(cauchyintegral))
    @eval begin
        $op{F<:Fun}(v::Vector{F},z)=mapreduce(f->$op(f,z),+,v)
        $op(v::Vector{Any},z)=mapreduce(f->$op(f,z),+,v)
        $op{P<:PiecewiseSpace,T}(v::Fun{P,T},z)=$op(vec(v),z)

        $op{F<:Fun}(s::Bool,v::Vector{F},z)=mapreduce(f->(z in domain(f))?$op(s,f,z):$op(f,z),+,v)
        $op(s::Bool,v::Vector{Any},z)=mapreduce(f->(z in domain(f))?$op(s,f,z):$op(f,z),+,v)
        $op{P<:PiecewiseSpace,T}(s::Bool,v::Fun{P,T},z)=$op(s,pieces(v),z)
    end
end

hilbert{F<:Union{Fun,Any}}(v::Vector{F},x)=mapreduce(f->(x in domain(f))?hilbert(f,x):-stieltjes(f,x)/π,+,v)
hilbert{P<:PiecewiseSpace,T}(v::Fun{P,T},z)=hilbert(pieces(v),z)



function cauchy{S<:ArraySpace,T}(v::Fun{S,T},z::Number)
    m=mat(v)
    Complex{Float64}[cauchy(m[k,j],z) for k=1:size(m,1),j=1:size(m,2)]
end



function cauchy{S<:ArraySpace,T}(s,v::Fun{S,T},z::Number)
    m=mat(v)
    Complex{Float64}[cauchy(s,m[k,j],z) for k=1:size(m,1),j=1:size(m,2)]
end
