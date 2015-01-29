
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




cauchy{F<:Fun}(v::Vector{F},z)=mapreduce(f->cauchy(f,z),+,v)
cauchy(v::Vector{Any},z)=mapreduce(f->cauchy(f,z),+,v)
cauchy{P<:PiecewiseSpace,T}(v::Fun{P,T},z::Number)=cauchy(vec(v),z)


function cauchy{S<:ArraySpace,T}(v::Fun{S,T},z::Number)
    m=mat(v)
    Complex{Float64}[cauchy(m[k,j],z) for k=1:size(m,1),j=1:size(m,2)]
end



function cauchy{S<:ArraySpace,T}(s,v::Fun{S,T},z::Number)
    m=mat(v)
    Complex{Float64}[cauchy(s,m[k,j],z) for k=1:size(m,1),j=1:size(m,2)]
end