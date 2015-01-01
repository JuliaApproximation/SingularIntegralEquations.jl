
cauchy{S,T}(f::Fun{S,T},z::Vector)=Complex{Float64}[cauchy(f,zk) for zk in z]
cauchy{S,T}(s::Int,f::Fun{S,T},z::Vector)=Complex{Float64}[cauchy(s,f,zk) for zk in z]
cauchy{S,T}(s,f::Fun{S,T},z::Vector)=Complex{Float64}[cauchy(s,f,zk) for zk in z]




cauchy{F<:Fun}(v::Vector{F},z)=mapreduce(f->cauchy(f,z),+,v)
cauchy{P<:PiecewiseSpace,T}(v::Fun{P,T},z::Number)=cauchy(vec(v),z)


function cauchy{S<:ArraySpace,T}(v::Fun{S,T},z::Number)
    m=mat(v)
    Complex{Float64}[cauchy(m[k,j],z) for k=1:size(m,1),j=1:size(m,2)]
end



function cauchy{S<:ArraySpace,T}(s,v::Fun{S,T},z::Number)
    m=mat(v)
    Complex{Float64}[cauchy(s,m[k,j],z) for k=1:size(m,1),j=1:size(m,2)]
end