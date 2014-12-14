
cauchy{S,T}(f::Fun{S,T},z::Vector)=Complex{Float64}[cauchy(f,zk) for zk in z]
cauchy{S,T}(s::Integer,f::Fun{S,T},z::Vector)=Complex{Float64}[cauchy(s,f,zk) for zk in z]
cauchy{S,T}(s,f::Fun{S,T},z::Vector)=Complex{Float64}[cauchy(s,f,zk) for zk in z]




cauchy{F<:Fun}(v::Vector{F},z)=mapreduce(f->cauchy(f,z),+,v)
cauchy{P<:PiecewiseSpace,T}(v::Fun{P,T},z::Number)=cauchy(vec(v),z)