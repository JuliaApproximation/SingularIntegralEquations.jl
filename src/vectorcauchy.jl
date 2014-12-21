
cauchy{S,T}(f::Fun{S,T},z::Vector)=Complex{Float64}[cauchy(f,zk) for zk in z]
cauchy{S,T}(s::Integer,f::Fun{S,T},z::Vector)=Complex{Float64}[cauchy(s,f,zk) for zk in z]
cauchy{S,T}(s,f::Fun{S,T},z::Vector)=Complex{Float64}[cauchy(s,f,zk) for zk in z]




cauchy{F<:Fun}(v::Vector{F},z)=mapreduce(f->cauchy(f,z),+,v)
cauchy{P<:PiecewiseSpace,T}(v::Fun{P,T},z::Number)=cauchy(vec(v),z)


## Operator


function Hilbert(S::PiecewiseSpace,k::Integer)
    @assert k==1
    sp=vec(S)
    C=BandedOperator[k==j?Hilbert(sp[k]):2im*Cauchy(sp[k],sp[j]) for j=1:length(sp),k=1:length(sp)]
    HilbertWrapper(interlace(C))
end

cauchy{S<:ArraySpace,T}(v::Fun{S,T},z::Number)=Complex{Float64}[cauchy(fk,z) for fk in mat(v)]