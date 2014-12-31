
cauchy{S,T}(f::Fun{S,T},z::Vector)=Complex{Float64}[cauchy(f,zk) for zk in z]
cauchy{S,T}(s::Integer,f::Fun{S,T},z::Vector)=Complex{Float64}[cauchy(s,f,zk) for zk in z]
cauchy{S,T}(s,f::Fun{S,T},z::Vector)=Complex{Float64}[cauchy(s,f,zk) for zk in z]




cauchy{F<:Fun}(v::Vector{F},z)=mapreduce(f->cauchy(f,z),+,v)
cauchy{P<:PiecewiseSpace,T}(v::Fun{P,T},z::Number)=cauchy(vec(v),z)


## Operator


function Hilbert(S::PiecewiseSpace,k::Integer)
    @assert k==1
    sp=vec(S)
    C=BandedOperator[k==j?Hilbert(sp[k]):2im*Cauchy(sp[k],sp[j].space) for j=1:length(sp),k=1:length(sp)]
    HilbertWrapper(interlace(C))
end

function cauchy{S<:ArraySpace,T}(v::Fun{S,T},z::Number)
    m=mat(v)
    Complex{Float64}[cauchy(m[k,j],z) for k=1:size(m,1),j=1:size(m,2)]
end
