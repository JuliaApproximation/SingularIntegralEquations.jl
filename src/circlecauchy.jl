

## cauchy

function cauchycircleS(cfs::AbstractVector,z::Number,s::Bool)
    ret=zero(Complex{Float64})

    if s
        zm = one(Complex{Float64})

        #odd coefficients are pos
        @simd for k=1:2:length(cfs)
            @inbounds ret += cfs[k]*zm
            zm *= z
        end
    else
        z=1/z
        zm = z

        #even coefficients are neg
        @simd for k=2:2:length(cfs)
            @inbounds ret -= cfs[k]*zm
            zm *= z
        end
    end

    ret
end


function stieltjes(sp::Laurent{DD,RR},f::AbstractVector,z::Number) where {DD<:Circle,RR}
    d=domain(sp)
    if !d.orientation
        return -stieltjes(reverseorientation(Fun(sp,f)),z)
    end

    z=mappoint(d,Circle(),z)
    -2π*im*cauchycircleS(f,z,abs(z) < 1)
end

function stieltjes(sp::Laurent{DD,RR},f::AbstractVector,z::Directed{s}) where {DD<:Circle,RR,s}
    d=domain(sp)
    if !d.orientation
        return -stieltjes(reverseorientation(Fun(sp,f)),reverseorientation(z))
    end

    z=mappoint(d,Circle(),z)
    -2π*im*cauchycircleS(f,undirected(z),orientation(z))
end

stieltjes(sp::Laurent{DD,RR},f,z::Vector) where {DD<:Circle,RR} = [stieltjes(sp,f,zk) for zk in z]
stieltjes(sp::Laurent{DD,RR},f,z::Matrix) where {DD<:Circle,RR} = reshape(stieltjes(sp,f,vec(z)),size(z,1),size(z,2))




stieltjes(sp::Fourier{DD,RR},f,z) where {DD<:Circle,RR} = stieltjes(Laurent(domain(sp)),coefficients(f,sp,Laurent(domain(sp))),z)



# we implement cauchy ±1 as canonical
# TODO: reimplement directly
hilbert(sp::Laurent{DD,RR},f,z) where {DD<:Circle,RR} = (stieltjes(sp,f,(z)⁺)+stieltjes(sp,f,(z)⁻))/(-2π)






## stieltjesintegral and logkernel


function stieltjesintegral(sp::Laurent{DD,RR},f,z::Number) where {DD<:Circle,RR}
    d=domain(sp)
    @assert d==Circle()  #TODO: radius
    ζ=Fun(d)
    r=stieltjes(integrate(f-f[2]/ζ),z)
    abs(z)<1 ? r : r+2π*im*f[2]*log(z)
end


stieltjesintegral(sp::Fourier{DD,RR},f,z::Number) where {DD<:Circle,RR}=stieltjesintegral(Fun(Fun(sp,f),Laurent),z)

function logkernel(sp::Fourier{DD,RR},g,z::Number) where {DD<:Circle,RR}
    d=domain(sp)
    c,r=d.center,d.radius
    z=z-c
    if abs(z) ≤r
        ret=2r*log(r)*g[1]
        for j=2:2:length(g)
            k=div(j,2)
            ret+=-g[j]*sin(k*angle(z))*abs(z)^k/(k*r^(k-1))
        end
        for j=3:2:length(g)
            k=div(j,2)
            ret+=-g[j]*cos(k*angle(z))*abs(z)^k/(k*r^(k-1))
        end
        ret
    else
        ret=2r*logabs(z)*g[1]
        for j=2:2:length(g)
            k=div(j,2)
            ret+=-g[j]*sin(k*angle(z))*r^(k+1)/(k*abs(z)^k)
        end
        for j=3:2:length(g)
            k=div(j,2)
            ret+=-g[j]*cos(k*angle(z))*r^(k+1)/(k*abs(z)^k)
        end
        ret
    end
end
logkernel(sp::Fourier{DD,RR},g,z::Vector) where {DD<:Circle,RR} =
    promote_type(eltype(g),eltype(z))[logkernel(sp,g,zk) for zk in z]
logkernel(sp::Fourier{DD,RR},g,z::Matrix) where {DD<:Circle,RR} =
    reshape(promote_type(eltype(g),eltype(z))[logkernel(sp,g,zk) for zk in z],size(z))

logkernel(sp::Laurent{DD,RR},g,z) where {DD<:Circle,RR} = logkernel(Fun(Fun(sp,g),Fourier),z)
