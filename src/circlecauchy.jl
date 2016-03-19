

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
        z=1./z
        zm = z

        #even coefficients are neg
        @simd for k=2:2:length(cfs)
            @inbounds ret -= cfs[k]*zm
            zm *= z
        end
    end

    ret
end


function stieltjes{DD<:Circle}(sp::Laurent{DD},f::AbstractVector,z,s::Bool)
    d=domain(sp)
    if !d.orientation
        return -stieltjes(reverseorientation(Fun(f,sp)),z,!s)
    end
    @assert in(z,d)
    -2π*im*cauchycircleS(f,mappoint(d,Circle(),z),s)
end
function stieltjes{DD<:Circle}(sp::Laurent{DD},f::AbstractVector,z::Number)
    d=domain(sp)
    if !d.orientation
        return -stieltjes(reverseorientation(Fun(f,sp)),z)
    end

    z=mappoint(d,Circle(),z)
    -2π*im*cauchycircleS(f,z,abs(z) < 1)
end

stieltjes{DD<:Circle}(sp::Laurent{DD},f,z::Vector)=[stieltjes(sp,f,zk) for zk in z]
stieltjes{DD<:Circle}(sp::Laurent{DD},f,z::Matrix)=reshape(stieltjes(sp,f,vec(z)),size(z,1),size(z,2))




stieltjes{DD<:Circle}(sp::Fourier{DD},f,z,s...)=stieltjes(Laurent(domain(sp)),coefficients(f,sp,Laurent(domain(sp))),z,s...)



# we implement cauchy ±1 as canonical
hilbert{DD<:Circle}(sp::Laurent{DD},f,z)=(stieltjes(sp,f,z,true)+stieltjes(sp,f,z,false))/(-2π)






## stieltjesintegral and logkernel


function stieltjesintegral{DD<:Circle}(sp::Laurent{DD},f,z::Number,s...)
    d=domain(sp)
    @assert d==Circle()  #TODO: radius
    ζ=Fun(d)
    r=stieltjes(integrate(f-f[2]/ζ),z,s...)
    abs(z)<1?r:r+2π*im*f[2]*log(z)
end


stieltjesintegral{DD<:Circle}(sp::Fourier{DD},f,z::Number,s...)=stieltjesintegral(Fun(Fun(f,sp),Laurent),z,s...)

function logkernel{DD<:Circle}(sp::Fourier{DD},g,z::Number)
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
logkernel{DD<:Circle}(sp::Fourier{DD},g,z::Vector) = promote_type(eltype(g),eltype(z))[logkernel(sp,g,zk) for zk in z]
logkernel{DD<:Circle}(sp::Fourier{DD},g,z::Matrix) = reshape(promote_type(eltype(g),eltype(z))[logkernel(sp,g,zk) for zk in z],size(z))

logkernel{DD<:Circle}(sp::Laurent{DD},g,z)=logkernel(Fun(Fun(g,sp),Fourier),z)
