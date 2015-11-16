

## cauchy

function cauchycircleS(cfs::Vector,z::Number,s::Bool)
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


function stieltjes{DD<:Circle}(f::Fun{Laurent{DD}},z,s::Bool)
    @assert in(z,d)
    -2π*im*cauchycircleS(cfs,mappoint(d,Circle(),z),s)
end
function stieltjes{DD<:Circle}(f::Fun{Laurent{DD}},z::Number)
    z=mappoint(domain(f),Circle(),z)
    -2π*im*cauchycircleS(coefficients(f),z,abs(z) < 1)
end

stieltjes{DD<:Circle}(f::Fun{Laurent{DD}},z::Vector)=[stieltjes(f,zk) for zk in z]
stieltjes{DD<:Circle}(f::Fun{Laurent{DD}},z::Matrix)=reshape(stieltjes(f,vec(z)),size(z,1),size(z,2))




stieltjes{DD<:Circle}(f::Fun{Fourier{DD}},z,s...)=stieltjes(Fun(f,Laurent(domain(f))),z,s...)



# we implement cauchy ±1 as canonical
hilbert{DD<:Circle}(f::Fun{Laurent{DD}},z)=(stieltjes(f,z,true)+stieltjes(f,z,false))/(-2π)






## stieltjesintegral and logkernel


function stieltjesintegral{DD<:Circle}(f::Fun{Laurent{DD}},z::Number,s...)
    d=domain(f)
    @assert d==Circle()  #TODO: radius
    ζ=Fun(d)
    r=stieltjes(integrate(f-f.coefficients[2]/ζ),z,s...)
    abs(z)<1?r:r+2π*im*f.coefficients[2]*log(z)
end


stieltjesintegral{DD<:Circle}(f::Fun{Fourier{DD}},z::Number,s...)=stieltjesintegral(Fun(f,Laurent),z,s...)

function logkernel{DD<:Circle}(g::Fun{Fourier{DD}},z::Number)
    d=domain(g)
    c,r=d.center,d.radius
    z=z-c
    if abs(z) ≤r
        ret=2r*log(r)*g.coefficients[1]
        for j=2:2:length(g)
            k=div(j,2)
            ret+=-g.coefficients[j]*sin(k*angle(z))*abs(z)^k/(k*r^(k-1))
        end
        for j=3:2:length(g)
            k=div(j,2)
            ret+=-g.coefficients[j]*cos(k*angle(z))*abs(z)^k/(k*r^(k-1))
        end
        ret
    else
        ret=2r*logabs(z)*g.coefficients[1]
        for j=2:2:length(g)
            k=div(j,2)
            ret+=-g.coefficients[j]*sin(k*angle(z))*r^(k+1)/(k*abs(z)^k)
        end
        for j=3:2:length(g)
            k=div(j,2)
            ret+=-g.coefficients[j]*cos(k*angle(z))*r^(k+1)/(k*abs(z)^k)
        end
        ret
    end
end
logkernel{DD<:Circle}(g::Fun{Fourier{DD}},z::Vector) = promote_type(eltype(g),eltype(z))[logkernel(g,zk) for zk in z]
logkernel{DD<:Circle}(g::Fun{Fourier{DD}},z::Matrix) = reshape(promote_type(eltype(g),eltype(z))[logkernel(g,zk) for zk in z],size(z))

logkernel{DD<:Circle}(g::Fun{Laurent{DD}},z)=logkernel(Fun(g,Fourier),z)
