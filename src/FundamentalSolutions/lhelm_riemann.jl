const NMAX = 100
const A = Array{Complex{Float64}}(NMAX,NMAX)# = Array{Complex{Float64}}(NMAX*(NMAX+1)-div((NMAX-1)^2,4))
const eighthim = inv(8.0im)
const sixteenthim = inv(16.0im)

function lhelm_riemanncomplex{T<:Complex{Float64}}(z::T,z0::T,ζ::T,ζ0::T,E::Float64)

    cst = -(E/4+(z0-ζ0)*eighthim)
    executerecurrence(cst)

    zz0,ζζ0 = z-z0,ζ-ζ0
    ret = A[min(NMAX,2NMAX),NMAX]
    for i=min(NMAX,2NMAX)-1:-1:div(NMAX+1,2)
        ret = muladd(zz0,ret,A[i,NMAX])
    end
    ret = ret*zz0^div(NMAX+3,2)

    for j=NMAX-1:-1:1
        temp = A[min(NMAX-1,2j),j]
        for i=min(NMAX-1,2j)-1:-1:div(j+1,2)
            temp = muladd(zz0,temp,A[i,j])
        end
        temp = temp*zz0^div(j+3,2)
        ret = muladd(ret,ζζ0,temp)
    end
    one(T)+ret
end

lhelm_riemann{T<:Number}(x::T,x0::T,y::T,y0::T,E::T) = lhelm_riemanncomplex(complex(x,y),complex(x0,y0),complex(x,-y),complex(x0,-y0),E)
lhelm_riemann(x::Number,y::Number,E::Real) = lhelm_riemann(real(x),real(y),imag(x),imag(y),E)

function executerecurrence(cst::Complex{Float64})
    A[1,1] = cst
    A[2,1] = -sixteenthim

    A[1,2] = sixteenthim
    A[2,2] = cst*A[1,1]/4
    A[3,2] = (cst*A[2,1] - eighthim*A[1,1])/6
    A[4,2] = -eighthim*A[2,1]/8

    for j=3:NMAX
        if isodd(j)
            i = div(j+1,2)
            A[i,j] = (cst*A[i-1,j-1] + eighthim*A[i-1,j-2])/(i*j)
        else
            i = div(j+1,2)
            A[i,j] = (eighthim*A[i-1,j-2])/(i*j)
            i+=1
            A[i,j] = (cst*A[i-1,j-1] + eighthim*A[i-1,j-2])/(i*j)
        end
        for i = div(j+4,2):min(2j-3,NMAX)
            A[i,j] = (cst*A[i-1,j-1] + eighthim*A[i-1,j-2] - eighthim*A[i-2,j-1])/(i*j)
        end
        if 2j-2 ≤ NMAX
            i = 2j-2
            A[i,j] = (cst*A[i-1,j-1] - eighthim*A[i-2,j-1])/(i*j)
            if 2j-1 ≤ NMAX
                i = 2j-1
                A[i,j] = (cst*A[i-1,j-1] - eighthim*A[i-2,j-1])/(i*j)
                if 2j ≤ NMAX
                    i = 2j
                    A[i,j] = (-eighthim*A[i-2,j-1])/(i*j)
                end
            end
        end
    end
end
