function lhelm_riemanncomplex{T<:Number}(z::T,z0::T,ζ::T,ζ0::T,E::Real)
    N = 100
    #A = T<:Complex ? zeros(T,N,N) : zeros(Complex{T},N,N)
    A = zeros(Complex{Float64},N,N)

    cst = -(E/4+(z0-ζ0)/8im)
    A[1,1] = cst
    A[2,1] = -one(T)/16im
    A[1,2] = one(T)/16im
    A[2,2] = cst*A[1,1]/4

    for i=3:N
        A[i,2] = (cst*A[i-1,1] - A[i-2,1]/8im)/(2i)
    end
    for j=3:N
        A[2,j] = (cst*A[1,j-1] + A[1,j-2]/8im)/(2j)
    end

    for j=3:N,i=3:N
        A[i,j] = (cst*A[i-1,j-1] + A[i-1,j-2]/8im - A[i-2,j-1]/8im)/(i*j)
    end

    ret = one(T)
    ζζ0 = ζ-ζ0
    ζζ = one(T)
    for j=1:N
        ζζ *= ζζ0
        zz0 = z-z0
        zz = one(T)
        for i=1:N
            zz *= zz0
            ret += A[i,j]*zz*ζζ#(z-z0)^i*(ζ-ζ0)^j
        end
    end
    ret
end

lhelm_riemann{T<:Number}(x::T,x0::T,y::T,y0::T,E::T) = lhelm_riemanncomplex(complex(x,y),complex(x0,y0),complex(x,-y),complex(x0,-y0),E)
lhelm_riemann(x::Number,y::Number,E::Real) = lhelm_riemann(real(x),real(y),imag(x),imag(y),E)
