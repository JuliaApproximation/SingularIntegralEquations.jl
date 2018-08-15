const NMAX = 100

for TYP in subtypes(AbstractFloat)
    A = Meta.parse("A"*string(TYP))
    @eval begin
        const $A = Array{Complex{$TYP}}(undef,NMAX,NMAX)
        riemann_array(::Type{$TYP}) = $A
    end
end

lhelm_riemanncomplex(z::Complex{T},z0::Complex{T},ζ::Complex{T},ζ0::Complex{T},E::T) where {T<:AbstractFloat} = lhelm_riemanncomplex(z,z0,ζ,ζ0,E,riemann_array(T))

function lhelm_riemanncomplex(z::Complex{T},z0::Complex{T},ζ::Complex{T},ζ0::Complex{T},E::T,A::Array{Complex{T}}) where T<:AbstractFloat
    cst = -(E/4+(z0-ζ0)/(8im))
    executerecurrence!(A,cst)

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
    1+ret
end

lhelm_riemann(x::AbstractFloat,x0::AbstractFloat,y::AbstractFloat,y0::AbstractFloat,E::AbstractFloat) = lhelm_riemanncomplex(complex(x,y),complex(x0,y0),complex(x,-y),complex(x0,-y0),E)
lhelm_riemann(x::Union{T1,Complex{T1}},y::Union{T2,Complex{T2}},E::AbstractFloat) where {T1<:AbstractFloat,T2<:AbstractFloat} = lhelm_riemann(real(x),real(y),imag(x),imag(y),E)

lhelm_riemann(x::Vector,y::Vector,E::AbstractFloat) = promote_type(eltype(x),eltype(y),typeof(E))[lhelm_riemann(x[k],y[k],E) for k in 1:length(x)]
lhelm_riemann(x::Matrix,y::Matrix,E::AbstractFloat) = reshape(lhelm_riemann(vec(x),vec(y),E),size(x))


function executerecurrence!(A::Array{Complex{T}},cst::Complex{T}) where T<:AbstractFloat
    eighthim = one(T)/(8im)
    sixteenthim = eighthim/2

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
