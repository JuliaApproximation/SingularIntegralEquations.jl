#
# This function calculates Φ(x,y,E), the fundamental solution of the Helmholtz equation in a linearly stratified medium:
#
# -(Δ + E + x_2)Φ(x,y,E) = δ(x-y)
#
# also known as the gravity Helmholtz equation.
#
# This is a Julia wrapper for the C-library described in:
#
# A. H. Barnett, B. J. Nelson, and J. M. Mahoney, High-order boundary integral equation solution of high frequency wave scattering from obstacles in an unbounded linearly stratified medium, J. Comp. Phys., 297:407–426, 2015.
#
#
# Below:
#
# x = trg is the target variable,
# y = src is the source variable,
# E is the energy, and
# derivs allows for calculation of partial derivatives as well.
#

const lhelmfspath = joinpath(Pkg.dir("SIE"), "deps", "liblhelmfs")

export lhelmfs

function lhelmfs(trg::Union(Vector{Float64},Vector{Complex{Float64}}),energies::Vector{Float64};derivs::Bool=false)
    trgn,energiesn = length(trg),length(energies)
    @assert trgn == energiesn
    n = trgn

    meth = 1
    stdquad = 400
    h = 0.25
    gamout = 0
    nquad = zeros(Int64,1)

    x1,x2 = real(trg),imag(trg)
    u = zeros(Complex{Float64},n)
    #if derivs
        ux = zeros(Complex{Float64},n)
        uy = zeros(Complex{Float64},n)
    #end

    ccall((:lhfs,lhelmfspath),Void,(Ptr{Float64},Ptr{Float64},Ptr{Float64},Int64,Int64,Ptr{Complex{Float64}},Ptr{Complex{Float64}},Ptr{Complex{Float64}},Int64,Float64,Int64,Int64,Ptr{Int64}),x1,x2,energies,derivs ? 1 : 0,n,u,ux,uy,stdquad,h,meth,gamout,nquad)

    if derivs
        return u/4π,ux/4π,uy/4π
    else
        return u/4π
    end
end

function lhelmfs(trg::Union(Matrix{Float64},Matrix{Complex{Float64}}),E::Matrix{Float64};derivs::Bool=false)
    sizetrg,sizeE = size(trg),size(E)
    @assert sizetrg == sizeE

    if derivs
        u,ux,uy = lhelmfs(vec(trg),vec(E);derivs=derivs)
        return reshape(u,sizetrg),reshape(ux,sizetrg),reshape(uy,sizetrg)
    else
        u = lhelmfs(vec(trg),vec(E);derivs=derivs)
        return reshape(u,sizetrg)
    end
end

lhelmfs(trg::Union(VecOrMat{Float64},VecOrMat{Complex{Float64}}),E::Float64;derivs::Bool=false) = lhelmfs(trg,fill(E,size(trg));derivs=derivs)

function lhelmfs(trg::Union(Float64,Complex{Float64}),E::Float64;derivs::Bool=false)
    if derivs
        u,ux,uy = lhelmfs([trg],[E];derivs=derivs)
        return u[1],ux[1],uy[1]
    else
        u = lhelmfs([trg],[E];derivs=derivs)
        return u[1]
    end
end

lhelmfs{T1<:Union(Float64,Complex{Float64}),T2<:Union(Float64,Complex{Float64})}(trg::Union(T1,VecOrMat{T1}),src::Union(T2,VecOrMat{T2}),E::Float64;derivs::Bool=false) = lhelmfs(trg-src,E+imag(src);derivs=derivs)


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
