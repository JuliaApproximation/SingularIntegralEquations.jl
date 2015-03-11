# This file calculates scattering from the Helmholtz equation via the adaptive spectral method.
# Δu + k^2u = 0,
# u|Γ = 0,
# u^i = e^{im k x⋅d},
# u = u^i + u^s.
# The normal derivative ∂u∂n of the entire wave is calculated on the sound-hard line [-1,1].
# The scattered wave is calculated by convolving ∂u∂n with the fundamental solution.
# Then, the total wave is obtained by summing the incident and the scattered waves.

using ApproxFun,SIE
include("Scatteraux.jl")

k = 50.
ω = 2π
d = (1,-1)
d = d[1]/hypot(d[1],d[2]),d[2]/hypot(d[1],d[2])
ui(x,y) = exp(im*k*(d⋅(x,y)))


    dom = Circle(0.0,1.0)
    sp = Fourier(dom)
    spP = Fourier(PeriodicInterval())
    xid = Fun(identity,sp)
    uiΓ,H0,⨍ = Fun(t->ui(real(xid[t]),imag(xid[t])),sp),SingularIntegral(sp,0),DefiniteLineIntegral(sp)

    f1(θ) = -besselj0(k*abs(2sin(θ/2)))/2π

    g1 = Fun(f1,spP)
    G1 = ProductFun(g1,spP,spP;method=:convolution)

    G1C = ProductFun(coefficients(G1),sp,sp)

    f2(θ) = θ == 0 ? -(log(k/2)+γ)/2/π^2 + 1/4π*im : (besselj0(k*abs(2sin(θ/2)))*(im*π/2+log(abs(2sin(θ/2))))/2π - bessely0(k*abs(2sin(θ/2)))/4)/π

    g2 = Fun(f2,spP)
    G2 = ProductFun(g2,spP,spP;method=:convolution)

    G2C = ProductFun(coefficients(G2),sp,sp)

    L,f = H0[G1C]+⨍[G2C],uiΓ

    @time ∂u∂n = L\f
    println("The length of ∂u∂n is: ",length(∂u∂n))

    g3(x,y) = im/4π*hankelh1(0,k*abs(y-x))

function us(x,y)
    ret = linesum(Fun(t->-g3(x-real(xid[t]),im*(y-imag(xid[t])))*∂u∂n[t],sp,length(∂u∂n)))
end
@vectorize_2arg Number us
