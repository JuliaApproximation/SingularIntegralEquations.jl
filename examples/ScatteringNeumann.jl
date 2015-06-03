# This file calculates scattering from the Helmholtz equation via the adaptive spectral method.
# Δu + k^2u = 0,
# ∂u∂n|Γ = 0,
# ∂u^i/∂n = -im k d[2] e^{im k x⋅d},
# u = u^i + u^s.
# The normal derivative ∂u/∂n of the entire wave is calculated on the sound-hard boundaries.
# The scattered wave is calculated by convolving [u] with the fundamental solution.
# Then, the total wave is obtained by summing the incident and the scattered waves.

using ApproxFun,SIE
include("Scatteraux.jl")

k = 25.
ω = 2π
d = (1,-1)
d = d[1]/hypot(d[1],d[2]),d[2]/hypot(d[1],d[2])
ui(x,y) = exp(im*k*(d⋅(x,y)))

# The Helmholtz Green's function, split into singular and nonsingular pieces.
g1(x,y) = besselj0(k*abs(y-x))/2
g2(x,y) = x == y ? -k^2/4 : -k*besselj1(k*abs(y-x))./abs(y-x)/2
g3(x,y) = x == y ? -(log(k/2)+γ)*k^2/4π + k^2/4π + im*k^2/8 : im*k/4*hankelh1(1,k*abs(y-x))./abs(y-x) - g1(x,y)./(y-x).^2/π -g2(x,y).*logabs(y-x)/π
g4(x,y) = im*k/4*hankelh1(1,k*abs(y-x))./abs(y-x).*imag(y-x)

    dom = Interval()
    sp = Space(dom)
    cwsp,cwsp2 = CauchyWeight(sp⊗sp,0),CauchyWeight(sp⊗sp,2)
    ∂ui∂nΓ,⨍ = Fun(t->-im*k*d[2]*ui(real(t),imag(t)),sp),DefiniteLineIntegral(dom)

    @time G = GreensFun(g1,cwsp2;method=:Cholesky) + GreensFun(g2,cwsp;method=:Cholesky) + GreensFun(g3,sp⊗sp;method=:Cholesky)

    @time u = ⨍[G]\-∂ui∂nΓ
    println("The length of u is: ",length(u))
#    us(x,y) = -logkernel(g1,∂u∂n,complex(x,y))-linesum(g2,∂u∂n,complex(x,y))
    us(x,y) = linesum(g4,u,complex(x,y))
