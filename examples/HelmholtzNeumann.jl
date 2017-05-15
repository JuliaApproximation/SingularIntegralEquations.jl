# This file calculates scattering from the Helmholtz equation via the adaptive spectral method.
# Δu + k^2u = 0,
# ∂u∂n|Γ = 0,
# ∂u^i/∂n = -im k d[2] e^{im k x⋅d},
# u = u^i + u^s.
# The normal derivative ∂u/∂n of the entire wave is calculated on the sound-hard boundaries.
# The scattered wave is calculated by convolving [u] with the fundamental solution.
# Then, the total wave is obtained by summing the incident and the scattered waves.

using ApproxFun, SingularIntegralEquations
include("Scatteraux.jl")

k = 50.
ω = 2π
d = (1,-1)
d = d[1]/hypot(d[1],d[2]),d[2]/hypot(d[1],d[2])
ui = (x,y) -> exp(im*k*(d[1]*x+d[2]*y))

# The Helmholtz Green's function, split into singular and nonsingular pieces.
g1 = (x,y) ->  besselj0(k*abs(y-x))/2
g2 = (x,y) ->  x == y ? -k^2/4 : -k*besselj1(k*abs(y-x))./abs(y-x)/2
g3 = (x,y) ->  g3neumann(x,y,k) # In /Scatteraux.jl
g4old = (x,y) ->  im*k/4*hankelh1(1,k*abs(y-x))./abs(y-x).*imag(y-x)
g4 = (x,y) ->  im*k/4*besselj1(k*abs(y-x))./abs(y-x).*imag(y-x)  # For linesum
g5 = (x,y) ->  -k/2*besselj1(k*abs(y-x))./abs(y-x).*imag(y-x)  # For logkernel
g6 = (x,y) ->  k/2*abs(y-x).*(bessely1(k*abs(y-x)) - 2besselj1(k*abs(y-x)).*logabs(y-x)/π) # For Re{Cauchy}


dom = Segment(-2,-1) ∪ Segment(1,2)
sp = Space(dom)
cwsp,cwsp2 = CauchyWeight(sp⊗sp,0),CauchyWeight(sp⊗sp,2)
∂ui∂nΓ,⨍ = Fun(t->-im*k*d[2]*ui(real(t),imag(t)),sp),
                DefiniteLineIntegral(PiecewiseSpace(map(d->JacobiWeight(.5,.5,Ultraspherical(1,d)),dom.domains)))

@time G = GreensFun(g1,cwsp2;method=:Cholesky) + GreensFun(g2,cwsp;method=:Cholesky) + GreensFun(g3,sp⊗sp;method=:Cholesky)
@time u = ⨍[G]\-∂ui∂nΓ
println("The length of u is: ",ncoefficients(u))
us = (x,y) ->  linesum(g4,u,complex(x,y))+logkernel(g5,u,complex(x,y))+π*real(cauchy(g6,real(u),complex(x,y)))+π*im*real(cauchy(g6,imag(u),complex(x,y)))
#dom += 0im
