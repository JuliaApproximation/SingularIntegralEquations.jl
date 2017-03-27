# This file calculates the solution to Laplace's equation via the adaptive spectral method.
# Δu = 0,
# u|Γ = u_0,
# u^i = log|z-z_s|,
# u = u^i + u^s.
# The normal derivative ∂u/∂n of the solution is calculated on the constant-charge boundary Γ.
# The reflected solution is calculated by convolving ∂u/∂n with the fundamental solution.
# Then, the total solution is obtained by summing the incident and the reflected solutions.

using ApproxFun, SingularIntegralEquations

z_s = 2.0
ui = (x,y) -> logabs(complex(x,y)-z_s)
g1 = (x,y) -> 1/2

# Set the domains.
N = 10
r = 1e-1
cr = exp.(im*2π*(0:N-1)/N)
crl = (1-2im*r)cr
crr = (1+2im*r)cr
dom = ∪(Segment.(crl,crr)) # All infinitesimal plates
#dom = ∪(Circle,cr,ones(length(cr))r) # All wires
#dom = ∪(Segment.(crl[1:2:end],crr[1:2:end])) ∪ ∪(Circle,cr[2:2:end],ones(length(cr[2:2:end]))r) # Interlaced wires and plates

sp = Space(dom)
cwsp = CauchyWeight(sp⊗sp,0)
uiΓ,⨍ = Fun(t->ui(real(t),imag(t))+0im,sp),DefiniteLineIntegral(dom)

@time G = GreensFun(g1,cwsp;method=:Cholesky)

@time φ0,∂u∂n=[0 ⨍;1 ⨍[G]]\[0.;uiΓ]

println("The length of ∂u∂n is: ",ncoefficients(∂u∂n))

us = (x,y) -> -logkernel(∂u∂n,complex(x,y))/2
ut = (x,y) -> ui(x,y) + us(x,y)
println("This is the approximate gradient: ",((ut(1e-5,0.)-ut(-1e-5,0.))/2e-5))
