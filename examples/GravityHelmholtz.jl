# This file calculates scattering from the gravity Helmholtz equation via the adaptive spectral method.
# Δu + (E+x_2)u = 0,
# u|Γ = 0,
# u^i = Φ(⋅,⋅,E)|Γ
# u = u^i + u^s.
# The normal derivative ∂u/∂n of the entire wave is calculated on the sound-soft boundaries.
# The scattered wave is calculated by convolving ∂u/∂n with the fundamental solution.
# Then, the total wave is obtained by summing the incident and the scattered waves.

using ApproxFun, SingularIntegralEquations
include("Scatteraux.jl")

E = 20.
ω = 2π
ui = (x,y) ->  lhelmfs(complex(x,y),-5.0im,E)

# The gravity Helmholtz Green's function.
g3 = (x,y) ->  lhelmfs(x,y,E)
r = (x,y) ->  lhelm_riemann(x,y,E)


dom = ∪(Segment.([-10.0-3.0im,5.0,-2+5im],[-5.0+0.0im,10.0-3im,2+5im]))
sp = Space(dom)
cwsp = CauchyWeight(sp⊗sp,0)
uiΓ,⨍ = Fun(t->ui(real(t),imag(t)),sp),DefiniteLineIntegral(dom)

@time G = GreensFun(g3,cwsp;method=:unsplit)

@time ∂u∂n = ⨍[G]\uiΓ
println("The length of ∂u∂n is: ",ncoefficients(∂u∂n))
us = (x,y) ->  -linesum(g3,∂u∂n,complex(x,y))
