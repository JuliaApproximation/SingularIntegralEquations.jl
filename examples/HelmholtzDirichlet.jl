# This file calculates scattering from the Helmholtz equation via the adaptive spectral method.
# Δu + k^2u = 0,
# u|Γ = 0,
# u^i = e^{im k x⋅d},
# u = u^i + u^s.
# The normal derivative ∂u/∂n of the entire wave is calculated on the sound-soft boundaries.
# The scattered wave is calculated by convolving [∂u/∂n] with the fundamental solution.
# Then, the total wave is obtained by summing the incident and the scattered waves.

using ApproxFun, SingularIntegralEquations
include("Scatteraux.jl")

k = 50.
ω = 2π
d = (1,-1)
d = d[1]/hypot(d[1],d[2]),d[2]/hypot(d[1],d[2])
ui(x,y) = exp(im*k*(d⋅(x,y)))

# The Helmholtz Green's function, split into singular and nonsingular pieces.
g1(x,y) = -besselj0(k*abs(y-x))/2
g2(x,y) = x == y ? -(log(k/2)+γ)/2/π + im/4 : im/4*hankelh1(0,k*abs(y-x)) - g1(x,y).*logabs(y-x)/π


# A variety of domains.

#dom = Circle(0.0,1/π)∪Interval(-1.0im,1.0)
#dom = ∪(Interval,[-2.5-.5im,-1.5+.5im,-.5-.5im,.5+.5im,1.5-.5im],[-1.5-.5im,-.5+.5im,.5-.5im,1.5+.5im,2.5-.5im])
#dom = ∪(Interval,[-1.0-0.4im,0.1+0.4im,-0.9-0.5im],[-0.1+0.4im,1.0-0.4im,0.9-0.5im])
#dom = ∪(Interval,[-1.0-0.4im,-0.5-0.4im,0.1+0.4im,0.2+0.0im,-1.4-0.75im],[-0.1+0.4im,-0.2+0.0im,1.0-0.4im,0.5-0.4im,1.4-0.75im])
#dom = ∪(Circle,[0.,-1.0im],[0.5,0.25])∪∪(Interval,[-1.5,0.5-1.0im,-0.5+1.0im],[-0.5-1.0im,1.5,0.5+1.0im])
dom = Interval()

sp = Space(dom)
cwsp = CauchyWeight(sp⊗sp,0)
uiΓ,⨍ = Fun(t->ui(real(t),imag(t)),sp),DefiniteLineIntegral(dom)

@time G = GreensFun(g1,cwsp;method=:Cholesky) + GreensFun(g2,sp⊗sp;method=:Cholesky)

@time ∂u∂n = ⨍[G]\uiΓ
println("The length of ∂u∂n is: ",length(∂u∂n))
us(x,y) = -logkernel(g1,∂u∂n,complex(x,y))-linesum(g2,∂u∂n,complex(x,y))
