using ApproxFun, SingularIntegralEquations, Plots


k = 50.
ω = 2π
d = (1,-1)
d = d[1]/hypot(d[1],d[2]),d[2]/hypot(d[1],d[2])
ui = (x,y) -> exp(im*k*(d[1]*x+d[2]*y))

# The Helmholtz Green's function, split into singular and nonsingular pieces.
g1 = (x,y) -> -besselj0(k*abs(y-x))/2
g2 = (x,y) -> x == y ? -(log(k/2)+γ)/2/π + im/4 : im/4*hankelh1(0,k*abs(y-x)) - g1(x,y).*logabs(y-x)/π



Γ = Curve(Fun(x->x+im*x^3))

sp = Space(Γ)
cwsp = CauchyWeight(sp⊗sp,0)
uiΓ,⨍ = Fun(t->ui(real(t),imag(t)),sp),DefiniteLineIntegral(Γ)

@time G = GreensFun(g1,cwsp;method=:Cholesky) + GreensFun(g2,sp⊗sp;method=:Cholesky)

@time ∂u∂n = ⨍[G]\uiΓ
println("The length of ∂u∂n is: ",ncoefficients(∂u∂n))
us = (x,y) -> -logkernel(g1,∂u∂n,complex(x,y))-linesum(g2,∂u∂n,complex(x,y))

us(0.1,0.2)
