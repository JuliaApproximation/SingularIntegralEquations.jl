# SIE.jl

An experimental Julia package for solving singular integral equations.

# Installation

Requires ApproxFun master:

```julia
Pkg.add("ApproxFun")
Pkg.checkout("ApproxFun","development")
Pkg.clone("https://github.com/ApproxFun/SIE.jl.git")
Pkg.build("SIE")
using ApproxFun, SIE
```

# Acoustic Scattering

[Scattering.jl](https://github.com/ApproxFun/SIE.jl/blob/master/examples/Scattering.jl) and [ScatteringNeumann.jl](https://github.com/ApproxFun/SIE.jl/blob/master/examples/ScatteringNeumann.jl) calculate the solution to the Helmholtz equation with Dirichlet and Neumann boundary conditions. The essential lines of code are:

```julia
k = 50 # Set wavenumber and fundamental solution for Helmholtz equation
g1(x,y) = -besselj0(k*abs(y-x))/2
g2(x,y) = x == y ? -(log(k/2)+γ)/2/π + im/4 : im/4*hankelh1(0,k*abs(y-x)) - g1(x,y).*logabs(y-x)/π

ui(x,y) = exp(im*k*(x-y)/sqrt(2)) # Incident plane wave at 45°

dom = Interval() # Set the domain
sp = Space(dom) # Canonical space on the domain
⨍ = DefiniteLineIntegral(dom) # Line integration functional
uiΓ = Fun(t->ui(real(t),imag(t)),sp) # Incident wave on Γ

G = GreensFun(g1,CauchyWeight(sp⊗sp,0)) + GreensFun(g2,sp⊗sp) # Instantiate the fundamental solution

∂u∂n = ⨍[G]\uiΓ # Solve for the density

us(x,y) = -logkernel(g1,∂u∂n,complex(x,y))-linesum(g2,∂u∂n,complex(x,y)) # Represent the scattered field
```

![Helmholtz Scattering](https://github.com/ApproxFun/SIE.jl/raw/master/images/Helmholtz.gif)

[GravityHelmholtz.jl](https://github.com/ApproxFun/SIE.jl/blob/master/examples/GravityHelmholtz.jl) calculates the solution to the gravity Helmholtz equation with Dirichlet boundary conditions.

![Gravity Helmholtz Scattering](https://github.com/ApproxFun/SIE.jl/raw/master/images/GravityHelmholtz.gif)


# The Faraday Cage

[Laplace.jl](https://github.com/ApproxFun/SIE.jl/blob/master/examples/Laplace.jl) calculates the solution to the Laplace equation with the origin shielded by infinitesimal plates centred at the Nth roots of unity. The essential lines of code are:

```julia
ui(x,y) = logabs(complex(x,y)-2)     # Single source at (2,0) of strength 2π

N,r = 10,1e-1
cr = exp(im*2π*[0:N-1]/N)
crl,crr = (1-2im*r)cr,(1+2im*r)cr
dom = ∪(Interval,crl,crr) # Set the shielding domain

sp = Space(dom)                      # Canonical space on the domain
⨍ = DefiniteLineIntegral(dom)        # Line integration functional
uiΓ = Fun(t->ui(real(t),imag(t)),sp) # Action of source on shields

G = GreensFun((x,y)->1/2,CauchyWeight(sp⊗sp,0)) # Instantiate the fundamental solution

# The first column augments the system for global unknown constant charge φ0
# The first row ensure constant charge φ0 on all plates
φ0,∂u∂n=vec([0 ⨍;1 ⨍[G]]\Any[0.,uiΓ]) # Solve for the density

us(x,y) = -logkernel(∂u∂n,complex(x,y))/2 # Represent the scattered field
```

![Faraday Cage](https://github.com/ApproxFun/SIE.jl/raw/master/images/FaradayCage.png)


# Riemann–Hilbert Problems

SIE has support for Riemann–Hilbert problems and Wiener–Hopf factorizations.  [Wiener-Hopf.jl](https://github.com/ApproxFun/SIE.jl/blob/master/examples/Wiener-Hopf.jl) uses the Winer–Hopf factorization to calculate the UL decomposition of a scalar and a block Toeplitz operator.  The essential lines of code in the matrix case are:

```julia
G=Fun(z->[-1 -3; -3 -1]/z +
         [ 2  2;  1 -3] +
         [ 2 -1;  1  2]*z,Circle())

C  = Cauchy(-1)
V  = V=(I+(I-G)*C)\(G-I)

L  = ToeplitzOperator(inv(I+C*V))
U  = ToeplitzOperator(I+V+C*V)
```

# References

R. M. Slevinsky & S. Olver, <a href="http://arxiv.org/abs/1507.00596">A fast and well-conditioned spectral method for singular integral equations</a>, arXiv:1507.00596, 2015.
