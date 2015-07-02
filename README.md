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

# The Faraday Cage

[Laplace.jl](https://github.com/ApproxFun/SIE.jl/blob/master/examples/Laplace.jl) calculates the solution to the Laplace equation with the origin shielded by infinitesimal plates. The essential lines of code are:

```julia
test
```

![Faraday Cage](https://github.com/ApproxFun/SIE.jl/raw/master/images/FaradayCage.png)


# Acoustic Scattering

[Scattering.jl](https://github.com/ApproxFun/SIE.jl/blob/master/examples/Scattering.jl) and [ScatteringNeumann.jl](https://github.com/ApproxFun/SIE.jl/blob/master/examples/ScatteringNeumann.jl) calculate the solution to the Helmholtz equation with Dirichlet and Neumann boundary conditions. The essential lines of code are:

```julia
test
```

![Helmholtz Scattering](https://github.com/ApproxFun/SIE.jl/raw/master/images/Helmholtz.gif)

[GravityHelmholtz.jl](https://github.com/ApproxFun/SIE.jl/blob/master/examples/GravityHelmholtz.jl) calculates the solution to the gravity Helmholtz equation with Dirichlet boundary conditions. The essential lines of code are:

```julia
test
```

![Gravity Helmholtz Scattering](https://github.com/ApproxFun/SIE.jl/raw/master/images/GravityHelmholtz.gif)

# Riemann--Hilbert Problems

SIE has support for Riemann--Hilbert problems:

```julia
Cp=Cauchy(1)
Cm=Cauchy(-1)
g=Fun(z->1.+.1z-.1z.^(-2),Circle())
u=(Cp-g*Cm)\g-1.
cauchy(u,.1+.1im)
```
