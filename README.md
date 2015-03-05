SIE.jl
=================


A Julia package for solving singular integral equations.  The package is still experimental so not designed for general usage at the moment.

![Scattering](https://github.com/ApproxFun/SIE.jl/raw/master/images/scattering.gif)

## Installation

Requires ApproxFun master:

```julia
Pkg.add("ApproxFun")
Pkg.checkout("ApproxFun")
Pkg.clone("https://github.com/ApproxFun/SIE.jl.git")
```

## Usage

```julia
Cp=Cauchy(1)
Cm=Cauchy(-1)
g=Fun(z->1.+.1z-.1z.^(-2),Circle())
u=(Cp-g*Cm)\g-1.
cauchy(u,.1+.1im)
```
