SIE.jl
=================


A Julia package for solving singular integral equations.  The package is still experimental so not designed for general usage at the moment.

## Installation

Requires ApproxFun master:

`Pkg.add("ApproxFun")
Pkg.checkout("ApproxFun")
Pkg.clone("https://github.com/ApproxFun/SIE.jl.git")`

## Usage

`Cp=Cauchy(1)
Cm=Cauchy(-1)
g=Fun(z->1.+.1z-.1z.^(-2),Circle())
u=(Cp-g*Cm)\g-1.
cauchy(u,.1+.1im)
`