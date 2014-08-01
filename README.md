RiemannHilbert.jl
=================

Usage

`Cp=CauchyOperator(1)
Cm=CauchyOperator(-1)

g=FFun(z->1.+.1z-.1z.^(-2),Circle())
u=(Cp-g*Cm)\g-1.

cauchy(u,.1+.1im)
'