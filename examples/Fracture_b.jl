# This file calculates the second application in Y.-S. Chan, A. C. Fannjiang, and G. H. Paulino,
# Integral equations with hypersingular kernels -- theory and applications to fracture mechanics,
# Int. J. Eng. Sci., 41:683--720, 2003.

using ApproxFun,SIE
include("Fractureaux.jl")

x = Fun(identity)
w = 1/sqrt(1-x^2)
d = domain(x)
d2 = d^2
B = dirichlet(d)
H0 = Hilbert(d,0)
H2 = Hilbert(d,2)
Σ = DefiniteIntegral(d)

β = -2.0
G = exp(β*x)

K = LowRankFun((x,y)->exp(β/2.*(y-x)).*β^2./2.^2.*(FK1(β.*(y-x)./2).*(log(abs(β)/4) + γ) - GK1(β.*(y-x)./2)./2),d2)
K0 = LowRankFun((x,y)->exp(β/2.*(y-x)).*β^2./2.^2.*FK1(β.*(y-x)./2),d2)
K2 = LowRankFun((x,y)->exp(β/2.*(y-x)),d2)

L = H2[K2*w] + H0[K0*w] + Σ[K*(w/π)]
p = -Fun(one)

uSIE = [B,L]\[zeros(2),p/G]
@time uSIE = [B,L]\[zeros(2),p/G]
println("The rank of K2 is: ",rank(K2)," the rank of K0 is: ",rank(K0),", and the rank of K is: ",rank(K),".")
println("The length of uSIE is: ",length(uSIE))
println("The extrema of uSIE are: ",extrema(uSIE))
println("The normalized Stress Intensity Factors are: ",(uSIE/(1-x^2)/G[1])[-1],"  ",(uSIE/(1-x^2)/G[-1])[1])
