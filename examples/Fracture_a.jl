# This file calculates the first application in Y.-S. Chan, A. C. Fannjiang, and G. H. Paulino,
# Integral equations with hypersingular kernels -- theory and applications to fracture mechanics,
# Int. J. Eng. Sci., 41:683--720, 2003.

using ApproxFun,SIE


x = Fun(identity)
w = 1/sqrt(1-x^2)
d = domain(x)
d2 = d^2
B = dirichlet(d)
H2 = Hilbert(d,2)
S = Σ(d)
ϵ = 2.0
K = LowRankFun((x,y)->-1./(x+y+2ϵ).^2+12(x+ϵ)./(x+y+2ϵ).^3-12(x+ϵ).^2./(x+y+2ϵ).^4,d2)
L = H2[w] + S[K*(w/π)]
f = -Fun(one)

uSIE = [B,L]\[zeros(2),f]
@time uSIE = [B,L]\[zeros(2),f]
println("The rank of K is: ",rank(K))
println("The length of uSIE is: ",length(uSIE))
println("The extrema of uSIE are: ",extrema(uSIE))
println("The normalized Stress Intensify Factors are: ",(uSIE/(1-x^2))[-1],"  ",(uSIE/(1-x^2))[1])


using PyCall
pygui(:tk)
using PyPlot

clf();m=length(uSIE)
spy([sparse(B[1][1:m]'),sparse(B[2][1:m]'),L[1:m-2,1:m]])
xlim([-.5,m-.5]);ylim([m-.5,-.5]);xlabel("j");ylabel("i")
savefig("Fracture_aspy.pdf")

uSIE *= w
clf();xx = linspace(-1,1,5001)
plot(xx,uSIE[xx],"-k")
xlabel("x");ylabel("u")
savefig("Fracture_aplot.pdf")