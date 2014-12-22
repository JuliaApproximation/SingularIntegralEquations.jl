# This file calculates the third application in Y.-S. Chan, A. C. Fannjiang, and G. H. Paulino,
# Integral equations with hypersingular kernels -- theory and applications to fracture mechanics,
# Int. J. Eng. Sci., 41:683--720, 2003.

using ApproxFun,SIE


x = Fun(identity)
w = 1/sqrt(1-x^2)
d = domain(x)
d2 = d^2
B = [dirichlet(d),neumann(d)]
H2 = Hilbert(d,2)
H4 = Hilbert(d,4)
ϵ = 0.2
L = -6ϵ^2*H4[w] + H2[w]
f = -Fun(one)

uSIE = [B,L]\[zeros(4),f]
@time uSIE = [B,L]\[zeros(4),f]
println("The length of uSIE is: ",length(uSIE))
println("The extrema of uSIE are: ",extrema(uSIE))

test0 = uSIE/(1-x^2)^2
test = Fun(x->test0[x],Ultraspherical{1}(d))
temp = 0.0
[temp+= i*test.coefficients[i] for i=1:length(test)];
println("The normalized generalized Stress Intensity Factors are: ",3ϵ*temp)