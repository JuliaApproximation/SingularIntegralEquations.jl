# This file calculates the second application in Y.-S. Chan, A. C. Fannjiang, and G. H. Paulino,
# Integral equations with hypersingular kernels -- theory and applications to fracture mechanics,
# Int. J. Eng. Sci., 41:683--720, 2003.

using ApproxFun,SIE
include("Fractureaux.jl")

d = Interval(-1.,1.)
sp = Space(d)
wsp = JacobiWeight(-.5,-.5,sp)
⨍ = PrincipalValue(wsp)
x = Fun(identity,d)

β = -2.0
G = exp(β*x)

K = ProductFun((x,y)->1/π*exp(β/2*(y-x))*β^2/2^2*(FK1(β*(y-x)/2)*(log(abs(β)/4) + γ) - GK1(β*(y-x)/2)/2),sp,wsp;method=:convolution)
K0 = ProductFun((x,y)->exp(β/2*(y-x))*abs(β)/2*besseli(1,abs(β*(y-x)/2))/abs(y-x),CauchyWeight{0}(sp⊗wsp))
K2 = ProductFun((x,y)->exp(β/2*(y-x)),CauchyWeight{2}(sp⊗wsp))

B = dirichlet(d)
L,p = ⨍[K2+K0+K],-Fun(one,sp)

uSIE = [B,L]\[zeros(2),p/G]
@time uSIE = [B,L]\[zeros(2),p/G]
println("The length of uSIE is: ",length(uSIE))
println("The extrema of uSIE are: ",extrema(uSIE))
println("The normalized Stress Intensity Factors are: ",(uSIE/(1-x^2)/G[1])[-1],"  ",(uSIE/(1-x^2)/G[-1])[1])
