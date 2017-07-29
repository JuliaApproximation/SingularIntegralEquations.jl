# This file calculates the second application in Y.-S. Chan, A. C. Fannjiang, and G. H. Paulino,
# Integral equations with hypersingular kernels -- theory and applications to fracture mechanics,
# Int. J. Eng. Sci., 41:683--720, 2003.

using ApproxFun, SingularIntegralEquations
include("Fractureaux.jl")

d = Segment(-1.,1.)
sp = Space(d)
wsp = JacobiWeight(-0.5,-0.5,sp)
⨍ = DefiniteLineIntegral(wsp)
x = Fun(identity,d)

β = -2.0
G = exp(β*x)

K = GreensFun((x,y)->(1/π*exp(β/2*(y-x))*β^2/2^2*(FK1(β*(y-x)/2)*(log(abs(β)/4) + γ) - GK1(β*(y-x)/2)/2))/sqrt(1-y^2),sp⊗wsp)
K0 = GreensFun((x,y)-> x == y ? abs(β)/(4sqrt(1-y^2)) :
                                exp(β/2*(y-x))*abs(β)/2*besseli(1,abs(β*(y-x)/2))/abs(y-x)/sqrt(1-y^2),
               CauchyWeight(sp⊗wsp,0))
K2 = GreensFun((x,y)->exp(β/2*(y-x))/sqrt(1-y^2),CauchyWeight(sp⊗wsp,2))



B = Dirichlet(d)

L,p = ⨍[K2+K0+K],-Fun(one,sp)

uSIE = [B;L]\[zeros(2),p/G]
@time uSIE = [B;L]\[zeros(2),p/G]
println("The length of uSIE is: ",ncoefficients(uSIE))
println("The extrema of uSIE are: ",extrema(uSIE))
println("The normalized Stress Intensity Factors are: ",(uSIE/(1-x^2)/G(1))(-1),"  ",(uSIE/(1-x^2)/G(-1))(1))
