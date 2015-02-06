# This file calculates scattering from the Helmholtz equation via the adaptive spectral method.
# Δu + k^2u = 0,
# u|Γ = 0 (which for now is [-1,1]),
# u^i = e^{im k x⋅d},
# u = u^i + u^s.
# The normal derivative ∂u∂n of the entire wave is calculated on the sound-hard line [-1,1].
# The scattered wave is calculated by convolving ∂u∂n with the fundamental solution.
# Then, the total wave is obtained by summing the incident and the scattered waves.

using ApproxFun,SIE
include("Scatteraux.jl")

k = 10.
ω = 2π
d = (1,-1)
d = d[1]/hypot(d[1],d[2]),d[2]/hypot(d[1],d[2])
ui(x,y) = exp(im*k*(d⋅(x,y)))

    dom = Interval(-1.,1.)
    sp = Chebyshev(dom)
    wsp = JacobiWeight(-.5,-.5,sp)
    x = Fun(identity,sp)
    uiΓ,H0,S = Fun(x->ui(x,0),sp),Hilbert(dom,0),Σ(dom)

    FK0LR = Fun(x->besselj0(k*x),Chebyshev([-length(dom),length(dom)]))
    FKr = Fun(x->(GK0(k*x)-besselj0(k*x)*(log(abs(k)/2)+γ))/2π,Chebyshev([-length(dom),length(dom)]))

    K0 = ProductFun(-FK0LR/2π,sp,wsp)
    Kim = ProductFun(FK0LR/4π,sp,wsp)
    Kr = ProductFun(FKr/π,sp,wsp)
    L,f = H0[K0] + S[Kr] + im*S[Kim],uiΓ

    @time ∂u∂n = L\f
    println("The length of ∂u∂n is: ",length(∂u∂n))

    us(x,y) = Fun(t->-im/4.*hankelh1(0,k.*sqrt((x.-t).^2.+y.^2))*∂u∂n[t],ApproxFun.ArraySpace(sp,length(x)),length(∂u∂n)).coefficients[1:length(x)]
#=
dom1 = Interval(-2.5,-.5)∪Interval(.5,2.5)
sp1 = Space(dom1)
wsp1 = ApproxFun.PiecewiseSpace([JacobiWeight(-.5,-.5,sp1.spaces[i]) for i=1:length(sp1)])
H1 = Hilbert(wsp1,1)
=#
