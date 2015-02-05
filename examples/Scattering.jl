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

k = 50.
ω = 2π
d = (1,-1)
d = d[1]/hypot(d[1],d[2]),d[2]/hypot(d[1],d[2])
ui(x,y) = exp(im*k*(d⋅(x,y)))

    dom = Interval(-1.,1.)
    sp = Ultraspherical{0}(dom)
    x = Fun(identity,dom)
    w = 1/sqrt((dom.b-x)*(x-dom.a))
    uiΓ,H0,S = Fun(x->ui(x,0),sp),Hilbert(dom,0),Σ(dom)

#=
    kd = int(k*(1+length(dom)/2))
    FK0LR = LowRankFun((x,y)->FK0(k*(y-x)),sp,sp,2kd,2kd;maxrank=kd)
    K0,Kim = -FK0LR/2pi,FK0LR/4
    println("The ranks of K0 and Kim are: ",rank(K0),".")
    Kr = LowRankFun((x,y)->(GK0(k.*(y-x))-FK0(k.*(y-x)).*(log(abs(k)/2)+γ))/2pi,sp,sp,2kd,2kd;maxrank=kd)
    println("The rank of Kr is: ",rank(Kr),".")
=#
    FK0LR = Fun(x->FK0(k*x),[-2.,2.])
    K0 = ProductFun(-coefficients(ProductFun(FK0LR))/2π,Ultraspherical{0}(),JacobiWeight(-.5,-.5,Ultraspherical{0}()))
    Kim = ProductFun(coefficients(ProductFun(FK0LR))/4π,Ultraspherical{0}(),JacobiWeight(-.5,-.5,Ultraspherical{0}()))
    #K0,Kim = -FK0LR/2pi,FK0LR/4
    #println("The ranks of K0 and Kim are: ",rank(K0),".")
    FKr = Fun(x->(GK0(k*x)-FK0(k*x).*(log(abs(k)/2)+γ))/2π,[-2.,2.])
    Kr = ProductFun(coefficients(ProductFun(FKr))/π,Ultraspherical{0}(),JacobiWeight(-.5,-.5,Ultraspherical{0}()))
    #Kr = LowRankFun((x,y)->(GK0(k.*(y-x))-FK0(k.*(y-x)).*(log(abs(k)/2)+γ))/2pi,sp,sp,2kd,2kd;maxrank=kd)
    #println("The rank of Kr is: ",rank(Kr),".")

    #L,f = H0[K0*w] + S[Kr*(w/π)] + im*S[Kim*(w/π)],uiΓ
    L,f = H0[K0] + S[Kr] + im*S[Kim],uiΓ

    @time ∂u∂n = L\f
    println("The length of ∂u∂n is: ",length(∂u∂n))

    us(x,y) = Fun(t->-im/4.*hankelh1(0,k.*sqrt((x.-t).^2.+y.^2))*∂u∂n[t],ApproxFun.ArraySpace(sp,length(x)),length(∂u∂n)).coefficients[1:length(x)]
