# This file calculates scattering from the Helmholtz equation via the adaptive spectral method.
# Δu + k^2u = 0,
# u|Γ = 0,
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

#=
    dom = Interval(-1.,1.)
    sp = Chebyshev(dom)
    wsp = JacobiWeight(-.5,-.5,sp)
    uiΓ,H0,Σ = Fun(x->ui(x,0),sp),Hilbert(dom,0),DefiniteIntegral(dom)

    FK0LR = Fun(x->besselj0(k*x),Chebyshev([-length(dom),length(dom)]))
    FKr = Fun(x->1/2π*besselj0(k*x)*log(abs(x))-bessely0(k*abs(x))/4,Chebyshev([-length(dom),length(dom)]))
    K0 = SymmetricProductFun(-FK0LR/2π,sp,wsp)
    Kim = SymmetricProductFun(FK0LR/4π,sp,wsp)
    Kr = SymmetricProductFun(FKr/π,sp,wsp)
    L,f = H0[K0] + Σ[Kr] + im*Σ[Kim],uiΓ
=#


    dom = Interval(-2.5,-.5)∪Interval(.5,2.5)
    sp = Space(dom)
    wsp = ApproxFun.PiecewiseSpace([JacobiWeight(-.5,-.5,sp.spaces[i]) for i=1:length(sp)])
    uiΓ,H1,H2,Σ1,Σ2 = Fun(x->ui(x,0),sp),Hilbert(wsp[1],0),Hilbert(wsp[2],0),DefiniteIntegral(wsp[1]),DefiniteIntegral(wsp[2])


    # Region 1 -> 1
    # RMS personal note: K011 division by π because FK011*2/π is smooth part of log of Hankel.
    # Kim11 division by π because S1 multiplies by π.
    # Kr11 division by π because S1 multiplies by π.
    FK011 = Fun(x->besselj0(k*x),Chebyshev([-length(dom[1]),length(dom[1])]))
    FKr11 = Fun(x->1/2π*besselj0(k*x)*log(abs(x))-bessely0(k*abs(x))/4,Chebyshev([-length(dom[1]),length(dom[1])]))
    K011 = SymmetricProductFun(-FK011/2π,sp[1],wsp[1])
    Kim11 = SymmetricProductFun(FK011/4π,sp[1],wsp[1])
    Kr11 = SymmetricProductFun(FKr11/π,sp[1],wsp[1])

    # Region 1 -> 2
    F12r = Fun(x->-.25bessely0(k*abs(3.-x)),Chebyshev([-length(dom[1]),length(dom[1])]))
    K12r = SymmetricProductFun(F12r/π,sp[2],wsp[1])
    F12im = Fun(x->.25besselj0(k*abs(3.-x)),Chebyshev([-length(dom[1]),length(dom[1])]))
    K12im = SymmetricProductFun(F12im/π,sp[2],wsp[1])

    # Region 2 -> 1
    F21r = Fun(x->-.25bessely0(k*abs(3.-x)),Chebyshev([-length(dom[1]),length(dom[1])]))
    K21r = SymmetricProductFun(F21r/π,sp[1],wsp[2])
    F21im = Fun(x->.25besselj0(k*abs(3.-x)),Chebyshev([-length(dom[1]),length(dom[1])]))
    K21im = SymmetricProductFun(F21im/π,sp[1],wsp[2])

    # Region 2 -> 2
    FK022 = Fun(x->besselj0(k*x),Chebyshev([-length(dom[2]),length(dom[2])]))
    FKr22 = Fun(x->1/2π*besselj0(k*x)*log(abs(x))-bessely0(k*abs(x))/4,Chebyshev([-length(dom[2]),length(dom[2])]))
    K022 = SymmetricProductFun(-FK022/2π,sp[2],wsp[2])
    Kim22 = SymmetricProductFun(FK022/4π,sp[2],wsp[2])
    Kr22 = SymmetricProductFun(FKr22/π,sp[2],wsp[2])

    L11 = H1[K011] + Σ1[Kr11] + im*Σ1[Kim11]
    L12 = Σ1[K12r] + im*Σ1[K12im]
    L21 = Σ2[K21r] + im*Σ2[K21im]
    L22 = H2[K022] + Σ2[Kr22] + im*Σ2[Kim22]

    L,f = [L11 L21; L12 L22],uiΓ

    @time ∂u∂n = L\f
    println("The length of ∂u∂n is: ",length(∂u∂n))

    ∂u∂nv = vec(∂u∂n)
    us(x,y) = Fun(t->-im/4.*hankelh1(0,k.*sqrt((x.-t).^2.+y.^2))*∂u∂nv[1][t],ApproxFun.ArraySpace(sp[1],length(x)),length(∂u∂nv[1])).coefficients[1:length(x)]+Fun(t->-im/4.*hankelh1(0,k.*sqrt((x.-t).^2.+y.^2))*∂u∂nv[2][t],ApproxFun.ArraySpace(sp[2],length(x)),length(∂u∂nv[2])).coefficients[1:length(x)]
