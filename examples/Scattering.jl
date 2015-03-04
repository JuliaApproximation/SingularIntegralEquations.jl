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
    cwsp = CauchyWeight{0}(sp⊗wsp)
    uiΓ,⨍ = Fun(x->ui(x,0),sp),PrincipalValue(dom)

    G = ProductFun((x,y)->-besselj0(k*(y-x))/2π,cwsp) + ProductFun((x,y)->(besselj0(k*(y-x))*(im*π/2+log(abs(y-x)))/2π - bessely0(k*abs(y-x))/4)/π,sp,wsp;method=:convolution)
    L,f = ⨍[G],uiΓ

    @time ∂u∂n = L\f
    println("The length of ∂u∂n is: ",length(∂u∂n))
    us(x,y) = Fun(t->-im/4.*hankelh1(0,k.*sqrt((x.-t).^2.+y.^2))*∂u∂n[t],ApproxFun.ArraySpace(sp,length(x)),length(∂u∂n)).coefficients[1:length(x)]
=#


    dom = Interval(-2.5,-.5)∪Interval(.5,2.5)
    sp = Space(dom)
    wsp = ApproxFun.PiecewiseSpace([JacobiWeight(-.5,-.5,sp.spaces[i]) for i=1:length(sp)])
    cwsp = [CauchyWeight{0}(sp[1]⊗wsp[1]) CauchyWeight{0}(sp[2]⊗wsp[2])]
    uiΓ,H1,H2,Σ1,Σ2 = Fun(x->ui(x,0),sp),Hilbert(wsp[1],0),Hilbert(wsp[2],0),DefiniteIntegral(wsp[1]),DefiniteIntegral(wsp[2])
    ⨍1,⨍2 = PrincipalValue(wsp[1]),PrincipalValue(wsp[2])

    g1(x,y) = -besselj0(k*(y-x))/2π
    g2(x,y) = (besselj0(k*(y-x))*(im*π/2+log(abs(y-x)))/2π - bessely0(k*abs(y-x))/4)/π
    g3(x,y) = im/4π*hankelh1(0,k*abs(y-x))

    K11 = ProductFun(g1,cwsp[1];method=:convolution) + ProductFun(g2,sp[1],wsp[1];method=:convolution)
    K12 = ProductFun(g3,sp[2],wsp[1];method=:convolution)
    K21 = ProductFun(g3,sp[1],wsp[2];method=:convolution)
    K22 = ProductFun(g1,cwsp[2];method=:convolution) + ProductFun(g2,sp[2],wsp[2];method=:convolution)

    L11 = ⨍1[K11]
    L12 = ⨍1[K12]
    L21 = ⨍2[K21]
    L22 = ⨍2[K22]


    L,f = [L11 L21; L12 L22],uiΓ

    @time ∂u∂n = L\f
    println("The length of ∂u∂n is: ",length(∂u∂n))
    ∂u∂nv = vec(∂u∂n)
    us(x,y) = Fun(t->-im/4.*hankelh1(0,k.*sqrt((x.-t).^2.+y.^2))*∂u∂nv[1][t],ApproxFun.ArraySpace(sp[1],length(x)),length(∂u∂nv[1])).coefficients[1:length(x)]+Fun(t->-im/4.*hankelh1(0,k.*sqrt((x.-t).^2.+y.^2))*∂u∂nv[2][t],ApproxFun.ArraySpace(sp[2],length(x)),length(∂u∂nv[2])).coefficients[1:length(x)]
