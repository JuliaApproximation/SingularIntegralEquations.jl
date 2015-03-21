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


    domS = ∪(Interval([-1.95+im,-2.+0.05im,-1.95+0.im,-1.-0.05im,-2.-im],[-1.+im,-2.+.95im,-1.05+0.im,-1.-.95im,-1.05-im]))
    domI = Interval([-1.0im,0.25im])∪Circle(0.5im,0.125)
    domE = ∪(Interval([1.-0.95im,1.05+im,1.05-im,1.1-0.0im],[1.+0.95im,2.+im,2.-im,1.75+0.0im]))
    dom = domS∪domI∪domE

    N = length(dom)
    sp = Space(dom)
    cwsp = CauchyWeight{0}(sp⊗sp)
    uiΓ,⨍ = Fun(t->ui(real(t),imag(t)),sp),DefiniteLineIntegral(dom)

    g1(x,y) = -besselj0(k*abs(y-x))/2π
    g2(x,y) = (besselj0(k*abs(y-x))*(im*π/2+logabs(y-x))/2π - bessely0(k*abs(y-x))/4)/π
    g3(x,y) = im/4*hankelh1(0,k*abs(y-x))

    @time G = GreensFun(g1,cwsp) + GreensFun(g2,sp⊗sp)

    L,f = ⨍[G],uiΓ

    @time ∂u∂n = L\f

    ∂u∂nv = vec(∂u∂n)
    wsp = map(space,∂u∂nv)

function us(x,y)
    ret = linesum(Fun(t->-g3(x-real(t),im*(y-imag(t)))*∂u∂nv[1][t],wsp[1],length(∂u∂nv[1])))
    for i=2:N
        ret += linesum(Fun(t->-g3(x-real(t),im*(y-imag(t)))*∂u∂nv[i][t],wsp[i],length(∂u∂nv[i])))
    end
    ret/π
end
@vectorize_2arg Number us
