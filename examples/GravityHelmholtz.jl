# This file calculates scattering from the gravity Helmholtz equation via the adaptive spectral method.
# Δu + (E+x_2)u = 0,
# u|Γ = 0,
# u^i = Φ(⋅,⋅,E)|Γ
# u = u^i + u^s.
# The normal derivative ∂u/∂n of the entire wave is calculated on the sound-soft boundaries.
# The scattered wave is calculated by convolving ∂u/∂n with the fundamental solution.
# Then, the total wave is obtained by summing the incident and the scattered waves.

using ApproxFun,SIE
include("Scatteraux.jl")

E = 20.
ω = 2π
ui(x,y) = lhelmfs(complex(x,y),-5.0im,E)

# The gravity Helmholtz Green's function.
g3(x,y) = lhelmfs(x,y,E)


    dom = ∪(Interval,[-8.0+0.0im,2.0],[-2.0+0.0im,8.0])#Interval(-8.0-4.0im,-2.0+2.0im)#∪(Interval,[-8.0-4.0im,2.0+2.0im],[-2.0+2.0im,8.0-4.0im])
    sp = Space(dom)
    cwsp = CauchyWeight(sp⊗sp,0)
    uiΓ,⨍ = Fun(t->ui(real(t),imag(t)),sp),DefiniteLineIntegral(dom)

    @time G = GreensFun(g3,cwsp;method=:unsplit)

    L,f = ⨍[G],uiΓ

    @time ∂u∂n = L\f
    println("The length of ∂u∂n is: ",length(∂u∂n))
    us(x,y) = -linesum(g3,∂u∂n,complex(x,y))
