# This file calculates scattering from the gravity Helmholtz equation via the adaptive spectral method.
# Δu + (E+x_2)u = 0,
# u|Γ = 0,
# u^i = Φ(⋅,⋅,E)|Γ
# u = u^i + u^s.
# The normal derivative ∂u/∂n of the entire wave is calculated on the sound-soft boundaries.
# The scattered wave is calculated by convolving ∂u/∂n with the fundamental solution.
# Then, the total wave is obtained by summing the incident and the scattered waves.

using ApproxFun,SIE

E = 5.
ui(x,y) = lhelmfs(complex(x,y),10.0im,E)

# The gravity Helmholtz Green's function.
g3(x,y) = lhelmfs(x,y,E)


    dom = Interval(-4.0+1.5im,4.0)
    sp = Space(dom)
    cwsp = CauchyWeight(sp⊗sp,0)
    uiΓ,⨍ = Fun(t->ui(real(t),imag(t)),sp),DefiniteLineIntegral(dom)

    G1 = LowRankFun(skewProductFun((x,y)->imag(g3(x,y)),sp⊗sp))
    G2 = skewProductFun((x,y)->g3(x,y) + G1[x,y]/G1[0.,0.]/2π*logabs(y-x),sp⊗sp,2^7,2^7+1)
    G1D = G1[0.0,0.0]
    G1 = ProductFun(coefficients(skewProductFun((x,y)->-imag(g3(x,y))/G1D/2,sp⊗sp)),cwsp)
    @time G = GreensFun([G1,G2])

    L,f = ⨍[G],uiΓ

    @time ∂u∂n = L\f
    println("The length of ∂u∂n is: ",length(∂u∂n))
    us(x,y) = -linesum(g3,∂u∂n,complex(x,y))
