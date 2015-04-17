# This file calculates scattering from the Helmholtz equation via the adaptive spectral method.
# Δu + k^2u = 0,
# u|Γ = 0,
# u^i = e^{im k x⋅d},
# u = u^i + u^s.
# The normal derivative ∂u/∂n of the entire wave is calculated on the sound-soft boundaries.
# The scattered wave is calculated by convolving ∂u/∂n with the fundamental solution.
# Then, the total wave is obtained by summing the incident and the scattered waves.

using ApproxFun,SIE
include("Scatteraux.jl")

k = 50.
ω = 2π
d = (1,-1)
d = d[1]/hypot(d[1],d[2]),d[2]/hypot(d[1],d[2])
ui(x,y) = exp(im*k*(d⋅(x,y)))

# The Helmholtz Green's function, split into singular and nonsingular pieces.
g1(x,y) = -besselj0(k*abs(y-x))/2
g2(x,y) = x == y ? -0.6041667250259302 + im/4 : besselj0(k*abs(y-x))*(im*π/2+logabs(y-x))/2π - bessely0(k*abs(y-x))/4
g3(x,y) = im/4*hankelh1(0,k*abs(y-x))


# A variety of domains.

    dom = Circle(0.0,1/π)∪Interval(-1.im,1.0)
#=
    N = 6
    r = rand(2N+1)
    cr = cumsum(r)
    ccr = -3+(cr-cr[1])*6/(cr[end]-cr[1]) # For a nice plot, try: [-3.0,-2.4710248798864565,-1.7779535080542614,-0.999257770563108,-0.9160576190726175,-0.5056650643725802,0.7258681480228484,1.2291671942613505,1.3417993440008456,1.485081132919861,1.7601585357456848,2.9542404467603642,3.0]
    dom = ∪(Interval(ccr+(3-ccr[end-1])/2)[1:2:end])
=#
    #dom = ∪(Interval([-2.5-.5im,-1.5+.5im,-.5-.5im,.5+.5im,1.5-.5im],[-1.5-.5im,-.5+.5im,.5-.5im,1.5+.5im,2.5-.5im]))
    #dom = ∪(Interval([-1.0-0.4im,0.1+0.4im,-0.9-0.5im],[-0.1+0.4im,1.0-0.4im,0.9-0.5im]))
    #dom = ∪(Interval([-1.0-0.4im,-0.5-0.4im,0.1+0.4im,0.2+0.0im,-1.4-0.75im],[-0.1+0.4im,-0.2+0.0im,1.0-0.4im,0.5-0.4im,1.4-0.75im]))
    #dom = Circle(0.,0.5)∪∪(Interval([-1.5,0.5-1.0im],[-0.5-1.0im,1.5]))

    sp = Space(dom)
    cwsp = CauchyWeight(sp⊗sp,0)
    uiΓ,⨍ = Fun(t->ui(real(t),imag(t)),sp),DefiniteLineIntegral(dom)

    @time G = GreensFun(g1,cwsp) + GreensFun(g2,sp⊗sp)

    L,f = ⨍[G],uiΓ

    @time ∂u∂n = L\f
    println("The length of ∂u∂n is: ",length(∂u∂n))
    us(x,y) = -linesum(g3,∂u∂n,complex(x,y))
