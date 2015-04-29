# This file calculates the solution to Laplace's equation via the adaptive spectral method.
# Δu = 0,
# u|Γ = 0,
# u^i = log|z-z_0|,
# u = u^i + u^s.
# The normal derivative ∂u/∂n of the solution is calculated on the constant-charge boundary Γ.
# The reflected solution is calculated by convolving ∂u/∂n with the fundamental solution.
# Then, the total solution is obtained by summing the incident and the reflected solutions.

using ApproxFun,SIE

#=  SIE plot
    ui(x,y) = -logabs(complex(x,y)-complex(2.5,1.5)) + logabs(complex(x,y)-complex(-0.6,0.9)) + logabs(complex(x,y)-complex(0.4,-1.2)) - logabs(complex(x,y)-complex(-2.1,-0.5)) + logabs(complex(x,y)-complex(1.7,-0.5))
    domS = Interval([-1.95+im,-1.+im])∪Interval([-2.+0.05im,-2.+.95im])∪Interval([-1.95+0.im,-1.05+0.im])∪Interval([-1.-0.05im,-1.-.95im])∪Interval([-2.-im,-1.05-im])
    domI = Interval([-1.0im,1.0im])
    domE = Interval([1.-0.95im,1.+0.95im])∪Interval([1.05+im,2.+im])∪Interval([1.05-im,2.-im])∪Interval([1.1-0.0im,1.75+0.0im])
    dom = domS∪domI∪domE
=#

    z_0 = 2.0
    ui(x,y) = logabs(complex(x,y)-z_0)
    g1(x,y) = 1/2

# Set the domains.
    N = 6
    r = 1e-1
    cr = exp(im*2π*[0:N-1]/N)
    crl = (1-2im*r)cr
    crr = (1+2im*r)cr
    #dom = ∪(Interval,crl,crr)
    #dom = ∪(Circle,cr,ones(length(cr))r)
    dom = ∪(Interval,crl[1:2:end],crr[1:2:end]) ∪ ∪(Circle,cr[2:2:end],ones(length(cr[2:2:end]))r)

    sp = Space(dom)
    cwsp = CauchyWeight(sp⊗sp,0)
    uiΓ,⨍ = Fun(t->ui(real(t),imag(t))+0im,sp),DefiniteLineIntegral(dom)

    @time G = GreensFun(g1,cwsp)

    L,f = ⨍[G],uiΓ
#=
    sp = isa(dom,UnionDomain)? PiecewiseSpace(map(ChebyshevDirichlet{1,1},dom.domains)) : ChebyshevDirichlet{1,1}(dom)
    wsp = JacobiWeight(-.5,-.5,sp)
    uiΓ,H0 = Fun(t->ui(real(t),imag(t)),sp),0.5SingularIntegral(wsp,0)
    L,f = H0,uiΓ
=#
    @time ∂u∂n = L\f
    println("The length of ∂u∂n is: ",length(∂u∂n))

    us(x,y) = -logkernel(∂u∂n,complex(x,y))/2π
    ut(x,y) = ui(x,y) + us(x,y)
    println("This is the approximate gradient: ",((ut(1e-5,0.)-ut(-1e-5,0.))/2e-5))

