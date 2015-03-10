# This file calculates the solution to Laplace's equation via the adaptive spectral method.
# Δu = 0,
# u|Γ = 0,
# u^i = 1/2π*log|z-z_0|,
# u = u^i + u^s.
# The normal derivative ∂u∂n of the solution is calculated on the constant-charge circle.
# The reflected solution is calculated by convolving ∂u∂n with the fundamental solution.
# Then, the total solution is obtained by summing the incident and the scattered waves.

using ApproxFun,SIE

#=  SIE plot
    ui(x,y) = 1/2π*(-logabs(complex(x,y)-complex(2.5,1.5)) + logabs(complex(x,y)-complex(-0.6,0.9)) + logabs(complex(x,y)-complex(0.4,-1.2)) - logabs(complex(x,y)-complex(-2.1,-0.5)) + logabs(complex(x,y)-complex(1.7,-0.5)))
    domS = Interval([-1.95+im,-1.+im])∪Interval([-2.+0.05im,-2.+.95im])∪Interval([-1.95+0.im,-1.05+0.im])∪Interval([-1.-0.05im,-1.-.95im])∪Interval([-2.-im,-1.05-im])
    domI = Interval([-1.0im,1.0im])
    domE = Interval([1.-0.95im,1.+0.95im])∪Interval([1.05+im,2.+im])∪Interval([1.05-im,2.-im])∪Interval([1.1-0.0im,1.75+0.0im])
    dom = domS∪domI∪domE
    N = length(dom)
=#

    z_0 = 2.0
    ui(x,y) = 1/2π*logabs(complex(x,y)-z_0)

    N = 20
    r = 2e-3
    cr = exp(im*2π*[1:N]/N)
    crl = (1-2im*r)cr
    crr = (1+2im*r)cr

    dom = Interval([crl[1],crr[1]])
    [dom = dom∪Interval([crl[j],crr[j]]) for j=2:N]

    sp = PiecewiseSpace(map(ChebyshevDirichlet{1,1},dom.domains))
    wsp = PiecewiseSpace([JacobiWeight(-.5,-.5,sp.spaces[i]) for i=1:N])


#=
    sp = Space(dom)
    xid = Fun(identity,sp)

    csp = [CauchyWeight{0}(sp[i]⊗sp[i]) for i=1:N]
    cwsp = [CauchyWeight{0}(sp[i]⊗wsp[i]) for i=1:N]
    uiΓ,⨍ = Fun(t->ui(real(xid[t]),imag(xid[t])),sp),DefiniteLineIntegral(wsp)

    g1(x,y) = 1/2
    g3(x,y) = 1/2π*logabs(y-x)

    G = Array(GreensFun,N,N)
    for i=1:N,j=1:N
        if i == j
            G[i,i] = ProductFun(g1,cwsp[i])
        else
            G[i,j] = ProductFun(g3,sp[i],wsp[j];method=:convolution)
        end
    end

    L,f = ⨍[G],uiΓ
=#
    uiΓ,H0 = chop(depiece([Fun(t->ui(real(t),imag(t)),sp[i],1024) for i=1:N]),eps()),0.5SingularIntegral(wsp,0)
#    uiΓ,H0 = Fun(t->ui(real(xid[t]),imag(xid[t])),sp),0.5Hilbert(wsp,0)
    L,f = H0,uiΓ

    @time ∂u∂n = L\f
    println("The length of ∂u∂n is: ",length(∂u∂n))

    us(x,y) = -1/2π*logkernel(∂u∂n,complex(x,y))
    ut(x,y) = ui(x,y) + us(x,y)
    println("This is the approximate gradient: ",(2π*(ut(1e-5,0.)-ut(-1e-5,0.))/2e-5))
