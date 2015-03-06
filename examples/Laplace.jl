# This file calculates the solution to Laplace's equation via the adaptive spectral method.
# Δu = 0,
# u|Γ = 0,
# u^i = 1/2π*log|z-z_0|,
# u = u^i + u^s.
# The normal derivative ∂u∂n of the solution is calculated on the constant-charge circle.
# The reflected solution is calculated by convolving ∂u∂n with the fundamental solution.
# Then, the total solution is obtained by summing the incident and the scattered waves.

using ApproxFun,SIE

z_0 = 0.0+im
ui(x,y) = 1/2π*SIE.logabs(complex(x,y)-z_0)


    N = 2
    dom = Interval([-2.5,-0.5])∪Interval([0.5,2.5])
#    sp = ApproxFun.PiecewiseSpace(map(ApproxFun.ChebyshevDirichlet{1,1},dom.domains))
    sp = Space(dom)
    wsp = ApproxFun.PiecewiseSpace([JacobiWeight(-.5,-.5,sp.spaces[i]) for i=1:N])

    csp = [CauchyWeight{0}(sp[i]⊗sp[i]) for i=1:N]
    cwsp = [CauchyWeight{0}(sp[i]⊗wsp[i]) for i=1:N]
#=
    dom = Interval([-1.,1.])
    sp = ApproxFun.ChebyshevDirichlet{1,1}(dom)
    wsp = JacobiWeight(-.5,-.5,sp)
=#
    #H0 = Fun(x->ui(x,0.0),sp),0.5Hilbert(wsp,0)

    uiΓ,⨍ = Fun(x->ui(x,0.0),sp),PrincipalValue(wsp)

    g1(x,y) = 1/2
    g3(x,y) = 1/2π*SIE.logabs(y-x)

    G = Array(GreensFun,N,N)
    for i=1:N,j=1:N
        if i == j
            G[i,i] = GreensFun([ProductFun(g1,csp[i])])
        else
            G[i,j] = GreensFun([ProductFun(g3,sp[i],sp[j];method=:convolution)])
        end
    end

    L,f = ⨍[G],uiΓ

    @time ∂u∂n = L\f
    println("The length of ∂u∂n is: ",length(∂u∂n))

#us(x,y) = -1/2π*logkernel(∂u∂n,complex(x,y))


    ∂u∂nv = vec(∂u∂n)
    function us(x,y)
        ret = -1/2π*logkernel(Fun(∂u∂nv[1].coefficients,wsp[1]),complex(x,y))
        for i=2:length(∂u∂nv)
            ret -= 1/2π*logkernel(Fun(∂u∂nv[i].coefficients,wsp[i]),complex(x,y))
        end
        ret
    end
