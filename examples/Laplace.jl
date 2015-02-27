# This file calculates the solution to Laplace's equation via the adaptive spectral method.
# Δu = 0,
# u|Γ = 0,
# u^i = 1/2π*log|z-z_0|,
# u = u^i + u^s.
# The normal derivative ∂u∂n of the solution is calculated on the constant-charge circle.
# The reflected solution is calculated by convolving ∂u∂n with the fundamental solution.
# Then, the total solution is obtained by summing the incident and the scattered waves.

using ApproxFun,SIE

z_0 = -1.5+1.0im
ui(x,y) = 1/2π*SIE.logabs(complex(x,y)-z_0)


    dom = Interval([-2.5,-0.5])∪Interval([0.5,2.5])
    sp = ApproxFun.PiecewiseSpace(map(ApproxFun.ChebyshevDirichlet{1,1},dom.domains))
    wsp = ApproxFun.PiecewiseSpace([JacobiWeight(-.5,-.5,sp.spaces[i]) for i=1:length(sp)])
    uiΓ,H0 = Fun(x->ui(x,0.0),sp),0.5Hilbert(wsp,0)
    L,f = H0,uiΓ

    @time ∂u∂n = L\f
    println("The length of ∂u∂n is: ",length(∂u∂n))

    ∂u∂nv = vec(∂u∂n)

    function us(x,y)
        ret = -1/2π*logkernel(∂u∂nv[1],complex(x,y))
        for i=2:length(∂u∂nv)
            ret -= 1/2π*logkernel(∂u∂nv[i],complex(x,y))
        end
        ret
    end
