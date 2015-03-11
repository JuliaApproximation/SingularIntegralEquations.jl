# This file calculates the solution to Laplace's equation via the adaptive spectral method.
# Δu = 0,
# u|Γ = 0,
# u^i = 1/2π*log|z-z_0|,
# u = u^i + u^s.
# The normal derivative ∂u∂n of the solution is calculated on the constant-charge circle.
# The reflected solution is calculated by convolving ∂u∂n with the fundamental solution.
# Then, the total solution is obtained by summing the incident and the scattered waves.

using ApproxFun,SIE

    z_0 = 2.0
    ui(x,y) = 1/2π*logabs(complex(x,y)-z_0)

    dom = Circle(0.0,0.5)#PeriodicInterval()
    sp = Fourier(dom)
    xid = Fun(identity,sp)
    uiΓ,H0 = Fun(t->ui(real(xid[t]),imag(xid[t])),sp),0.5SingularIntegral(sp,0)
    #uiΓ,H0 = Fun(t->ui(.7cos(t),.7sin(t)),sp),0.5SingularIntegral(sp,0)

    L,f = H0,uiΓ

    @time ∂u∂n = L\f
    println("The length of ∂u∂n is: ",length(∂u∂n))


    #us(x,y) = -1/2π*linesum(Fun(z->logabs(z-complex(x,y)).*∂u∂n[z],sp,3length(∂u∂n) ))
    #us(x,y) = -1/2π*linesum(Fun(z->logabs(.7exp(im*z)-complex(x,y)).*∂u∂n[.7exp(im*z)].*0.7,sp,length(∂u∂n) ))

    us(x,y) = -1/2π*logkernel(∂u∂n,complex(x,y))
    #@vectorize_2arg Number us

    ut(x,y) = ui(x,y) + us(x,y)
#    println("This is the approximate gradient: ",(2π*(ut(1e-5,0.)-ut(-1e-5,0.))/2e-5))
