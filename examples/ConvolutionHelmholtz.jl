# This file calculates scattering from the Helmholtz equation via the adaptive spectral method.
# Δu + k^2u = 0,
# u|Γ = 0,
# u^i = e^{im k x⋅d},
# u = u^i + u^s.
# The normal derivative ∂u/∂n of the entire wave is calculated on the sound-soft boundaries.
# The scattered wave is calculated by convolving [∂u/∂n] with the fundamental solution.
# Then, the total wave is obtained by summing the incident and the scattered waves.

# For periodic domains, the convolution property can be leveraged to make the bandwidth more precise.

using ApproxFun, SingularIntegralEquations
include("Scatteraux.jl")

K = logspace(1,3,25)
ts = Float64[]

for k in K

    ω = 2π
    d = (1,-1)
    d = d[1]/hypot(d[1],d[2]),d[2]/hypot(d[1],d[2])
    ui = (x,y) ->  exp(im*k*(d⋅(x,y)))

    # The Helmholtz Green's function, split into singular and nonsingular pieces.
    g1 = z -> -besselj0(k*abs(2sin(log(z)/2im)))/2
    g2 = z -> abs(z-1) < 5eps() ? -(log(k/2)+γ)/2/π + im/4 : im/4*hankelh1(0,k*abs(2sin(log(z)/2im))) - g1(z).*logabs(abs(2sin(log(z)/2im)))/π

    #g1 = z -> -besselj0(k*sqrt(2*(1-(z+inv(z))/2)))/2
    #g2 = z -> abs(z-1) < 5eps() ? -(log(k/2)+γ)/2/π + im/4 : im/4*hankelh1(0,k*sqrt(2*(1-(z+inv(z))/2))) - g1(z).*logabs(sqrt(2*(1-(z+inv(z))/2)))/π

    # A single circular domain.

    dof = round(Int,2.15(k+30))

    dom = Circle()
    sp = Space(dom)
    uiΓ = Fun(t->ui(real(t),imag(t)),sp,dof)
    ⨍ = DefiniteLineIntegral(sp)
    SI = SingularIntegral(sp,0)

    G1 = Fun(g1,sp,dof)
    G2 = Fun(g2,sp,dof)
    SingleLayer = Convolution(SI,G1) + Convolution(⨍,G2)

    tic()
    ∂u∂n = SingleLayer\uiΓ
    t = toc()
    push!(ts,t)
    println("The length of ∂u∂n is: ",ncoefficients(∂u∂n))
    #us = (x,y) ->  -logkernel(g1,∂u∂n,complex(x,y))-linesum(g2,∂u∂n,complex(x,y))

end

#=
using Plots
pyplot()
plot(K,ts;xscale=:log10,yscale=:log10,label="Time")
plot!(K,0.000001K.^2;xscale=:log10,yscale=:log10,label="\$ \\mathcal{O}(k^2) \$")
xlabel!("\$ k \$")
ylabel!("Execution Time (s)")
savefig("time.pdf")

using Plots
pyplot()
pad!(∂u∂n,ncoefficients(∂u∂n)-2)
plot(1:ncoefficients(G1),abs(coefficients(G1));yscale=:log10,label="\$ G_1(|x-y|) \$",)
plot!(1:ncoefficients(G2),abs(coefficients(G2));yscale=:log10,label="\$ G_2(|x-y|) \$",)
plot!(1:ncoefficients(uiΓ),abs(coefficients(uiΓ));yscale=:log10,label="\$ u^i|_\\Gamma \$",)
plot!(1:ncoefficients(∂u∂n),abs(coefficients(∂u∂n));yscale=:log10,label="\$ \\partial u / \\partial n \$",)

xlabel!("\$ n \$")
ylabel!("Laurent coefficients")
savefig("coefficients.pdf")
=#
