# This file calculates scattering from the Helmholtz equation via the adaptive spectral method.
# Δu + k^2u = 0,
# u|Γ = 0 (which for now is [-1,1]),
# u^i = e^{im k x⋅d},
# u = u^i + u^s.
# The normal derivative ∂u∂n of the entire wave is calculated on the sound-hard line [-1,1].
# The scattered wave is calculated by convolving ∂u∂n with the fundamental solution.
# Then, the total wave is obtained by summing the incident and the scattered waves.

using ApproxFun,SIE
include("Scatteraux.jl")

k = 10.
ω = 2π
d = (1,-1)
d = d[1]/hypot(d[1],d[2]),d[2]/hypot(d[1],d[2])
ui(x,y) = exp(im*k*(d⋅(x,y)))

    dom = Interval(-1.,1.)
    sp = UltrasphericalSpace{0}(dom)
    x = Fun(identity,dom)
    w = 1/sqrt((dom.b-x)*(x-dom.a))
    uiΓ,H0,S = Fun(x->ui(x,0),sp),Hilbert(dom,0),Σ(dom)

    kd = int(k*(1+length(dom)/2))
    FK0LR = LowRankFun((x,y)->FK0(k*(y-x)),sp,sp,2kd,2kd;maxrank=kd)
    K0,Kim = -FK0LR/2pi,FK0LR/4
    println("The ranks of K0 and Kim are: ",rank(K0),".")
    Kr = LowRankFun((x,y)->(GK0(k.*(y-x))-FK0(k.*(y-x)).*(log(abs(k)/2)+γ))/2pi,sp,sp,2kd,2kd;maxrank=kd)
    println("The rank of Kr is: ",rank(Kr),".")
    L,f = H0[K0*w] + S[Kr*(w/π)] + im*S[Kim*(w/π)],uiΓ

    @time ∂u∂n = L\f
    println("The length of ∂u∂n is: ",length(∂u∂n))

    us(x,y) = Fun(t->-im/4.*hankelh1(0,k.*sqrt((x.-t).^2.+y.^2))*∂u∂n[t],ApproxFun.VectorDomainSpace(sp,length(x)),length(∂u∂n)).coefficients[1:length(x)]


using PyCall
pygui(:tk)
using PyPlot

clf()
x = linspace(-3,3,101);y = [linspace(-2,-.02,51),linspace(.02,2,51)]
xx,yy = x.+0.*y',0.*x.+y'

@time u = ui(xx,yy) + reshape(us(xx[:],yy[:]),size(xx))

line = [dom.a,dom.b]
plot(line,0line,"-k",linewidth=2.0)
contourf(xx,yy,real(u),25)
xlabel("\$x\$");ylabel("\$y\$")
savefig("ScatteringPDEplot.pdf")
tomovie(xx,yy,u,20;plotfunction=contourf,seconds=1)